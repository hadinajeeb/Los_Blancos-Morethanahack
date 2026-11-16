"""Train the possession model on external matches only.

    python train.py --out runs/seqmodel/<name> --train dfl --val-names DFL-MAT-J03WMX,DFL-MAT-J03WR9 [--also sc metrica]

Training data: clean external matches, degraded on the fly to look like the target's automatic
tracking (degrade.py). Validation: held-out external matches, degraded with fixed seeds, plus a
mild copy. The target match is never read here.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from degrade import degrade, gap_aux, sample_cfg, spell_weights, time_warp
from model import PossNet, flip_y, predict, swap_teams

OPT = dict(spell_w=0.0, warp=0.0, scale=0.0, domtok=False, bnd=0.0, degv=1, sc_mask=False)  # set before the pool forks
DIL = (1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64) if C.FPS < 20 else (1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128)
CLEAN = {}  # name -> canonical clean match (filled before the pool forks)
KIND = {}   # name -> dataset kind


def _version(args):
    name, seed, strength, profile, *rest = args
    ver = rest[0] if rest else OPT["degv"]
    m = CLEAN[name]
    rng = np.random.default_rng(seed)
    kind = KIND[name]
    train_aug = strength != "mild" and seed < 700000   # validation seeds start at 777000
    if train_aug and kind != "sc" and OPT["warp"] > 0:
        m = time_warp(m, float(np.exp(rng.uniform(-OPT["warp"], OPT["warp"]))))
    if train_aug and kind != "sc" and OPT["scale"] > 0:
        sx, sy = np.exp(rng.uniform(-OPT["scale"], OPT["scale"], 2)).astype(np.float32)
        m = dict(m)
        for k in ("home", "away", "ball"):
            m[k] = m[k] * np.array([sx, sy], np.float32)
    if kind == "sc":
        d = dict(m)  # already automatic tracking: use as it is
    elif strength == "mild":
        d = degrade(m, rng, cfg=sample_cfg(rng, 0.0), resim_ball=True, do_players=False, do_ball_drop=False)
    else:
        d = degrade(m, rng, cfg=sample_cfg(rng, strength, profile, ver), resim_ball=True)
    w = (d["label"] != C.X).astype(np.float32)
    if train_aug:
        w = w * spell_weights(d["label"], OPT["spell_w"])
    if kind == "sc":  # broadcast cut: nothing tracked, nothing to learn from
        none = np.isnan(d["home"][..., 0]).all(1) & np.isnan(d["away"][..., 0]).all(1)
        w[none] = 0.0
        if OPT["sc_mask"]:
            w[np.isnan(d["ball"][:, 0])] = 0.0
    # boundary targets: 1 within about 0.08 s of a label change; column 0 direct H<->A, column 1 any change
    lab = d["label"]
    ch = np.flatnonzero((lab[1:] != lab[:-1]) & (lab[1:] != C.X) & (lab[:-1] != C.X)) + 1
    bt = np.zeros((len(lab), 2), np.float32)
    r = max(1, int(round(0.08 * C.FPS)))
    for c in ch:
        bt[max(0, c - r):c + r + 1, 1] = 1.0
        if lab[c] < 2 and lab[c - 1] < 2:
            bt[max(0, c - r):c + r + 1, 0] = 1.0
    return dict(name=name, kind=kind, bt=bt, dom={"dfl": 0, "metrica": 1, "sc": 2}[kind] if OPT["domtok"] else 0, home=d["home"], away=d["away"], ball=d["ball"], aux=gap_aux(d["ball"]),
                label=np.minimum(d["label"], 2).astype(np.int64), w=w, half=d["half"], gt=d["label"])


def to_dev(v, dev):
    return {k: (torch.from_numpy(x).to(dev) if isinstance(x, np.ndarray) and k not in ("half", "gt") else x) for k, x in v.items()}


def evaluate(model, versions, dev, tta=False, c_ha=4.0, c_d=4.0, min_len=0, return_prob=False):
    rows, probs = [], []
    for v in versions:
        prob = np.zeros((len(v["gt"]), 3), np.float32)
        prob[:, 2] = 1.0
        for a, b in C.segments_by_half(v):
            p = predict(model, v["home"][a:b], v["away"][a:b], v["ball"][a:b], v["aux"][a:b], tta=tta, dom=v["dom"])
            prob[a:b] = p.cpu().numpy()
        path = np.full(len(prob), C.X, np.int8)
        for a, b in C.segments_by_half(v):
            path[a:b] = C.decode(prob[a:b], c_ha, c_d, min_len)
        g = v["gt"] != C.X
        sc = C.score(v["gt"], path, v["half"])
        sc["acc"] = float((prob.argmax(1)[g] == v["gt"][g]).mean())
        sc["name"] = v["name"]; sc["tag"] = v.get("tag", "")
        rows.append(sc)
        probs.append(prob)
    return (rows, probs) if return_prob else rows


def summarise(rows):
    out = {}
    for tag in sorted({r["tag"] for r in rows}):
        rr = [r for r in rows if r["tag"] == tag]
        out[tag] = {k: float(np.mean([r[k] for r in rr])) for k in ("spell_f1", "macro_f1", "acc", "n_gt", "n_pred")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", nargs="+", default=["dfl"])            # datasets to train on
    ap.add_argument("--val-names", default="")                         # comma list of match names held out
    ap.add_argument("--val-sets", nargs="*", default=[])               # whole datasets used only for validation
    ap.add_argument("--metrica-label", default="label")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=768)
    ap.add_argument("--versions", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--smooth", type=float, default=0.15)
    ap.add_argument("--w-sc", type=float, default=1.0)
    ap.add_argument("--w-metrica", type=float, default=1.0)
    ap.add_argument("--no-ball", action="store_true")
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--spell-w", type=float, default=0.0)
    ap.add_argument("--domtok", action="store_true")
    ap.add_argument("--video-mix", action="store_true")
    ap.add_argument("--bnd", type=float, default=0.0)   # weight of the boundary-head loss; 0 = no head
    ap.add_argument("--degv", type=int, default=1)       # degradation version (2 = tightened on more target input statistics)
    ap.add_argument("--sc-mask-noball", action="store_true")  # SkillCorner: no loss where its ball is missing (its D there is circular)
    ap.add_argument("--kink", action="store_true")       # extra ball velocity-change inputs
    ap.add_argument("--pseudo", default="")       # npz from make_pseudo.py: extra training sequences with pseudo-labels
    ap.add_argument("--w-pseudo", type=float, default=1.0)
    ap.add_argument("--init", default="")         # checkpoint to start from
    ap.add_argument("--warp", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=0.0)
    ap.add_argument("--ch", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--eval-every", type=int, default=1500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    OPT.update(spell_w=a.spell_w, warp=a.warp, scale=a.scale, domtok=a.domtok, bnd=a.bnd, degv=a.degv, sc_mask=a.sc_mask_noball)
    if C.FPS == 25.0 and a.T == 768:
        a.T = 1536
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    val_names = set(filter(None, a.val_names.split(",")))

    t0 = time.time()
    train_names, val_list = [], []
    for ds in sorted(set(a.train) | set(a.val_sets)):
        for path in C.list_matches(ds):
            kw = dict(label_key=a.metrica_label) if ds == "metrica" else {}
            m = C.load_external(path, **kw)
            CLEAN[m["name"]] = m; KIND[m["name"]] = ds
            if ds in a.train and m["name"] not in val_names:
                train_names.append(m["name"])
            else:
                val_list.append(m["name"])
    print(f"loaded {len(CLEAN)} matches in {time.time()-t0:.0f}s; train {len(train_names)} val {len(val_list)}", flush=True)

    jobs = []
    for i, n in enumerate(train_names):
        if KIND[n] == "sc":
            jobs.append((n, 0, 1.0, "provided"))
        else:
            for k in range(a.versions):
                jobs.append((n, a.seed * 1000 + i * 37 + k, "mild" if k == 0 else (0.5 if k % 4 == 1 else 1.0), "video" if (a.video_mix and k % 2 == 0) else "provided"))
    vjobs = []
    for i, n in enumerate(val_list):
        if KIND[n] == "sc":
            vjobs.append((n, 0, 1.0, "provided", "asis"))
        else:
            vjobs += [(n, 777000 + i * 10 + k, 1.0, "provided", "degraded") for k in range(2)] + [(n, 778000 + i, "mild", "provided", "mild")]
            vjobs += [(n, 779000 + i * 10 + k, 1.0, "video", "video") for k in range(2)]
    with Pool(a.procs) as pool:
        tv = pool.map(_version, jobs, chunksize=1)
        vv = pool.map(_version, [j[:4] for j in vjobs], chunksize=1)
    for v, j in zip(vv, vjobs):
        v["tag"] = f"{KIND[v['name']]}-{j[4]}"
    print(f"made {len(tv)} train versions, {len(vv)} val versions in {time.time()-t0:.0f}s", flush=True)
    dsw = {"dfl": 1.0, "metrica": a.w_metrica, "sc": a.w_sc, "pseudo": a.w_pseudo * a.versions}
    if a.pseudo:
        z = np.load(a.pseudo)
        for i in range(int(z["n"])):
            tv.append(dict(name=str(z["names"][i]), kind="pseudo", dom=0, **{k: z[f"{i}_{k}"] for k in ("home", "away", "ball", "aux", "label", "w", "half")},
                           gt=z[f"{i}_gt"], bt=np.zeros((len(z[f"{i}_gt"]), 2), np.float32)))
        print(f"added {int(z['n'])} pseudo-labelled sequences", flush=True)
    tv = [to_dev(v, dev) for v in tv]
    vv = [to_dev(v, dev) for v in vv]
    # window index: (version, start range) per half segment, sampled in proportion to weight x length
    segs, wts = [], []
    for vi, v in enumerate(tv):
        spans = [(0, len(v["label"]))] if v["kind"] == "pseudo" else C.segments_by_half(v)
        for s, e in spans:
            if e - s > a.T:
                segs.append((vi, s, e - a.T)); wts.append((e - s) * dsw[v["kind"]] / (1 if v["kind"] == "sc" else a.versions))
    wts = np.array(wts) / np.sum(wts)
    rng = np.random.default_rng(a.seed)

    model = PossNet(use_ball=not a.no_ball, drop=a.drop, ch=a.ch, dilations=DIL, bnd=a.bnd > 0, kink=a.kink).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev, weights_only=True)["model"], strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    log = dict(args=vars(a), train=train_names, val=val_list, evals=[])
    model.train()
    run = 0.0
    for step in range(1, a.steps + 1):
        pick = rng.choice(len(segs), size=a.batch, p=wts)
        hb, ab, bb, xb, yb, wb, db, tb = [], [], [], [], [], [], [], []
        for si in pick:
            vi, lo, hi = segs[si]
            s = int(rng.integers(lo, hi)); v = tv[vi]; sl = slice(s, s + a.T)
            hb.append(v["home"][sl]); ab.append(v["away"][sl]); bb.append(v["ball"][sl]); xb.append(v["aux"][sl])
            yb.append(v["label"][sl]); wb.append(v["w"][sl]); db.append(v["dom"]); tb.append(v["bt"][sl])
        home, away, ball, aux = torch.stack(hb), torch.stack(ab), torch.stack(bb), torch.stack(xb)
        y, w = torch.stack(yb), torch.stack(wb)
        # symmetry augmentation, half of the batch each
        h = a.batch // 2
        sh, sa, sb, sx = swap_teams(home[:h], away[:h], ball[:h], aux[:h])
        ys = y[:h].clone(); ys[y[:h] == 0] = 1; ys[y[:h] == 1] = 0
        home, away, ball, aux, y = torch.cat([sh, home[h:]]), torch.cat([sa, away[h:]]), torch.cat([sb, ball[h:]]), torch.cat([sx, aux[h:]]), torch.cat([ys, y[h:]])
        q = slice(0, None, 2)
        fh, fa, fb, fx = flip_y(home[q], away[q], ball[q], aux[q])
        home[q], away[q], ball[q], aux[q] = fh, fa, fb, fx
        logits, bl = model(home, away, ball, aux, dom=torch.tensor(db, device=dev), with_bnd=True)
        ce = (F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1), reduction="none") * w.reshape(-1)).sum() / w.sum().clamp(min=1)
        lp = F.log_softmax(logits, -1)
        tm = ((lp[:, 1:] - lp[:, :-1].detach()) ** 2).clamp(max=16).mean()
        loss = ce + a.smooth * tm
        if bl is not None:
            bt = torch.stack(tb)
            m = (w > 0).float().unsqueeze(-1)
            bce = F.binary_cross_entropy_with_logits(bl, bt, pos_weight=torch.tensor(5.0, device=dev), reduction="none")
            loss = loss + a.bnd * (bce * m).sum() / m.sum().clamp(min=1) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * ce.item() if step > 1 else ce.item()
        if step % 100 == 0:
            print(f"step {step} ce {run:.4f} {time.time()-t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            rows = evaluate(model, vv, dev)
            sm = summarise(rows)
            log["evals"].append(dict(step=step, summary=sm, rows=rows))
            print("EVAL", step, json.dumps(sm), flush=True)
            model.train()
            torch.save(dict(model=model.state_dict(), args=dict(vars(a), fps=C.FPS, dilations=list(DIL))), f"{a.out}/model.pt")
            json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
