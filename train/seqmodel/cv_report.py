"""Aggregate cross-validation posteriors (cv_eval.py) into scores.

    SEQ_FPS=<fps of the runs> python cv_report.py --glob 'runs/seqmodel/cv/v0_f*.npz' [--grid] [--c-ha 4 --c-d 4 --min-s 0 --beta 0] [--json out.json]

Per condition: official spellF1 and macroF1 per match (seeds of one match averaged), then mean,
standard deviation, minimum and maximum over the matches; spell recall and precision by duration
bucket (own greedy matching at IoU 0.5, analysis only); and label-free statistics of the
predictions (entropy, switch rate, spell lengths) for the domain-gap comparison.
"""
import argparse
import glob
import itertools
import json
from multiprocessing import Pool

import numpy as np

import common as C
from errors import match, spells

BINS = [0, 1, 2, 4, 8, 16, 1e9]


def label_free_stats(prob, path, half, fps):
    g = half > 0
    p = np.clip(prob[g], 1e-6, 1)
    ent = -(p * np.log(p)).sum(1)
    pg = path[g]
    sw = np.flatnonzero(np.diff(pg) != 0)
    st = C.spell_stats(pg, fps)
    mins = g.sum() / fps / 60
    return dict(entropy=float(ent.mean()), unsure=float((p.max(1) < 0.9).mean()), switches_per_min=float(len(sw) / mins),
                poss_per_90=float(st["n_poss"] / mins * 90), d_share=float(st["shares"][2]), poss_p10=st["poss_pct"][0], poss_p50=st["poss_pct"][2], poss_p90=st["poss_pct"][4],
                under1=st["under1"], under2=st["under2"])


def _one(args):
    prob, bnd, gt, half, fps, c_ha, c_d, min_s, beta, full, gt25, half25 = args
    path = np.full(len(gt), C.X, np.int8)
    for a, b in C.segments_by_half(dict(half=half)):
        path[a:b] = C.decode(prob[a:b], c_ha, c_d, int(round(min_s * fps)), None if bnd is None else bnd[a:b], beta)
    if gt25 is not None:   # model ran at 12.5 fps: repeat each label and score against the 25 fps labels
        path = np.repeat(path, 2)[:len(gt25)]
        if len(path) < len(gt25):
            path = np.concatenate([path, np.repeat(path[-1:], len(gt25) - len(path))])
        prob = np.repeat(prob, 2, axis=0)[:len(gt25)] if full else prob
        if full and len(prob) < len(gt25):
            prob = np.concatenate([prob, np.repeat(prob[-1:], len(gt25) - len(prob), axis=0)])
        gt, half, fps = gt25, half25, 25.0
    sc = C.score(gt, path, half)
    if not full:
        return sc["spell_f1"], sc["macro_f1"]
    gs, ps = spells(gt, half), spells(path, half)
    gi, pj = match(gs, ps)
    gl = np.array([(b - a) / fps for a, b, _ in gs]); gm = np.array([i in gi for i in range(len(gs))])
    pl = np.array([(b - a) / fps for a, b, _ in ps]); pm_ = np.array([j in pj for j in range(len(ps))])
    rec = [(int(gm[(gl >= lo) & (gl < hi)].sum()), int(((gl >= lo) & (gl < hi)).sum())) for lo, hi in zip(BINS[:-1], BINS[1:])]
    prec = [(int(pm_[(pl >= lo) & (pl < hi)].sum()), int(((pl >= lo) & (pl < hi)).sum())) for lo, hi in zip(BINS[:-1], BINS[1:])]
    g = gt != C.X
    alive = (gt < 2) & g
    both = alive & (path < 2)
    return dict(spell=sc["spell_f1"], macro=sc["macro_f1"], n_pred=sc["n_pred"], n_gt=sc["n_gt"], rec=rec, prec=prec,
                acc=float((path[g] == gt[g]).mean()), alive_acc=float(((path[g] < 2) == (gt[g] < 2)).mean()), team_acc=float((path[both] == gt[both]).mean()),
                lf=label_free_stats(prob, path, half, fps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--c-ha", type=float, default=None)
    ap.add_argument("--c-d", type=float, default=None)
    ap.add_argument("--min-s", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--json", default="")
    ap.add_argument("--only", default="")   # comma list of match-name suffixes to keep, e.g. WN1,WOH
    a = ap.parse_args()
    ent = []
    for f in sorted(glob.glob(a.glob)):
        z = np.load(f)
        fps = float(z["fps"])
        for i in range(int(z["n"])):
            ent.append(dict(name=str(z["names"][i]), tag=str(z["tags"][i]), prob=z[f"{i}_prob"], gt=z[f"{i}_gt"], half=z[f"{i}_half"],
                            bnd=z[f"{i}_bnd"] if f"{i}_bnd" in z.files else None, fps=fps))
    if a.only:
        keep = tuple(a.only.split(","))
        ent = [e for e in ent if e["name"].endswith(keep)]
    assert ent, "no files"
    full_rate = {}
    for e in ent:
        if e["fps"] < 20:
            if e["name"] not in full_rate:
                z = np.load(f"{C.P}/external/dfl/common/{e['name']}.npz")
                full_rate[e["name"]] = (z["label"].astype(np.uint8), z["half"].astype(np.uint8))
            e["gt25"], e["half25"] = full_rate[e["name"]]
        else:
            e["gt25"] = e["half25"] = None
    fps = ent[0]["fps"]
    k = fps / 12.5                       # switch costs scale with the frame rate
    c_ha = a.c_ha if a.c_ha is not None else 4.0 * k
    c_d = a.c_d if a.c_d is not None else 4.0 * k
    has_b = ent[0]["bnd"] is not None
    tags = sorted({e["tag"] for e in ent})
    names = sorted({e["name"] for e in ent})
    print(f"{len(ent)} sequences, {len(names)} matches, fps {fps}, boundary head {has_b}")
    with Pool(96) as pool:
        if a.grid:
            grid = list(itertools.product([2 * k, 4 * k, 8 * k], [4 * k], [0.0, 0.5], [0.0, 0.5, 1.0, 2.0] if has_b else [0.0]))
            res = pool.map(_one, [(e["prob"], e["bnd"], e["gt"], e["half"], fps, *g, False, e["gt25"], e["half25"]) for g in grid for e in ent], chunksize=2)
            res = np.array(res).reshape(len(grid), len(ent), 2)
            rows = []
            for gi, g in enumerate(grid):
                r = dict(c_ha=g[0], c_d=g[1], min_s=g[2], beta=g[3])
                for t in tags:
                    idx = [i for i, e in enumerate(ent) if e["tag"] == t]
                    r[t] = round(float(res[gi, idx, 0].mean()), 4)
                r["mean3"] = round(float(np.mean([r[t] for t in ("deg2", "video2", "rules_deg") if t in r])), 4)
                rows.append(r)
            rows.sort(key=lambda r: -r["mean3"])
            print("decode grid, sorted by the mean of deg2, video2, rules_deg (spellF1):")
            for r in rows:
                print("  ", json.dumps(r))
            return
        res = pool.map(_one, [(e["prob"], e["bnd"], e["gt"], e["half"], fps, c_ha, c_d, a.min_s, a.beta, True, e["gt25"], e["half25"]) for e in ent], chunksize=1)
    out = dict(decode=dict(c_ha=c_ha, c_d=c_d, min_s=a.min_s, beta=a.beta), fps=fps, tags={})
    for t in tags:
        per = {}
        for n in names:
            rr = [r for r, e in zip(res, ent) if e["tag"] == t and e["name"] == n]
            per[n] = dict(spell=float(np.mean([r["spell"] for r in rr])), macro=float(np.mean([r["macro"] for r in rr])),
                          n_pred=float(np.mean([r["n_pred"] for r in rr])), n_gt=float(np.mean([r["n_gt"] for r in rr])))
        rr = [r for r, e in zip(res, ent) if e["tag"] == t]
        sp = np.array([per[n]["spell"] for n in names]); ma = np.array([per[n]["macro"] for n in names])
        rec = np.array([r["rec"] for r in rr]).sum(0); prec = np.array([r["prec"] for r in rr]).sum(0)
        lf = {k2: float(np.mean([r["lf"][k2] for r in rr])) for k2 in rr[0]["lf"]}
        out["tags"][t] = dict(spell_mean=float(sp.mean()), spell_std=float(sp.std(ddof=1)), spell_min=float(sp.min()), spell_max=float(sp.max()),
                              macro_mean=float(ma.mean()), per_match=per, recall=(rec[:, 0] / np.maximum(rec[:, 1], 1)).round(3).tolist(),
                              precision=(prec[:, 0] / np.maximum(prec[:, 1], 1)).round(3).tolist(), n_gt_bucket=rec[:, 1].tolist(), n_pred_bucket=prec[:, 1].tolist(),
                              acc=float(np.mean([r["acc"] for r in rr])), alive_acc=float(np.mean([r["alive_acc"] for r in rr])), team_acc=float(np.mean([r["team_acc"] for r in rr])),
                              n_pred=float(np.mean([r["n_pred"] for r in rr])), n_gt=float(np.mean([r["n_gt"] for r in rr])), label_free=lf)
        o = out["tags"][t]
        print(f"{t:12s} spellF1 {o['spell_mean']:.3f} ± {o['spell_std']:.3f} [{o['spell_min']:.3f}, {o['spell_max']:.3f}]  macroF1 {o['macro_mean']:.3f}  team_acc {o['team_acc']:.3f} alive_acc {o['alive_acc']:.3f}  n_pred/n_gt {o['n_pred']:.0f}/{o['n_gt']:.0f}")
        print(f"             recall by length {o['recall']}  precision {o['precision']}")
        print("             per match: " + "  ".join(f"{n[-3:]} {per[n]['spell']:.3f}" for n in names))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
