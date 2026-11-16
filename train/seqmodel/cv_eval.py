"""Posteriors of one fold's checkpoint(s) on its held-out DFL matches, under five input conditions.

    python cv_eval.py --ckpt fold.pt --names DFL-MAT-A,DFL-MAT-B --out cv_x_f0.npz

Conditions (tags): deg (our provided-ball noise, 2 seeds), video (our video-ball noise, 2 seeds),
mild (dead-ball ball re-simulated, nothing else), rules_deg and rules_clean (workstream `rules`'
noise model and its ball-blanked-on-D set). Seeds depend only on the match, so every variant and
fold sees identical inputs. Scoring happens in cv_report.py.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import torch

import common as C
import eval_rulesdeg as E
import train as T
from degrade import gap_aux
from model import load_model, predict_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = a.names.split(",")
    jobs, tags = [], []
    paths = C.list_matches("dfl")
    for i, path in enumerate(paths):
        m = C.load_external(path)
        if m["name"] not in names:
            continue
        T.CLEAN[m["name"]] = m; T.KIND[m["name"]] = "dfl"
        for k in range(2):
            jobs.append((m["name"], 777000 + i * 10 + k, 1.0, "provided", 1)); tags.append("deg")
            jobs.append((m["name"], 779000 + i * 10 + k, 1.0, "video", 1)); tags.append("video")
            jobs.append((m["name"], 777000 + i * 10 + k, 1.0, "provided", 2)); tags.append("deg2")
            jobs.append((m["name"], 779000 + i * 10 + k, 1.0, "video", 2)); tags.append("video2")
        jobs.append((m["name"], 778000 + i, "mild", "provided", 1)); tags.append("mild")
    with Pool(18) as pool:
        vv = pool.map(T._version, jobs, chunksize=1)
    seqs = [dict(name=v["name"], tag=t, home=v["home"], away=v["away"], ball=v["ball"], gt=v["gt"], half=v["half"]) for v, t in zip(vv, tags)]
    for path in paths:
        name = os.path.basename(path)[:-4]
        if name not in names:
            continue
        seed = sum(ord(c) for c in os.path.basename(path))
        for tag in ("rules_clean", "rules_deg"):
            M = E.rules_features.load_common(path)
            M = E.rules_degrade.mask_dead_ball(M) if tag == "rules_clean" else E.rules_degrade.degrade(M, seed=seed)
            v = E.to_canonical(M)
            seqs.append(dict(name=name, tag=tag, home=v["home"], away=v["away"], ball=v["ball"], gt=v["label"], half=v["half"]))
    models = [load_model(ck)[0] for ck in a.ckpt]
    out = dict(n=np.array(len(seqs)), names=np.array([q["name"] for q in seqs]), tags=np.array([q["tag"] for q in seqs]), fps=np.array(C.FPS))
    for i, q in enumerate(seqs):
        prob = np.zeros((len(q["gt"]), 3), np.float32); prob[:, 2] = 1
        bnd = np.zeros((len(q["gt"]), 2), np.float32)
        has_b = False
        aux = gap_aux(q["ball"])
        for s, e in C.segments_by_half(q):
            t = [torch.from_numpy(x[s:e]).cuda() for x in (q["home"], q["away"], q["ball"], aux)]
            ps, bs = zip(*[predict_full(m, *t) for m in models])
            prob[s:e] = torch.stack(ps).mean(0).cpu().numpy()
            if bs[0] is not None:
                bnd[s:e] = torch.stack(bs).mean(0).cpu().numpy(); has_b = True
        out[f"{i}_prob"] = prob; out[f"{i}_gt"] = q["gt"]; out[f"{i}_half"] = q["half"]
        out[f"{i}_ballok"] = ~np.isnan(q["ball"][:, 0])
        if has_b:
            out[f"{i}_bnd"] = bnd
    np.savez_compressed(a.out, **out)
    print("wrote", a.out, len(seqs), "sequences")


if __name__ == "__main__":
    main()
