"""Held-out external evaluation of one or more checkpoints (ensemble), with a decode sweep.

    python evalx.py --ckpt a.pt [b.pt] --sets dfl metrica sc --names <comma list or empty = all> --out probs.npz [--sweep]

Builds the same fixed-seed validation versions as train.py, saves posteriors, and (with --sweep)
reports mean spellF1 / macroF1 per tag over a grid of decode parameters. The target is not read.
"""
import argparse
import itertools
import json
from multiprocessing import Pool

import numpy as np
import torch

import common as C
import train as T
from model import PossNet, predict


def _score_one(args):
    prob, gt, half, c_ha, c_d, ml = args
    path = np.full(len(gt), C.X, np.int8)
    for a, b in C.segments_by_half(dict(half=half)):
        path[a:b] = C.decode(prob[a:b], c_ha, c_d, ml)
    r = C.score(gt, path, half)
    return r["spell_f1"], r["macro_f1"], r["n_pred"], r["n_gt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", default=["dfl"])
    ap.add_argument("--names", default="")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    names = set(filter(None, a.names.split(",")))
    vjobs = []
    i = 0
    for ds in a.sets:
        for path in C.list_matches(ds):
            m = C.load_external(path)
            if names and m["name"] not in names:
                continue
            T.CLEAN[m["name"]] = m; T.KIND[m["name"]] = ds
            if ds == "sc":
                vjobs.append((m["name"], 0, 1.0, "provided", "asis"))
            else:
                vjobs += [(m["name"], 777000 + i * 10 + k, 1.0, "provided", "degraded") for k in range(2)] + [(m["name"], 778000 + i, "mild", "provided", "mild")]
                vjobs += [(m["name"], 779000 + i * 10 + k, 1.0, "video", "video") for k in range(2)]
            i += 1
    with Pool(48) as pool:
        vv = pool.map(T._version, [j[:4] for j in vjobs], chunksize=1)
    for v, j in zip(vv, vjobs):
        v["tag"] = f"{T.KIND[v['name']]}-{j[4]}"
    models = []
    for ck in a.ckpt:
        sd = torch.load(ck, map_location="cuda", weights_only=True)
        m = PossNet(use_ball=not sd["args"].get("no_ball", False), drop=sd["args"].get("drop", 0.15), ch=sd["args"].get("ch", 192), dilations=tuple(sd["args"].get("dilations", (1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64)))).cuda()
        m.load_state_dict(sd["model"], strict=False); m.eval(); models.append(m)
    probs = []
    for v in vv:
        d = T.to_dev(v, "cuda")
        prob = np.zeros((len(v["gt"]), 3), np.float32); prob[:, 2] = 1
        for s, e in C.segments_by_half(v):
            p = torch.stack([predict(m, d["home"][s:e], d["away"][s:e], d["ball"][s:e], d["aux"][s:e], tta=not a.no_tta, dom=v["dom"]) for m in models]).mean(0)
            prob[s:e] = p.cpu().numpy()
        probs.append(prob)
    grid = list(itertools.product([0, 2, 4, 6, 8, 12], [2, 4, 8, 12], [0, 4, 8, 12, 19])) if a.sweep else [(4, 4, 0)]
    jobs = [(probs[i], vv[i]["gt"], vv[i]["half"], *g) for g in grid for i in range(len(vv))]
    with Pool(96) as pool:
        res = pool.map(_score_one, jobs, chunksize=4)
    res = np.array(res).reshape(len(grid), len(vv), 4)
    tags = sorted({v["tag"] for v in vv})
    table = []
    for gi, g in enumerate(grid):
        row = dict(c_ha=g[0], c_d=g[1], min_len=g[2])
        for t in tags:
            idx = [i for i, v in enumerate(vv) if v["tag"] == t]
            row[t] = [round(float(res[gi, idx, k].mean()), 4) for k in range(4)]
        table.append(row)
    key = "dfl-degraded" if "dfl-degraded" in tags else tags[0]
    table.sort(key=lambda r: -r[key][0])
    print("tags:", tags, " cells: [spellF1, macroF1, n_pred, n_gt]; sorted by", key)
    for r in table[:12]:
        print(json.dumps(r))
    base = [r for r in table if (r["c_ha"], r["c_d"], r["min_len"]) == (4, 4, 0)]
    print("default (4,4,0):", json.dumps(base[0]) if base else None)
    acc = {}
    for t in tags:
        idx = [i for i, v in enumerate(vv) if v["tag"] == t]
        acc[t] = round(float(np.mean([(probs[i].argmax(1) == vv[i]["gt"])[vv[i]["gt"] != C.X].mean() for i in idx])), 4)
    print("frame accuracy:", json.dumps(acc))
    if a.out:
        json.dump(dict(table=table, acc=acc, ckpt=a.ckpt), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
