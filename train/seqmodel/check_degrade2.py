"""More input statistics, target vs degraded DFL (inputs only): spread of tracked players along the
pitch, resting-ball share, ball speed, close player pairs, players near the ball.

    python check_degrade2.py [version]      # degradation version 1 (default) or 2
"""
import sys
import numpy as np
import common as C
from degrade import degrade, sample_cfg

VER = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def stats(home, away, ball, tag):
    P = np.concatenate([home, away], 1)
    x = P[..., 0]
    n = (~np.isnan(x)).sum(1)
    ok = n >= 6
    xs = np.sort(np.where(np.isnan(x), np.inf, x), 1)
    ext = np.array([np.nanmax(x[i]) - np.nanmin(x[i]) for i in np.flatnonzero(ok)[::5]])
    q = np.nanpercentile(x[ok][::5], [10, 90], axis=1)
    idr = q[1] - q[0]
    bp = ~np.isnan(ball[:, 0])
    k = int(round(C.FPS))
    d1 = np.hypot(*(ball[k:] - ball[:-k]).T)
    okd = bp[k:] & bp[:-k]
    sp = np.hypot(*(ball[2:] - ball[:-2]).T) / (2 * C.DT)
    oks = bp[2:] & bp[:-2]
    # players within 2.5 m of the ball, and pairs of players within 1 m
    db = np.hypot(P[..., 0] - ball[:, None, 0], P[..., 1] - ball[:, None, 1])
    near = (db < 2.5).sum(1)[bp]
    sub = P[::10]
    dd = np.hypot(sub[:, :, None, 0] - sub[:, None, :, 0], sub[:, :, None, 1] - sub[:, None, :, 1])
    iu = np.triu_indices(P.shape[1], 1)
    close = (dd[:, iu[0], iu[1]] < 1.0).sum(1).mean()
    print(f"{tag:26s} x max-min p5/50/95 {np.percentile(ext,[5,50,95]).round(0)}  x p90-p10 p50 {np.median(idr):5.1f}  rest(1s<0.5m) {np.mean(d1[okd]<0.5):.3f}"
          f"  speed p50/90/99 {np.percentile(sp[oks],[50,90,99]).round(1)}  near-ball>=3 {np.mean(near>=3):.3f} near=0 {np.mean(near==0):.3f}  pairs<1m {close:.2f}  outside {np.mean((np.abs(ball[bp,0])>52.65)|(np.abs(ball[bp,1])>34)):.3f}")


for ballkey in ("provided", "video_first_xy"):
    for t in C.load_target(ball=ballkey):
        g = t["half"] > 0
        stats(t["home"][g], t["away"][g], t["ball"][g], f"{t['name']} {ballkey[:8]}")
m = C.load_external(C.list_matches("dfl")[2])
for prof in ("provided", "video"):
    for seed in range(4):
        rng = np.random.default_rng(seed)
        d = degrade(m, rng, cfg=sample_cfg(rng, 1.0, prof, version=VER))
        stats(d["home"], d["away"], d["ball"], f"dfl {prof} v{VER} s{seed}")
