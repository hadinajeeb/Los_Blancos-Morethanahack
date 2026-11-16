"""Classical white-blob search near the projected provided ball. Purpose: (a) measure pseudo-label
noise and any frame offset, (b) give refined pixel labels to bootstrap the detector.

For every graded frame with a provided ball, find white compact blobs within +-R px of the
projection and store up to K of them (x, y, area), nearest first.
Usage: python blob_refine.py <out.npz> [workers]
"""
import sys
from multiprocessing import Pool

import cv2
import numpy as np

from common import Reader, ball_pixels, graded_mask, N_FRAMES, W, H

R, K, CH = 80, 6, 1000
P = ball_pixels()
G = graded_mask()
SEL = G & ~np.isnan(P[:, 0])


def blobs(img, cx, cy):
    x0, y0 = int(round(cx)) - R, int(round(cy)) - R
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + 2 * R, W), min(y0 + 2 * R, H)
    if xb - xa < 20 or yb - ya < 20:
        return []
    c = img[ya:yb, xa:xb]
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 2] > 165) & (hsv[..., 1] < 90)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 14 or a > 220 or w > 22 or h > 22:
            continue
        if not (0.55 < w / h < 1.8) or a / (w * h) < 0.5:
            continue
        out.append((cen[i][0] + xa, cen[i][1] + ya, a))
    out.sort(key=lambda t: (t[0] - cx) ** 2 + (t[1] - cy) ** 2)
    return out[:K]


def work(a):
    b = min(a + CH, N_FRAMES)
    if not SEL[a:b].any():
        return a, None
    res = np.full((b - a, K, 3), np.nan, np.float32)
    r = Reader()
    for i, img in r.read(a, b):
        if SEL[i]:
            for k, t in enumerate(blobs(img, P[i, 0], P[i, 1])):
                res[i - a, k] = t
    r.close()
    return a, res


if __name__ == "__main__":
    out = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    C = np.full((N_FRAMES, K, 3), np.nan, np.float32)
    starts = [a for a in range(0, N_FRAMES, CH) if SEL[a:a + CH].any()]
    with Pool(nw) as p:
        for j, (a, res) in enumerate(p.imap_unordered(work, starts)):
            if res is not None:
                C[a:a + len(res)] = res
            if j % 20 == 0:
                print(j, len(starts), flush=True)
    np.savez_compressed(out, cand=C, proj=P.astype(np.float32), sel=SEL)
    d = np.hypot(C[:, 0, 0] - P[:, 0], C[:, 0, 1] - P[:, 1])
    s = SEL
    print("tracked graded frames", s.sum(), " with a blob within R:", np.isfinite(d[s]).mean().round(3))
    print("nearest-blob distance px pct 10/25/50/75/90:", np.nanpercentile(d[s], [10, 25, 50, 75, 90]).round(1))
    for thr in [5, 10, 15, 25, 40]:
        print(f"  within {thr:2d}px: {(d[s] < thr).mean():.3f}")
    # frame-offset test: compare blob at frame f against projection of frame f+k
    for k in range(-4, 5):
        idx = np.flatnonzero(s)
        idx = idx[(idx + k >= 0) & (idx + k < N_FRAMES)]
        dk = np.hypot(C[idx, 0, 0] - P[idx + k, 0], C[idx, 0, 1] - P[idx + k, 1])
        print(f"  offset {k:+d}: median {np.nanmedian(dk):6.2f}  within 6px {(dk < 6).mean():.3f}")
