"""Extract training/validation crops for the ball heatmap net.

Sample = 3 frames (t-2, t, t+2), one CS x CS window at the same pixel position, uint8.
Two kinds: 'pos' window placed at random so the projected provided ball is inside it;
'neg' window placed at random elsewhere in the same frame (no provided ball within it).
Labels per sample: projection (crop coords) and the nearest classical blob within 25 px (or NaN).
Validation = time blocks: (frame // 1500) % 10 == 7.
Usage: python make_crops.py <outdir> <n_train_frames> <n_val_frames> [workers]
"""
import os
import sys
from collections import deque
from multiprocessing import Pool

import numpy as np
from numpy.lib.format import open_memmap

from common import Reader, graded_mask, N_FRAMES, RUNS, W, H

CS, DT, CH = 320, 2, 1000
outdir, ntr, nva = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
nw = int(sys.argv[4]) if len(sys.argv) > 4 else 96
z = np.load(f"{RUNS}/blobs.npz")
P, C, SEL = z["proj"].astype(np.float64), z["cand"], z["sel"]
inb = (P[:, 0] > 8) & (P[:, 0] < W - 8) & (P[:, 1] > 8) & (P[:, 1] < H - 8)
ok = SEL & inb
ok[:DT] = False; ok[-DT:] = False
isval = (np.arange(N_FRAMES) // 1500) % 10 == 7
rng = np.random.default_rng(0)
tr = rng.choice(np.flatnonzero(ok & ~isval), ntr, replace=False)
va = rng.choice(np.flatnonzero(ok & isval), nva, replace=False)
frames = np.sort(np.concatenate([tr, va]))
n = len(frames)
# two windows per frame: index 2*i = pos, 2*i+1 = neg
wx = np.zeros((n, 2), np.int32); wy = np.zeros((n, 2), np.int32)
for i, f in enumerate(frames):
    px, py = P[f]
    m = 24  # keep the projection at least m px from the window edge where possible
    def place(p, L):
        lo, hi = max(0, int(p) - CS + m), min(L - CS, max(0, int(p) - m))
        lo = min(lo, hi)
        return int(rng.integers(lo, hi + 1))
    x0, y0 = place(px, W), place(py, H)
    wx[i, 0], wy[i, 0] = x0, y0
    for _ in range(50):
        x1, y1 = int(rng.integers(0, W - CS + 1)), int(rng.integers(0, H - CS + 1))
        if not (x1 - 70 < px < x1 + CS + 70 and y1 - 70 < py < y1 + CS + 70):
            break
    wx[i, 1], wy[i, 1] = x1, y1
d = np.hypot(C[frames, 0, 0] - P[frames, 0], C[frames, 0, 1] - P[frames, 1])
blob = np.where((d < 25)[:, None], C[frames, 0, :2], np.nan)

os.makedirs(outdir, exist_ok=True)
X_path = f"{outdir}/X.npy"


def work(a):
    b = min(a + CH, N_FRAMES)
    lo, hi = np.searchsorted(frames, a), np.searchsorted(frames, b)
    if lo == hi:
        return 0
    X = np.load(X_path, mmap_mode="r+")
    want = {int(frames[i]): i for i in range(lo, hi)}
    r = Reader()
    buf = deque(maxlen=2 * DT + 1)
    for i, img in r.read(max(a - DT, 0), min(b + DT, N_FRAMES)):
        buf.append((i, img))
        c = i - DT
        if c in want and len(buf) == 2 * DT + 1 and buf[0][0] == c - DT:
            j = want[c]
            for k in range(2):
                x0, y0 = wx[j, k], wy[j, k]
                for t, bi in enumerate((0, DT, 2 * DT)):
                    X[2 * j + k, t] = buf[bi][1][y0:y0 + CS, x0:x0 + CS]
    r.close()
    X.flush()
    return hi - lo


if __name__ == "__main__":
    open_memmap(X_path, mode="w+", dtype=np.uint8, shape=(2 * n, 3, CS, CS, 3)).flush()
    with Pool(nw) as p:
        done = 0
        for k in p.imap_unordered(work, range(0, N_FRAMES, CH)):
            done += k
    print("frames written", done, "of", n)
    rep = lambda v: np.repeat(v, 2)
    np.savez(f"{outdir}/meta.npz", frame=rep(frames), kind=np.tile([1, 0], n), x0=wx.reshape(-1), y0=wy.reshape(-1),
             val=rep(isval[frames]), proj=np.repeat(P[frames], 2, 0), blob=np.repeat(blob, 2, 0))
    print("train frames", ntr, "val frames", nva, "blob-labelled share", np.isfinite(blob[:, 0]).mean().round(3))
