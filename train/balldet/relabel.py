"""Self-training step: replace the classical blob labels by the net's own peak near the projection.

For each 'pos' window, take the highest heatmap peak within R px of the projected provided ball,
if its confidence exceeds THR. Output: lab (N,2) in window coordinates, NaN = no label.
Usage: python relabel.py --crops DIR --ckpt C --out L.npz [--thr 0.4] [--r 64]
"""
import argparse

import numpy as np
import torch

from model import BallNet, peaks

ap = argparse.ArgumentParser()
ap.add_argument("--crops"); ap.add_argument("--ckpt"); ap.add_argument("--out")
ap.add_argument("--thr", type=float, default=0.4); ap.add_argument("--r", type=float, default=64.0)
a = ap.parse_args()
dev = "cuda"
m = np.load(f"{a.crops}/meta.npz")
X = np.load(f"{a.crops}/X.npy", mmap_mode="r")
N, _, CS, _, _ = X.shape
proj = m["proj"] - np.stack([m["x0"], m["y0"]], 1)
old = m["blob"] - np.stack([m["x0"], m["y0"]], 1)
net = BallNet().to(dev).to(memory_format=torch.channels_last).eval()
net.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True))
lab = np.full((N, 2), np.nan, np.float32); conf = np.zeros(N, np.float32)
ign = np.full((N, 3, 2), np.nan, np.float32)  # other confident peaks: possible real balls, not penalised
kind = m["kind"]
FAR, IGN = 0.6, 0.6
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for s in range(0, N, 128):
        idx = np.arange(s, min(s + 128, N))
        x = torch.from_numpy(np.ascontiguousarray(X[idx])).to(dev).permute(0, 1, 4, 2, 3).reshape(len(idx), 9, CS, CS).float() / 255.0
        pk = peaks(net(((x - 0.45) / 0.25).contiguous(memory_format=torch.channels_last)), k=4, thr=0.05).cpu().numpy()
        for i, j in enumerate(idx):
            b = -1
            if kind[j] == 1:
                d = np.hypot(pk[i, :, 0] - proj[j, 0], pk[i, :, 1] - proj[j, 1])
                okk = (d < a.r) & (pk[i, :, 2] > a.thr)
                if okk.any():
                    b = np.flatnonzero(okk)[0]          # peaks are sorted by confidence
                elif pk[i, 0, 2] > FAR:
                    b = 0                               # confident ball far from the projection (provided track is off)
                if b >= 0:
                    lab[j] = pk[i, b, :2]; conf[j] = pk[i, b, 2]
            others = [q for q in range(4) if q != b and pk[i, q, 2] > IGN][:3]
            for t, q in enumerate(others):
                ign[j, t] = pk[i, q, :2]
pos = np.flatnonzero(kind == 1)
np.savez(a.out, lab=lab, conf=conf, ign=ign)
had = np.isfinite(old[pos, 0]); has = np.isfinite(lab[pos, 0])
print("pos windows", len(pos), " old blob labels", had.mean().round(3), " new labels", has.mean().round(3))
both = had & has
dd = np.hypot(lab[pos][both, 0] - old[pos][both, 0], lab[pos][both, 1] - old[pos][both, 1])
print("where both: agree within 6px", (dd < 6).mean().round(3), " new-only", (has & ~had).mean().round(3), " lost", (had & ~has).mean().round(3))
print("offset new label vs projection px pct 25/50/75/90:", np.percentile(np.hypot(lab[pos][has, 0] - proj[pos][has, 0], lab[pos][has, 1] - proj[pos][has, 1]), [25, 50, 75, 90]).round(1))
