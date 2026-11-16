"""Train the ball heatmap net on pseudo-labelled crops.

Round 0: positive = nearest classical white blob within 25 px of the projected provided ball.
A disk of IGN px around the projection is ignored (the true ball may be there, unlabelled).
Round >=1: pass --labels <npz with 'lab' (N,2) crop-frame pixel labels, NaN = none> made by relabel.py;
then the whole window is supervised where a label exists.
Usage: python train.py --crops DIR --out CKPT [--labels L.npz] [--init CKPT] [--epochs 8]
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import BallNet, STRIDE, peaks

ap = argparse.ArgumentParser()
ap.add_argument("--crops"); ap.add_argument("--out"); ap.add_argument("--labels"); ap.add_argument("--init")
ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--ign", type=float, default=64.0)
a = ap.parse_args()
torch.manual_seed(0); np.random.seed(0)
dev = "cuda"
m = np.load(f"{a.crops}/meta.npz")
X = np.load(f"{a.crops}/X.npy", mmap_mode="r")
N, _, CS, _, _ = X.shape
off = np.stack([m["x0"], m["y0"]], 1)
proj = m["proj"] - off
if a.labels:
    _z = np.load(a.labels)
    lab = _z["lab"].astype(np.float64)  # already in window coords
    ign_pts = _z["ign"].astype(np.float64)
    full_sup = True
else:
    lab = m["blob"] - off
    ign_pts = np.full((len(lab), 3, 2), np.nan)
    full_sup = False
kind, val = m["kind"], m["val"]
if not a.labels:
    lab[kind == 0] = np.nan
tr_idx, va_idx = np.flatnonzero(~val), np.flatnonzero(val)
G = CS // STRIDE
gy, gx = np.mgrid[0:G, 0:G].astype(np.float32)
cx, cy = (gx + 0.5) * STRIDE, (gy + 0.5) * STRIDE


class DS(torch.utils.data.Dataset):
    def __init__(self, idx, train):
        self.idx, self.train = idx, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = np.ascontiguousarray(X[j])  # 3,CS,CS,3
        l, p, k, ig = lab[j].copy(), proj[j].copy(), kind[j], ign_pts[j].copy()
        flip = self.train and np.random.rand() < 0.5
        if flip:
            x = x[:, :, ::-1]; l[0] = CS - 1 - l[0]; p[0] = CS - 1 - p[0]; ig[:, 0] = CS - 1 - ig[:, 0]
        tgt = np.zeros((G, G), np.float32); w = np.ones((G, G), np.float32)
        for q in ig:
            if np.isfinite(q[0]):
                w[np.hypot(cx - q[0], cy - q[1]) < 16] = 0
        if k == 1 and not (full_sup and np.isfinite(l[0])):
            w[np.hypot(cx - p[0], cy - p[1]) < a.ign] = 0
        if k == 1:
            if np.isfinite(l[0]):
                d2 = (cx - l[0]) ** 2 + (cy - l[1]) ** 2
                tgt = np.exp(-d2 / (2 * 5.0 ** 2)).astype(np.float32)  # sigma 5 px
                tgt[np.unravel_index(d2.argmin(), d2.shape)] = 1.0
                w[d2 < 20 ** 2] = 1
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).reshape(9, CS, CS)
        return x, torch.from_numpy(tgt), torch.from_numpy(w), torch.tensor(np.nan_to_num(l, nan=-1e4), dtype=torch.float32), torch.tensor(p, dtype=torch.float32), int(k)


def focal(logit, tgt, w):
    p = torch.sigmoid(logit).clamp(1e-5, 1 - 1e-5)
    pos = (tgt == 1).float()
    lp = -torch.log(p) * (1 - p) ** 2 * pos
    ln = -torch.log(1 - p) * p ** 2 * (1 - tgt) ** 4 * (1 - pos)
    return ((lp + ln) * w).sum() / pos.sum().clamp_min(1.0)


def prep(x, train):
    x = x.to(dev, non_blocking=True).float() / 255.0
    if train:  # mild photometric jitter, same for the three frames
        B = x.shape[0]
        x = x * (0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=dev)) + 0.1 * (torch.rand(B, 1, 1, 1, device=dev) - 0.5)
    return (x - 0.45) / 0.25


def evaluate(net, dl):
    net.eval(); rec = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, tgt, w, l, p, k in dl:
            pk = peaks(net(prep(x, False)), k=1, thr=0.0)[:, 0].cpu().numpy()
            for i in range(len(k)):
                rec.append((int(k[i]), float(l[i, 0]), np.hypot(pk[i, 0] - l[i, 0], pk[i, 1] - l[i, 1]),
                            np.hypot(pk[i, 0] - p[i, 0], pk[i, 1] - p[i, 1]), pk[i, 2]))
    r = np.array(rec); out = {}
    pos, neg = r[r[:, 0] == 1], r[r[:, 0] == 0]
    hasl = pos[:, 1] > -1e3
    for thr in (0.3, 0.5):
        out[f"recall@{thr}"] = float(((pos[hasl, 2] < 12) & (pos[hasl, 4] > thr)).mean())       # labelled windows: peak on the label
        out[f"near_proj_unlab@{thr}"] = float(((pos[~hasl, 3] < 64) & (pos[~hasl, 4] > thr)).mean())  # unlabelled: a peak near the projection
        out[f"neg_fp@{thr}"] = float((neg[:, 4] > thr).mean())                                    # ball-free windows that fire
        out[f"pos_far_fp@{thr}"] = float(((pos[:, 3] >= 64) & (pos[:, 4] > thr)).mean())        # top peak far from the projection
    return out


if __name__ == "__main__":
    net = BallNet().to(dev).to(memory_format=torch.channels_last)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev, weights_only=True))
    dl = torch.utils.data.DataLoader(DS(tr_idx, True), batch_size=a.bs, shuffle=True, num_workers=24, drop_last=True, pin_memory=True, persistent_workers=True)
    dv = torch.utils.data.DataLoader(DS(va_idx, False), batch_size=96, num_workers=16)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    print("train windows", len(tr_idx), "val windows", len(va_idx), "labelled pos share", np.isfinite(lab[kind == 1][:, 0]).mean().round(3), flush=True)
    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); tot = 0.0
        for it, (x, tgt, w, l, p, k) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(prep(x, True).contiguous(memory_format=torch.channels_last))
            loss = focal(out.float(), tgt.to(dev), w.to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        ev = evaluate(net, dv)
        print(f"ep {ep} loss {tot/len(dl):.4f} {time.time()-t0:.0f}s " + " ".join(f"{k}={v:.3f}" for k, v in ev.items()), flush=True)
        torch.save(net.state_dict(), a.out)
