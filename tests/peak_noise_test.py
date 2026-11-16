"""How sensitive is the submission to tiny numeric differences in the ball detector's output?

Float32 inference on a CPU and on a GPU gives heatmap peaks that differ by about 1e-6 in confidence
and 1e-4 px in position (measured). This test takes the peaks of a finished run, optionally swaps
in peaks computed on other hardware for some chunks, and/or adds random perturbations of a chosen
size to every peak, then reruns linking, the sequence model, and decoding, and counts how many
labels change.

    python tests/peak_noise_test.py <run dir> [--swap <other peaks dir>] [--noise CONF PX --seeds N] [--device cuda]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from mtah import ball_link, decode, seq  # noqa: E402
from mtah.prep import N_FRAMES  # noqa: E402


def pipeline(PK, ran, M, target, cfg, device):
    r = ball_link.link(PK, ran, M, target["ball"].astype(np.float64))
    post, _ = seq.posterior([os.path.join(HERE, c) for c in cfg["seq_ckpts"]], target, r[cfg["ball_key"]], fps=cfg["seq_fps"],
                            dom=cfg["convention"], device=device, chunk_s=cfg.get("seq_chunk_s"), overlap_s=cfg.get("seq_overlap_s", 60.0))
    return r[cfg["ball_key"]], post, decode.labels_from_posterior(cfg, post, target)[0]


def report(tag, ref, new):
    (b0, p0, l0), (b1, p1, l1) = ref, new
    n0, n1 = np.isnan(b0[:, 0]), np.isnan(b1[:, 0])
    both = ~n0 & ~n1
    dm = np.hypot((b0[both, 0] - b1[both, 0]) * 105.3, (b0[both, 1] - b1[both, 1]) * 68.0)
    out = dict(tag=tag, ball_presence_differs=int((n0 != n1).sum()), ball_frames_over_1mm=int((dm > 1e-3).sum()), ball_frames_over_1m=int((dm > 1).sum()),
               post_max_diff=float(np.abs(p0 - p1).max()), post_frames_over_1e3=int((np.abs(p0 - p1).max(1) > 1e-3).sum()),
               labels_differ=int((l0 != l1).sum()))
    print(json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--swap", default=""); ap.add_argument("--noise", nargs=2, type=float, default=None)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--device", default="cuda"); ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    w = os.path.join(a.run, "work")
    target = dict(np.load(f"{w}/target.npz")); M = np.load(f"{w}/homography.npy")
    PK, ran = ball_link.load_peaks(f"{w}/peaks")
    ref = pipeline(PK, ran, M, target, cfg, a.device)
    if a.swap:
        P2 = PK.copy(); n = 0
        for f in sorted(glob.glob(f"{a.swap}/pk_*.npy")):
            s = int(f[-10:-4]); x = np.load(f); P2[s:s + len(x)] = x; n += len(x)
        print("swapped in", n, "frames of peaks from", a.swap, "; bit-identical to the run's own:", bool(np.array_equal(P2, PK)))
        report("swap", ref, pipeline(P2, ran, M, target, cfg, a.device))
    if a.noise:
        dc, dp = a.noise
        for s in range(a.seeds):
            rng = np.random.default_rng(s)
            P2 = PK.copy()
            valid = P2[..., 2] > 0
            P2[..., 2] = np.where(valid, np.clip(P2[..., 2] + rng.uniform(-dc, dc, P2.shape[:2]), 0, 1), P2[..., 2]).astype(np.float32)
            P2[..., :2] = np.where(valid[..., None], P2[..., :2] + rng.uniform(-dp, dp, P2.shape[:2] + (2,)), P2[..., :2]).astype(np.float32)
            report(f"noise conf {dc:g} px {dp:g} seed {s}", ref, pipeline(P2, ran, M, target, cfg, a.device))


if __name__ == "__main__":
    main()
