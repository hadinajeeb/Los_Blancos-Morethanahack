"""Compare two pipeline runs stage by stage: peaks, ball track, posterior, labels.

    python tests/compare_runs.py <run dir A> <run dir B>          # each holds work/ and submission.csv
    python tests/compare_runs.py <run dir A> --csv <other.csv>    # labels only, against any submission file

Prints, per stage, whether the arrays are bit-identical and how large the differences are. The
last line says whether the two CSV files are byte-identical.
"""
import glob
import hashlib
import os
import sys

import numpy as np


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def labels(p):
    return np.array([l.split(",")[1] for l in open(p).read().split("\n")[1:] if l])


def peaks(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/work/peaks/pk_*.npy")):
        out[os.path.basename(f)] = np.load(f)
    return out


def cmp_csv(a, b):
    la, lb = labels(a), labels(b)
    d = np.flatnonzero(la != lb)
    print(f"labels: {len(d)} of {len(la)} frames differ")
    if len(d):
        e = np.flatnonzero(np.diff(d) > 1)
        regs = list(zip(np.r_[d[0], d[e + 1]], np.r_[d[e], d[-1]]))
        print(f"  {len(regs)} regions, lengths (frames): {sorted((int(y - x + 1) for x, y in regs), reverse=True)[:30]}")
        kinds = {}
        for x, y in zip(la[d], lb[d]):
            kinds[x + ">" + y] = kinds.get(x + ">" + y, 0) + 1
        print("  changes A>B:", kinds)
    print("CSV byte-identical:", sha(a) == sha(b), sha(a)[:16], sha(b)[:16])


def main():
    a = sys.argv[1]
    if sys.argv[2] == "--csv":
        return cmp_csv(f"{a}/submission.csv", sys.argv[3])
    b = sys.argv[2]
    pa, pb = peaks(a), peaks(b)
    common = sorted(set(pa) & set(pb))
    if common:
        A = np.concatenate([pa[k] for k in common]); B = np.concatenate([pb[k] for k in common])
        same = np.array_equal(A, B)
        conf_ok = (A[..., 2] > 0.05) | (B[..., 2] > 0.05)
        dpx = np.hypot(A[..., 0] - B[..., 0], A[..., 1] - B[..., 1])[conf_ok]
        dcf = np.abs(A[..., 2] - B[..., 2])
        print(f"peaks: {len(common)} chunks in common, bit-identical: {same}; max |d conf| {dcf.max():.2e}; "
              f"position diff px: max {dpx.max():.3g}, share over 0.01 px {float((dpx > 0.01).mean()):.2e}, share over 1 px {float((dpx > 1).mean()):.2e}")
    fa, fb = f"{a}/work/ball_video.npz", f"{b}/work/ball_video.npz"
    if os.path.exists(fa) and os.path.exists(fb):
        x, y = np.load(fa)["video_first_xy"], np.load(fb)["video_first_xy"]
        nx, ny = np.isnan(x[:, 0]), np.isnan(y[:, 0])
        both = ~nx & ~ny
        dm = np.hypot((x[both, 0] - y[both, 0]) * 105.3, (x[both, 1] - y[both, 1]) * 68.0)
        print(f"ball track: bit-identical: {np.array_equal(x, y, equal_nan=True)}; presence differs on {int((nx != ny).sum())} frames; "
              f"distance m: max {dm.max():.3g}, frames over 1e-3 m {int((dm > 1e-3).sum())}, over 1 m {int((dm > 1).sum())}")
    for fa in sorted(glob.glob(f"{a}/work/post_*.npz")):
        name = os.path.basename(fa)
        fb = f"{b}/work/{name}"
        if os.path.exists(fb):
            x, y = np.load(fa)["post"], np.load(fb)["post"]
            d = np.abs(x - y)
            print(f"posterior ({name}): bit-identical: {np.array_equal(x, y)}; max |diff| {d.max():.2e}; frames with diff over 1e-3: {int((d.max(1) > 1e-3).sum())}, over 0.1: {int((d.max(1) > 0.1).sum())}")
    for fa in sorted(glob.glob(f"{a}/*.csv")):
        fb = f"{b}/{os.path.basename(fa)}"
        if os.path.exists(fb):
            print(os.path.basename(fa))
            cmp_csv(fa, fb)


if __name__ == "__main__":
    main()
