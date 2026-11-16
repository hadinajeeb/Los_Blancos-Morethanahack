"""Stage 3: link per-frame heatmap peaks into one ball track.

Viterbi over K candidates plus a 'none' state per frame.
  emission(candidate) = conf - TAU - OUT_PEN * [more than 2 m outside the pitch]
  emission(none)      = 0
  candidate -> candidate = 0 if the pitch-plane step is <= VMAX metres, else -JUMP
  none -> candidate      = -BIRTH
Short 'none' runs between consistent detections are linearly interpolated, and so are holes of up
to 10 s where the ball reappears within 1 m of where it was last seen (a resting ball hidden by a
player). All constants are generic (speeds, confidence levels); none refers to a frame of the match.

Output `ball_video.npz`: det_xy (video ball, normalised pitch), video_first_xy (video ball where
present, provided ball elsewhere), provided_xy.
"""
import glob
import os

import numpy as np

from .prep import HALVES, N_FRAMES

PITCH = np.array([105.3, 68.0])
TAU, BIRTH, JUMP, VMAX, OUT_PEN, MAXFILL = 0.30, 1.0, 1.0, 2.5, 0.2, 12
STATIC_FILL, STATIC_M = 250, 1.0
MARGIN = 50


def load_peaks(pdir):
    files = sorted(glob.glob(f"{pdir}/pk_*.npy"))
    K = np.load(files[0]).shape[1]
    PK = np.zeros((N_FRAMES, K, 3), np.float32)
    ran = np.zeros(N_FRAMES, bool)
    for f in files:
        s = int(f[-10:-4]); a = np.load(f)
        PK[s:s + len(a)] = a; ran[s:s + len(a)] = True
    return PK, ran


def link(PK, ran, M, provided):
    K = PK.shape[1]
    Minv = np.linalg.inv(M)
    px, py, cf = PK[..., 0].astype(np.float64), PK[..., 1].astype(np.float64), PK[..., 2].astype(np.float64)
    d = Minv[:, None, 2, 0] * px + Minv[:, None, 2, 1] * py + Minv[:, None, 2, 2]
    X = (Minv[:, None, 0, 0] * px + Minv[:, None, 0, 1] * py + Minv[:, None, 0, 2]) / d
    Y = (Minv[:, None, 1, 0] * px + Minv[:, None, 1, 1] * py + Minv[:, None, 1, 2]) / d
    valid = cf > 0.05
    Xm, Ym = X * PITCH[0], Y * PITCH[1]
    outside = (Xm < -2) | (Xm > PITCH[0] + 2) | (Ym < -2) | (Ym > PITCH[1] + 2)
    E = np.where(valid, cf - TAU - OUT_PEN * outside, -1e9)  # (N,K)

    sel = np.full(N_FRAMES, -1, np.int16)
    for a, b in [(s - MARGIN, e + MARGIN + 1) for s, e in HALVES]:
        n = b - a
        S = np.zeros((n, K + 1)); BP = np.zeros((n, K + 1), np.int8)
        S[0, :K] = E[a] - BIRTH; S[0, K] = 0
        for t in range(1, n):
            f = a + t
            step = np.hypot(Xm[f][None, :] - Xm[f - 1][:, None], Ym[f][None, :] - Ym[f - 1][:, None])  # prev x cur
            T = np.where(step <= VMAX, 0.0, -JUMP)
            cand = S[t - 1, :K, None] + T
            frm_none = S[t - 1, K] - BIRTH
            best_prev = cand.argmax(0); best = cand.max(0)
            use_none = frm_none > best
            S[t, :K] = np.where(use_none, frm_none, best) + E[f]
            BP[t, :K] = np.where(use_none, K, best_prev)
            j = S[t - 1].argmax(); S[t, K] = S[t - 1, j]; BP[t, K] = j
        k = int(S[-1].argmax())
        for t in range(n - 1, -1, -1):
            sel[a + t] = k if k < K else -1
            k = int(BP[t, k]) if t > 0 else k

    has = sel >= 0
    idx = np.arange(N_FRAMES)
    det_xy = np.full((N_FRAMES, 2), np.nan, np.float32)
    det_conf = np.zeros(N_FRAMES, np.float32)
    det_xy[has, 0] = X[idx[has], sel[has]]; det_xy[has, 1] = Y[idx[has], sel[has]]
    det_conf[has] = cf[idx[has], sel[has]]
    interp = np.zeros(N_FRAMES, bool)
    h = np.flatnonzero(has)
    for i0, i1 in zip(h[:-1], h[1:]):
        g = i1 - i0
        if g > 1 and ran[i0:i1 + 1].all():
            dist = np.hypot(*((det_xy[i1] - det_xy[i0]) * PITCH))
            if (g <= MAXFILL + 1 and dist <= VMAX * g) or (g <= STATIC_FILL + 1 and dist <= STATIC_M):
                w = (np.arange(i0 + 1, i1) - i0)[:, None] / g
                det_xy[i0 + 1:i1] = det_xy[i0] * (1 - w) + det_xy[i1] * w
                interp[i0 + 1:i1] = True
    prov = provided.astype(np.float32)
    hp, hv = ~np.isnan(prov[:, 0]), ~np.isnan(det_xy[:, 0])
    vfirst = np.where(hv[:, None], det_xy, prov)
    return dict(det_xy=det_xy, det_conf=det_conf, det_interp=interp, ran=ran, provided_xy=prov,
                video_first_xy=vfirst.astype(np.float32), video_first_src=np.where(hv, 2, np.where(hp, 1, 0)).astype(np.uint8))


def run(work):
    out = os.path.join(work, "ball_video.npz")
    if os.path.exists(out):
        return out
    PK, ran = load_peaks(os.path.join(work, "peaks"))
    M = np.load(os.path.join(work, "homography.npy"))
    provided = np.load(os.path.join(work, "target.npz"))["ball"].astype(np.float64)
    r = link(PK, ran, M, provided)
    np.savez_compressed(out + ".tmp.npz", **r)
    os.replace(out + ".tmp.npz", out)
    g = np.zeros(N_FRAMES, bool)
    for a, b in HALVES:
        g[a:b + 1] = True
    hv = ~np.isnan(r["det_xy"][:, 0]); hp = ~np.isnan(r["provided_xy"][:, 0])
    print(f"ball_link: graded frames with provided ball {hp[g].mean():.3f}, video track {hv[g].mean():.3f}, either {(hp | hv)[g].mean():.3f}", flush=True)
    return out
