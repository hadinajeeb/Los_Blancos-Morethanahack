"""Input statistics of the video ball track on the target (no labels), to set the 'video' degradation profile."""
import numpy as np
import common as C
from degrade import _runs

z = np.load(f"{C.P}/derived/ball_video.npz")
print({k: z[k].shape for k in z.files})
g = np.zeros(C.N_TARGET, bool)
for a, b in C.TARGET_HALVES:
    g[a:b + 1] = True
for key in ("provided_xy", "det_xy", "video_first_xy", "merged_xy"):
    b = z[key]
    ok = ~np.isnan(b[:, 0])
    s, e = _runs(~ok & g); gl = (e - s) / 25
    s2, e2 = _runs(ok & g); vl = (e2 - s2) / 25
    m = b * np.array([105.3, 68.0])
    v = np.hypot(*(m[2:] - m[:-2]).T) / (2 / 25)
    okv = ok[2:] & ok[:-2] & g[1:-1]
    acc = np.hypot(*(m[2:] - 2 * m[1:-1] + m[:-2]).T)       # second difference, metres per frame^2
    oka = okv & ok[1:-1]
    out = ok & g & ((b[:, 0] < 0) | (b[:, 0] > 1) | (b[:, 1] < 0) | (b[:, 1] > 1))
    print(f"{key:15s} present {ok[g].mean():.3f} gaps {len(gl)} gap s p10/50/90 {np.percentile(gl,[10,50,90]).round(1)} vis s p10/50/90 {np.percentile(vl,[10,50,90]).round(1)}"
          f" speed m/s p50/90/99 {np.percentile(v[okv],[50,90,99]).round(1)} 2nd-diff m p50/90 {np.percentile(acc[oka],[50,90]).round(3)} slow(<0.5m/s) {np.mean(v[okv]<0.5):.3f} outside {out.sum()/max(ok[g].sum(),1):.3f}")
src = z["video_first_src"][g]
print("video_first_src shares (0 none, 1 provided, 2 video):", np.round(np.bincount(src, minlength=3) / len(src), 3))
print("interp share of graded:", float(z["det_interp"][g].mean()))
