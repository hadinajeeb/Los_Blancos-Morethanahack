"""Load a common-format match, optionally resampled to another frame rate (default 25 fps).

    from loader import load
    m = load("<MTAH_ROOT>/external/skillcorner/common/sc_1874553.npz", fps=25)
    m["home"], m["away"], m["ball"], m["label"], m["half"], m["home_dir"], m["pitch"], m["meta"]

Resampling: positions are linearly interpolated between two source frames only when both are
valid, otherwise NaN. Labels, half, home_dir and boolean masks take the nearest source frame.
`detected_only=True` (SkillCorner 2024/25 files) drops extrapolated players and ball first.
"""
import json
import numpy as np


def load(path, fps=25.0, detected_only=False, label_key="label"):
    d = np.load(path)
    m = {k: d[k] for k in d.files}
    m["meta"] = json.loads(str(m["meta"]))
    m["label"] = m[label_key]
    if detected_only and "home_det" in m:
        for k, dk in (("home", "home_det"), ("away", "away_det")):
            a = m[k].copy(); a[~m[dk]] = np.nan; m[k] = a
        b = m["ball"].copy(); b[~m["ball_det"]] = np.nan; m["ball"] = b
    src = float(m["fps"])
    if fps is None or abs(src - fps) < 1e-6:
        return m
    N = len(m["label"])
    M = int(np.floor((N - 1) * fps / src)) + 1
    t = np.arange(M) * src / fps                    # position in source frames
    i0 = np.minimum(np.floor(t).astype(int), N - 1); i1 = np.minimum(i0 + 1, N - 1)
    w = (t - i0).astype(np.float32)
    near = np.minimum(np.rint(t).astype(int), N - 1)
    out = {}
    for k, v in m.items():
        if not isinstance(v, np.ndarray) or v.ndim == 0 or len(v) != N:
            out[k] = v
        elif v.dtype.kind == "f":
            ww = w.reshape((-1,) + (1,) * (v.ndim - 1))
            out[k] = v[i0] * (1 - ww) + v[i1] * ww   # NaN if either neighbour is NaN
        else:
            out[k] = v[near]
    out["fps"] = np.float32(fps); out["src_fps"] = np.float32(src)
    return out
