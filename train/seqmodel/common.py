"""Shared loading, canonicalisation, scoring, and decoding for workstream `seqmodel`.

Canonical view used by the model:
- 12.5 fps (every second frame of 25 fps data; 10 fps data is interpolated).
- Metres, origin at the centre spot, and the pitch rotated by 180 degrees whenever the home team
  attacks toward x = 0, so that home always attacks toward +x.
- `home`, `away`: float32 (N, S, 2), NaN = not tracked. `ball`: float32 (N, 2), NaN = no ball.
- `label`: uint8 (N,) 0 H, 1 A, 2 D, 3 X. `half`: uint8 (N,).
"""
import glob
import json
import os
import sys

import numpy as np

P = os.environ.get("MTAH_ROOT", ".")
sys.path.insert(0, f"{P}/official-repo/evaluation")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external"))   # loader.py (SkillCorner resampling)
import possession_metrics as pm  # noqa: E402

H, A, D, X = 0, 1, 2, 3
FPS = float(os.environ.get("SEQ_FPS", "12.5"))   # 12.5 (default) or 25
STRIDE = int(round(25.0 / FPS))
DT = 1.0 / FPS
S_MAX = 26  # slots per team after padding


def list_matches(dataset):
    pats = {
        "dfl": "external/dfl/common/*.npz",
        "metrica": "external/metrica/common/*.npz",
        "sc": "external/skillcorner/common/sc_*.npz",
        "sc2020": "external/skillcorner/common_v2020/*.npz",
    }
    return sorted(glob.glob(f"{P}/{pats[dataset]}"))


def _pad_slots(a, s=S_MAX):
    n, k, _ = a.shape
    if k >= s:
        # keep the s slots that are present most often
        keep = np.argsort(-(~np.isnan(a[..., 0])).sum(0), kind="stable")[:s]
        return a[:, np.sort(keep)]
    out = np.full((n, s, 2), np.nan, np.float32)
    out[:, :k] = a
    return out


def canonical(home, away, ball, home_dir, pitch):
    """Normalised 0-1 arrays -> metres, centred, home attacking +x."""
    L, W = float(pitch[0]), float(pitch[1])
    s = np.where(home_dir >= 0, 1.0, -1.0).astype(np.float32)  # dir 0 handled by the caller

    def conv(a, sgn):
        out = np.empty_like(a, dtype=np.float32)
        out[..., 0] = (a[..., 0] - 0.5) * L * sgn
        out[..., 1] = (a[..., 1] - 0.5) * W * sgn
        return out

    return conv(home, s[:, None]), conv(away, s[:, None]), conv(ball, s)


def load_external(path, label_key="label", detected_only=False):
    """One external match in the canonical 12.5 fps view. Returns a dict."""
    z = np.load(path)
    src = float(z["fps"])
    if abs(src - 25.0) < 1e-6:
        sl = slice(0, None, STRIDE)
        m = {k: z[k][sl] for k in ("home", "away", "ball", "half", "home_dir")}
        m["label"] = z[label_key][sl]
        pitch = z["pitch"]
    else:
        from loader import load as sc_load
        r = sc_load(path, fps=FPS, detected_only=detected_only, label_key=label_key)
        m = {k: r[k] for k in ("home", "away", "ball", "half", "home_dir", "label")}
        pitch = r["pitch"]
    hd = m["home_dir"].astype(np.int8)
    home, away, ball = canonical(m["home"], m["away"], m["ball"][:, :2], hd, pitch)
    lab = m["label"].astype(np.uint8).copy()
    lab[m["half"] == 0] = X
    return dict(name=os.path.basename(path)[:-4], home=_pad_slots(home), away=_pad_slots(away), ball=ball,
                label=lab, half=m["half"].astype(np.uint8), pitch=np.asarray(pitch, np.float32))


def segments_by_half(m):
    """Contiguous [a, b) index ranges, one per half, where half > 0."""
    out = []
    h = m["half"]
    for k in (1, 2):
        idx = np.flatnonzero(h == k)
        if len(idx):
            out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


# ---------------------------------------------------------------- target match
TARGET_HALVES = ((11185, 90651), (110808, 190197))  # graded video frames, inclusive (official DATA.md)
N_TARGET = 199319


def load_target(phase=0, margin_s=60.0, ball="provided"):
    """The target match, one dict per half, canonical 12.5 fps view.

    phase 0 takes even 25 fps frames, phase 1 odd ones; running both gives a 25 fps posterior.
    Each half carries `margin_s` seconds of ungraded context on both sides. `frames` holds the
    original video frame index of every row.

    Goalkeeper slots: the tracker holds or invents the two keeper slots (the last slot of each
    team) when they are off screen, and the away keeper slot sometimes sits on the home keeper.
    A keeper cannot stand in the opponent's defensive third in open play, so a keeper slot found
    there is dropped. This is a general rule, not a frame-specific correction.
    """
    z = np.load(f"{P}/derived/target_common.npz")
    pitch = z["pitch"]
    # ball source: the provided track, or a key of derived/ball_video.npz (e.g. "video_first_xy")
    ball_all = z["ball"][:, :2] if ball == "provided" else np.load(f"{P}/derived/ball_video.npz")[ball]
    half = z["half"]
    hd_full = z["home_dir"]
    out = []
    mar = int(round(margin_s * 25))
    for k, (a, b) in enumerate(TARGET_HALVES, start=1):
        d = int(hd_full[a])
        lo, hi = max(0, a - mar), min(N_TARGET, b + 1 + mar)
        fr = np.arange(lo + ((phase - lo) % STRIDE), hi, STRIDE)
        hdir = np.full(len(fr), d, np.int8)
        home, away, ball_c = canonical(z["home"][fr], z["away"][fr], ball_all[fr], hdir, pitch)
        third = float(pitch[0]) / 6.0  # |x| beyond L/6 on the far side = opponent's defensive third
        gk_h, gk_a = home.shape[1] - 1, away.shape[1] - 1
        bad_h = home[:, gk_h, 0] > third       # home keeper deep in the half home attacks
        bad_a = away[:, gk_a, 0] < -third      # away keeper deep in the half away attacks
        home[bad_h, gk_h] = np.nan
        away[bad_a, gk_a] = np.nan
        lab = z["label"][fr].astype(np.uint8)
        out.append(dict(name=f"target_h{k}", home=_pad_slots(home), away=_pad_slots(away), ball=ball_c, label=lab,
                        half=half[fr].astype(np.uint8), frames=fr, pitch=np.asarray(pitch, np.float32),
                        gk_dropped=(int(bad_h.sum()), int(bad_a.sum()))))
    return out


# ---------------------------------------------------------------- scoring and decoding
def runs_of(label, half):
    key = np.where(label != X, half.astype(np.int16), -1)
    edges = np.flatnonzero(np.diff(key) != 0) + 1
    bounds = np.concatenate(([0], edges, [len(key)]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if key[a] >= 0]


def score(gt, pred, half):
    """Official spellF1 and macroF1; runs split at half boundaries and at X."""
    gt = gt.astype(np.int8)
    pred = pred.astype(np.int8)
    runs = runs_of(gt, half)
    filled = pm.fill_abstentions(pred, runs)
    sp = pm.spell_scores(gt, filled, runs)
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    return dict(spell_f1=float(sp["mean_f1"]), macro_f1=float(pm.macro_f1(cm3)), n_gt=int(sp["n_gt"]), n_pred=int(sp["n_pred"]))


def viterbi(logp, c_ha, c_d):
    """Most likely H/A/D path under per-frame log posteriors and fixed switch costs.

    c_ha: cost of a direct H<->A switch. c_d: cost of entering or leaving D.
    """
    n = len(logp)
    cost = np.array([[0, c_ha, c_d], [c_ha, 0, c_d], [c_d, c_d, 0]], np.float64)
    back = np.zeros((n, 3), np.int8)
    v = logp[0].astype(np.float64).copy()
    lp = logp.astype(np.float64)
    for t in range(1, n):
        cand = v[:, None] - cost          # cand[i, j]: come from i, go to j
        back[t] = np.argmax(cand, axis=0)
        v = cand[back[t], np.arange(3)] + lp[t]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(v))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def min_duration(path, min_len, prob=None):
    """Remove runs shorter than min_len frames by giving them the label of a neighbour.

    The shortest run goes first. It takes the neighbour label with the higher mean posterior over
    the run (or the previous label when no posterior is given). Repeats until no short run is left.
    """
    path = path.copy()
    if min_len <= 1:
        return path
    while True:
        edges = np.flatnonzero(np.diff(path) != 0) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [len(path)]))
        lens = ends - starts
        if len(lens) <= 1:
            return path
        i = int(np.argmin(lens))
        if lens[i] >= min_len:
            return path
        a, b = starts[i], ends[i]
        cands = []
        if i > 0:
            cands.append(int(path[a - 1]))
        if i < len(lens) - 1:
            cands.append(int(path[b]))
        if prob is not None and len(cands) == 2 and cands[0] != cands[1]:
            new = cands[int(np.argmax([prob[a:b, c].mean() for c in cands]))]
        else:
            new = cands[0]
        path[a:b] = new


def viterbi_tv(logp, cost_ha, cost_d):
    """Viterbi with a switch cost per frame: cost_ha[t], cost_d[t] apply to a switch between t-1 and t."""
    n = len(logp)
    back = np.zeros((n, 3), np.int8)
    lp = logp.astype(np.float64)
    v = lp[0].copy()
    ar = np.arange(3)
    for t in range(1, n):
        a, d = cost_ha[t], cost_d[t]
        cand = v[:, None] - np.array([[0, a, d], [a, 0, d], [d, d, 0]])
        back[t] = np.argmax(cand, axis=0)
        v = cand[back[t], ar] + lp[t]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(v))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(prob, c_ha=4.0, c_d=4.0, min_len=0, bnd=None, beta=0.0):
    """Posterior (N, 3) -> label path.

    With bnd (N, 2) = P(direct H<->A change at t), P(any change at t) and beta > 0, the switch cost at
    frame t becomes max(0, c - beta * logit(p_boundary[t])): cheaper where the boundary head fires,
    dearer where it is silent."""
    logp = np.log(np.clip(prob, 1e-6, 1.0))
    if bnd is None or beta <= 0:
        path = viterbi(logp, c_ha, c_d)
    else:
        b = np.clip(bnd.astype(np.float64), 1e-4, 1 - 1e-4)
        lo = np.clip(np.log(b / (1 - b)), -6, 6)
        path = viterbi_tv(logp, np.maximum(0.0, c_ha - beta * lo[:, 0]), np.maximum(0.0, c_d - beta * lo[:, 1]))
    return min_duration(path, min_len, prob)


def spell_stats(path, fps=FPS):
    edges = np.flatnonzero(np.diff(path) != 0) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(path)]))
    lab = path[starts]
    dur = (ends - starts) / fps
    poss = dur[lab != D]
    return dict(n_poss=int((lab != D).sum()), n_dead=int((lab == D).sum()),
                poss_pct=np.percentile(poss, [10, 25, 50, 75, 90]).round(1).tolist() if len(poss) else [],
                under1=float((poss < 1).mean()) if len(poss) else 0.0, under2=float((poss < 2).mean()) if len(poss) else 0.0,
                shares=(np.bincount(path, minlength=3)[:3] / len(path)).round(3).tolist())


def kickoff_info(t, min_players=8, tol_m=1.5, sustain_s=0.5, cluster_gap_s=30.0):
    """Kick-off formations in one half of the target: (end frame of the first one, list of runs).

    Formation: at least `min_players` tracked per team, every home player in the half home defends
    (x <= tol) and every away player in the other (x >= -tol), for at least `sustain_s`. Teams stand
    like this only while a kick-off is pending, so the ball is dead during a run. Runs less than
    `cluster_gap_s` apart belong to one kick-off (a player steps over the line and back). Play
    cannot be live before the half's first kick-off, so graded frames before the end of the first
    cluster are dead ball. `t` is one dict from load_target (canonical view, home attacks +x).
    General rules: no frame number of the match is used. Returns video frames.
    """
    hx, ax = t["home"][..., 0], t["away"][..., 0]
    nh, na = (~np.isnan(hx)).sum(1), (~np.isnan(ax)).sum(1)
    ok = (nh >= min_players) & (na >= min_players) & (np.max(np.where(np.isnan(hx), -99, hx), 1) <= tol_m) & (np.min(np.where(np.isnan(ax), 99, ax), 1) >= -tol_m)
    m = np.concatenate([[False], ok, [False]]).astype(np.int8)
    d = np.diff(m)
    fr = t["frames"]
    runs = [(int(fr[a]), int(fr[min(b, len(fr) - 1)])) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if (b - a) * DT >= sustain_s]
    if not runs:
        return None, []
    end = runs[0][1]
    for a, b in runs[1:]:
        if a - end > cluster_gap_s * 25:
            break
        end = b
    return end, runs
