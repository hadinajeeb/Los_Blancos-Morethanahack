"""Stage 5: posterior -> labels -> submission CSV.

1. Decoding: per half, the mean posterior of each pair of 25 fps frames (12.5 fps, the rate the
   switch costs were chosen at), then Viterbi over H/A/D with a fixed cost for every label change,
   repeated back to 25 fps. numpy on the CPU; ties resolve to the lowest label index (numpy
   argmax), so the path is a pure function of the posterior.
2. Dead-ball rules that hold by definition. Two variants, chosen in config.json:
   "kickoff_formation" (default): teams stand entirely in their own halves only while a kick-off
     is pending. Graded frames before a half's first such formation ends, and frames inside any
     such formation, are D. No frame number and no metadata of the match is used.
   "anchors_and_empty_tail" (first candidate files): graded frames before the organisers'
     published kick-off anchor are D, and so is an empty pitch that runs to the end of a half.
3. Frames outside the graded halves are X.
"""
import numpy as np

from .prep import FPS, HALVES, KICKOFF, N_FRAMES

H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])


def viterbi(logp, c_ha, c_d):
    """Most likely H/A/D path. c_ha: cost of a direct H<->A switch. c_d: cost of entering or leaving D."""
    n = len(logp)
    cost = np.array([[0, c_ha, c_d], [c_ha, 0, c_d], [c_d, c_d, 0]], np.float64)
    back = np.zeros((n, 3), np.int8)
    lp = logp.astype(np.float64)
    v = lp[0].copy()
    ar = np.arange(3)
    for t in range(1, n):
        cand = v[:, None] - cost
        back[t] = np.argmax(cand, axis=0)
        v = cand[back[t], ar] + lp[t]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(v))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def structural_mask(target):
    force = np.zeros(N_FRAMES, bool)
    n_h = (~np.isnan(target["home"][:, :, 0])).sum(1)
    n_a = (~np.isnan(target["away"][:, :, 0])).sum(1)
    empty = (n_h < 3) | (n_a < 3)
    info = {}
    for k, (s, e) in enumerate(HALVES, 1):
        ko = KICKOFF[k]
        if ko - 1 > s:
            force[s:ko - 1] = True          # one frame of tolerance on the anchor
        t = e; last_empty = None; hole = 0
        while t > s:                          # walk back from the half's end, allowing holes of up to 1 s
            if empty[t]:
                last_empty = t; hole = 0
            else:
                hole += 1
                if hole > FPS:
                    break
            t -= 1
        if last_empty is not None and e - last_empty >= FPS:
            force[last_empty:e + 1] = True
        info[k] = dict(pre_kickoff_frames=int(max(0, ko - 1 - s)),
                       empty_tail_frames=int(0 if last_empty is None or e - last_empty < FPS else e + 1 - last_empty))
    return force, info


def decode(post25, force=None, c_ha=4.0, c_d=4.0, round_decimals=None):
    """25 fps posterior (N, 3) -> labels (N,) with X outside the halves."""
    lab = np.full(N_FRAMES, X, np.int8)
    post = post25.astype(np.float64)
    if round_decimals is not None:
        post = np.round(post, round_decimals)
    add = None
    if force is not None:
        add = np.zeros((N_FRAMES, 3), np.float32)
        add[force, 2] = 1e3
    for a, b in HALVES:
        n = b + 1 - a
        m = n // 2
        lp = np.log(np.clip(post[a:a + 2 * m].reshape(m, 2, 3).mean(1), 1e-6, 1.0))
        if add is not None:
            lp = lp + add[a:a + 2 * m].astype(np.float64).reshape(m, 2, 3).mean(1)
        out = np.repeat(viterbi(lp, c_ha, c_d), 2)
        if n % 2:
            out = np.concatenate([out, out[-1:]])
        lab[a:b + 1] = out
    return lab


def write_csv(lab, path):
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.write("\n".join(f"{i},{c}" for i, c in enumerate(LETTERS[lab])))
        f.write("\n")


def check_csv(path):
    """The format rules of SUBMISSION.md. The organisers' validate_submission.py is the authority; run it too."""
    with open(path) as f:
        rows = f.read().split("\n")
    assert rows[0] == "frame,label", rows[0]
    body = rows[1:-1] if rows[-1] == "" else rows[1:]
    assert len(body) == N_FRAMES, len(body)
    for i, r in enumerate(body):
        fr, lab = r.split(",")
        assert int(fr) == i and lab in "HADX" and len(lab) == 1, (i, r)
    return True


def decode_pairmean32(post25, c_ha=4.0, c_d=4.0):
    """The decode of the v2 files: pair mean and log in float32, Viterbi in float64."""
    lab = np.full(N_FRAMES, X, np.int8)
    for a, b in HALVES:
        n = b + 1 - a
        k = n // 2
        p = post25[a:a + 2 * k].reshape(k, 2, 3).mean(1)
        path = viterbi(np.log(np.clip(p, 1e-6, 1.0)), c_ha, c_d)
        out = np.repeat(path, 2)
        if n % 2:
            out = np.concatenate([out, out[-1:]])
        lab[a:b + 1] = out
    return lab


def kickoff_formations(view, dt, min_players=8, tol_m=1.5, sustain_s=0.5, cluster_gap_s=30.0):
    """Kick-off formations in one half: (end frame of the first cluster, list of runs), in video frames.

    `view` is one half from seq.load_target (metres, home attacks +x). Formation: at least
    `min_players` tracked per team, every home player in the half home defends (x <= tol) and
    every away player in the other (x >= -tol), for at least `sustain_s`. Runs less than
    `cluster_gap_s` apart belong to one kick-off (a player steps over the line and back).
    """
    hx, ax = view["home"][..., 0], view["away"][..., 0]
    nh, na = (~np.isnan(hx)).sum(1), (~np.isnan(ax)).sum(1)
    ok = (nh >= min_players) & (na >= min_players) & (np.max(np.where(np.isnan(hx), -99, hx), 1) <= tol_m) \
        & (np.min(np.where(np.isnan(ax), 99, ax), 1) >= -tol_m)
    m = np.concatenate([[False], ok, [False]]).astype(np.int8)
    d = np.diff(m)
    fr = view["frames"]
    runs = [(int(fr[a]), int(fr[min(b, len(fr) - 1)])) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if (b - a) * dt >= sustain_s]
    if not runs:
        return None, []
    end = runs[0][1]
    for a, b in runs[1:]:
        if a - end > cluster_gap_s * 25:
            break
        end = b
    return end, runs


def apply_kickoff_rule(lab, views, dt):
    """Set D before each half's first kick-off and inside every kick-off formation. Returns a summary."""
    info = {}
    for k, ((x, y), view) in enumerate(zip(HALVES, views), 1):
        ko, runs = kickoff_formations(view, dt)
        if ko is not None and ko > x:
            lab[x:min(ko, y + 1)] = D
        for r0, r1 in runs:
            lab[max(r0, x):min(r1, y + 1)] = D
        info[k] = dict(first_kickoff_minus_half_start=None if ko is None else int(ko - x), formations=len(runs))
    return info


def labels_from_posterior(cfg, post, target):
    """Posterior -> labels with the decode style and dead-ball rules named in the config."""
    from . import seq
    d = cfg["decode"]
    if cfg["dead_rules"] == "anchors_and_empty_tail":
        assert d["style"] == "pairmean64"
        force, info = structural_mask(target)
        return decode(post, force, d["c_ha"], d["c_d"]), info
    assert cfg["dead_rules"] == "kickoff_formation" and d["style"] == "pairmean32"
    lab = decode_pairmean32(post, d["c_ha"], d["c_d"])
    stride = int(round(25.0 / cfg["seq_fps"]))
    views = seq.load_target(target, target["ball"][:, :2], 0, stride)
    info = apply_kickoff_rule(lab, views, 1.0 / cfg["seq_fps"])
    return lab, info
