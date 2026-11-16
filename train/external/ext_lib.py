"""Loader and scoring helpers for matches in the common format (ws/FORMAT.md).

    from ext_lib import load, list_matches, runs_of, score, nearest_team, baseline_labels

Scoring uses the official code in official-repo/evaluation/possession_metrics.py, with one change:
the runs are split at half boundaries as well as at X frames. The official valid_runs() splits only
at X, which is enough for the target (halftime is X) but not for external matches whose two halves
are stored back to back.
"""
import glob
import json
import os
import sys

import numpy as np

P = os.environ.get("MTAH_ROOT", ".")
sys.path.insert(0, f"{P}/official-repo/evaluation")
sys.path.insert(0, f"{P}/official-repo/starter")
sys.path.insert(0, f"{P}/official-repo")
import possession_metrics as pm  # noqa: E402

H, A, D, X = 0, 1, 2, 3


def list_matches(dataset=None):
    """Paths of converted external matches. dataset in {None, 'dfl', 'metrica', ...}."""
    pat = f"{P}/external/{dataset or '*'}/common/*.npz"
    return sorted(glob.glob(pat))


def load(path, label_key="label"):
    """dict with fps, pitch, home, away, ball, label, half, home_dir, meta (parsed JSON), name."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    d["fps"] = float(d["fps"])
    d["name"] = os.path.basename(path)[:-4]
    if label_key != "label":
        d["label"] = d[label_key]
    return d


def runs_of(label, half):
    """Half-open [start, stop) spans of graded frames, never crossing a half boundary."""
    key = np.where(label != X, half.astype(np.int16), -1)
    edges = np.flatnonzero(np.diff(key) != 0) + 1
    bounds = np.concatenate(([0], edges, [len(key)]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if key[a] >= 0]


def score(gt, pred, half):
    """spellF1 (mean over IoU 0.3-0.7), macroF1, per-class F1, spell counts. Official functions."""
    runs = runs_of(gt, half)
    filled = pm.fill_abstentions(pred.astype(np.int8), runs)
    sp = pm.spell_scores(gt, filled, runs)
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    out = dict(spell_f1=sp["mean_f1"], macro_f1=pm.macro_f1(cm3), n_gt=sp["n_gt"], n_pred=sp["n_pred"],
               per_iou={k: v["f1"] for k, v in sp["per_iou"].items()})
    for k, name in enumerate(pm.SCORED):
        out["f1_" + name] = pm.per_class_prf(cm3, k)[2]
    return out


def nearest_team(m, ball=None):
    """Per frame: team (0 home, 1 away) of the player nearest the ball in metres, -1 without a ball.
    Also returns the distance of the nearest home and nearest away player (metres, inf if none)."""
    ball = m["ball"] if ball is None else ball
    L, W = float(m["pitch"][0]), float(m["pitch"][1])
    out = []
    for team in ("home", "away"):
        p = m[team]
        d2 = ((p[:, :, 0] - ball[:, None, 0]) * L) ** 2 + ((p[:, :, 1] - ball[:, None, 1]) * W) ** 2
        d2 = np.where(np.isnan(d2), np.inf, d2)
        out.append(np.sqrt(d2.min(axis=1)))
    dh, da = out
    team = np.where(dh <= da, 0, 1).astype(np.int8)
    team[np.isnan(ball[:, 0]) | (np.isinf(dh) & np.isinf(da))] = -1
    return team, dh, da


def baseline_labels(m, window, ball=None):
    """The official baseline on a common-format match: nearest player, then the official smooth()."""
    from baseline import smooth  # official majority vote, unchanged
    codes, _, _ = nearest_team(m, ball)
    letters = smooth(codes, window)
    return np.where(letters == "H", H, np.where(letters == "A", A, D)).astype(np.int8)
