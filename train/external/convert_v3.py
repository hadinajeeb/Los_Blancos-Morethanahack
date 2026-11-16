"""Convert SkillCorner opendata (2024/25 A-League, tracking_extrapolated + phases_of_play) to the
common match format (ws/FORMAT.md).

Labels (derivation, no human in the loop on our side; SkillCorner's own products):
  H / A : frame lies inside a row of `<id>_phases_of_play.csv`, half-open [frame_start, frame_end);
          team = team_in_possession_id. The README states "Phases of play are only defined when
          the ball is in play. When the ball is out of play there is no phase of play."
  D     : frame is inside a match period (match.json `match_periods`) and inside no phase.
  X     : frame is outside both periods.

Extra keys beyond FORMAT.md (optional for consumers):
  home_det, away_det : bool (N, S)   SkillCorner `is_detected` (False = extrapolated position)
  ball_det           : bool (N,)     ball `is_detected`
  ball_z             : float32 (N,)
  trk_poss           : uint8 (N,)    per-frame `possession.group` of the tracking file
                                     (0 home, 1 away, 3 none) -- an independent second label source
  frame              : int32 (N,)    SkillCorner frame id (10 fps)

Usage: python convert_v3.py <opendata repo dir> <lfs dir with *_tracking_extrapolated.jsonl> <out dir> [match_id ...]
"""
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

H, A, D, X = 0, 1, 2, 3


def convert(args):
    repo, lfs, out, mid = args
    mdir = f"{repo}/data/matches/{mid}"
    meta = json.load(open(f"{mdir}/{mid}_match.json"))
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    hp = sorted(p["id"] for p in meta["players"] if p["team_id"] == home_id)
    ap = sorted(p["id"] for p in meta["players"] if p["team_id"] == away_id)
    hslot = {p: i for i, p in enumerate(hp)}
    aslot = {p: i for i, p in enumerate(ap)}

    frames = [json.loads(l) for l in open(f"{lfs}/{mid}_tracking_extrapolated.jsonl")]
    N = len(frames)
    fid = np.array([f["frame"] for f in frames], dtype=np.int32)
    assert np.array_equal(fid, np.arange(N)), "frame ids are not 0..N-1"

    home = np.full((N, len(hp), 2), np.nan, np.float32)
    away = np.full((N, len(ap), 2), np.nan, np.float32)
    hdet = np.zeros((N, len(hp)), bool)
    adet = np.zeros((N, len(ap)), bool)
    ball = np.full((N, 2), np.nan, np.float32)
    ballz = np.full(N, np.nan, np.float32)
    bdet = np.zeros(N, bool)
    tposs = np.full(N, X, np.uint8)
    unknown = 0
    for i, f in enumerate(frames):
        b = f["ball_data"]
        if b["x"] is not None:
            ball[i] = (b["x"], b["y"])
            ballz[i] = b["z"] if b["z"] is not None else np.nan
            bdet[i] = bool(b["is_detected"])
        g = f["possession"].get("group")
        if g == "home team":
            tposs[i] = H
        elif g == "away team":
            tposs[i] = A
        for p in f["player_data"]:
            pid = p["player_id"]
            if pid in hslot:
                home[i, hslot[pid]] = (p["x"], p["y"]); hdet[i, hslot[pid]] = bool(p["is_detected"])
            elif pid in aslot:
                away[i, aslot[pid]] = (p["x"], p["y"]); adet[i, aslot[pid]] = bool(p["is_detected"])
            else:
                unknown += 1

    # metres, origin at centre, y up  ->  normalised 0-1, origin top-left, y down (Metrica ATD)
    def norm(a):
        a = a.copy()
        a[..., 0] = (a[..., 0] + L / 2) / L
        a[..., 1] = (W / 2 - a[..., 1]) / W
        return a
    home, away, ball = norm(home), norm(away), norm(ball)

    half = np.zeros(N, np.uint8)
    home_dir = np.zeros(N, np.int8)
    sides = meta["home_team_side"]
    for k, per in enumerate(meta["match_periods"][:2]):
        a, b = per["start_frame"], min(per["end_frame"], N - 1)
        half[a:b + 1] = per["period"]
        home_dir[a:b + 1] = 1 if sides[k] == "left_to_right" else -1

    label = np.where(half > 0, D, X).astype(np.uint8)
    n_phase = 0
    with open(f"{mdir}/{mid}_phases_of_play.csv") as fh:
        for r in csv.DictReader(fh):
            a, b = int(r["frame_start"]), int(r["frame_end"])
            t = int(r["team_in_possession_id"])
            label[a:b] = H if t == home_id else A
            n_phase += 1
    label[half == 0] = X

    info = {
        "source": "SkillCorner opendata (github.com/SkillCorner/opendata), A-League 2024/25, broadcast tracking",
        "match_id": mid,
        "home": meta["home_team"]["short_name"], "away": meta["away_team"]["short_name"],
        "licence": "MIT (repo LICENSE); credit SkillCorner",
        "labels": "H/A from phases_of_play.csv team_in_possession (half-open frame ranges); "
                  "D = inside a match period and inside no phase; X = outside periods",
        "caveats": "tracking_extrapolated: every player has a position when the frame has data "
                   "(is_detected False = extrapolated); frames without data are broadcast cuts. "
                   "10 fps. Labels are SkillCorner's automatic Game Intelligence product, not hand labels.",
        "n_phases": n_phase, "unknown_player_rows": unknown,
    }
    os.makedirs(out, exist_ok=True)
    path = f"{out}/sc_{mid}.npz"
    np.savez_compressed(path, fps=np.float32(10.0), pitch=np.array([L, W], np.float32), home=home, away=away,
                        ball=ball, label=label, half=half, home_dir=home_dir, meta=json.dumps(info),
                        home_det=hdet, away_det=adet, ball_det=bdet, ball_z=ballz, trk_poss=tposs, frame=fid)
    return path, N, n_phase, unknown


if __name__ == "__main__":
    repo, lfs, out = sys.argv[1:4]
    mids = sys.argv[4:] or sorted(d for d in os.listdir(f"{repo}/data/matches") if d.isdigit())
    with ProcessPoolExecutor(10) as ex:
        for res in ex.map(convert, [(repo, lfs, out, m) for m in mids]):
            print(*res)
