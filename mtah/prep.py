"""Stage 1: read the provided bundle into arrays.

Inputs (tactical source of the data bundle):
    <data>/tactical/BAR-ATH_tactical_atd.txt     raw tracking, one line per frame
    <data>/tactical/BAR-ATH_tactical_atd.xml     metadata: slot -> team, goalkeeper slots, direction
    <data>/tactical/BAR-ATH_tactical_homography.csv
Outputs in the work directory:
    target.npz       home (N, Sh, 2), away (N, Sa, 2), ball (N, 2), half (N,), home_dir (N,), pitch (2,)
    homography.npy   (N, 3, 3) float64, normalised pitch -> pixels

Row format of the ATD text: frame : p1x,p1y,p1s ; ... ; pNx,pNy,pNs : ballx,bally,ballz
Positions are normalised 0-1. Missing values are the string NaN. Row i is video frame i.

Frame ranges of the two halves come from the organisers' games.py (vendored, MIT). No other frame
number of the match appears anywhere in this pipeline.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from .third_party import games

G = games.GAMES[games.GAME]
N_FRAMES = G["n_video_frames"]
FPS = G["fps"]
HALVES = tuple((int(a), int(b)) for _, a, b in games.half_video_blocks(games.GAME))   # video frames, inclusive
KICKOFF = {1: games.anchor_video_frame(G["kickoff"]), 2: games.anchor_video_frame(G["kickoff_2nd"])}


def paths(data_dir):
    return games.asset_paths(games.GAME, data_dir, "tactical")


def parse_atd(path):
    frames, players, ball = [], [], []
    with open(path) as f:
        for line in f:
            fid, pl, bl = line.rstrip("\n").split(":")
            frames.append(int(fid))
            players.append([float(v) for slot in pl.split(";") for v in slot.split(",")])
            ball.append([float(v) for v in bl.split(",")])
    frames = np.asarray(frames, dtype=np.int32)
    players = np.asarray(players, dtype=np.float32).reshape(len(frames), -1, 3)
    ball = np.asarray(ball, dtype=np.float32)
    return frames, players, ball


def build_target(data_dir):
    """Team arrays for the tactical tracking, with the keeper-slot repair. No labels are read."""
    p = paths(data_dir)
    fid, P, B = parse_atd(p["atd_raw"])
    N = len(P)
    assert N == N_FRAMES, (N, N_FRAMES)
    assert np.array_equal(fid, np.arange(1, N + 1)), "ATD rows are not frames 1..N in order"

    x = ET.parse(p["atd_meta"]).getroot()
    names = {t.get("id"): t.findtext("Name") for t in x.iter("Team")}
    direction = {}
    for t in x.iter("Team"):
        for q in t.iter("ProviderParameter"):
            if q.findtext("Name") == "attack_direction_first_half":
                direction[t.get("id")] = q.findtext("Value")
    slot_team = np.array([pl.get("teamId") for pl in x.iter("Player")])
    home_id = x.find(".//Score").get("idLocalTeam")
    away_id = x.find(".//Score").get("idVisitingTeam")
    assert len(slot_team) == P.shape[1], (len(slot_team), P.shape)
    hs, as_ = np.flatnonzero(slot_team == home_id), np.flatnonzero(slot_team == away_id)
    gk = []
    for pl in x.iter("Player"):
        typ = [q.findtext("Value") for q in pl.iter("ProviderParameter") if q.findtext("Name") == "position_type"]
        gk.append(bool(typ) and typ[0] == "Goalkeeper")
    gk = np.array(gk)

    half = np.zeros(N, np.uint8)
    for k, (a, b) in enumerate(HALVES, 1):
        half[a:b + 1] = k
    first = +1 if direction[home_id] == "left_to_right" else -1
    home_dir = np.zeros(N, np.int8)
    home_dir[half == 1] = first
    home_dir[half == 2] = -first

    # Keeper-slot repair. The tracker holds or invents the keeper slots when they are off screen,
    # and a keeper slot can sit on the other team's keeper. A keeper slot standing within 25 m of a
    # goal line belongs to the team that defends that goal; if that is the other team, the position
    # moves to a spare slot of that team. A general rule: no frame of the match is named.
    home_xy = P[:, hs, :2].copy()
    away_xy = P[:, as_, :2].copy()
    moved = {}
    for slot in np.flatnonzero(gk):
        own_home = slot_team[slot] == home_id
        xs = P[:, slot, 0]
        home_goal_x = np.where(home_dir > 0, 0.0, 1.0)
        near_home_goal = np.abs(xs - home_goal_x) < 25 / 105.3
        near_away_goal = np.abs(xs - (1 - home_goal_x)) < 25 / 105.3
        wrong = (home_dir != 0) & (near_away_goal if own_home else near_home_goal)
        src = home_xy if own_home else away_xy
        dst = away_xy if own_home else home_xy
        j_src = int(np.flatnonzero((hs if own_home else as_) == slot)[0])
        spare = int(np.argmin((~np.isnan(dst[..., 0])).sum(0)))
        free = wrong & np.isnan(dst[:, spare, 0])
        dst[free, spare] = src[free, j_src]
        src[wrong, j_src] = np.nan
        moved[int(slot)] = int(wrong.sum())
    info = dict(home=names[home_id], away=names[away_id], home_slots=len(hs), away_slots=len(as_),
                keeper_slots=np.flatnonzero(gk).tolist(), keeper_frames_reassigned=moved)
    return dict(home=home_xy, away=away_xy, ball=B[:, :2].copy(), half=half, home_dir=home_dir,
                pitch=np.asarray(G["pitch"], np.float32)), info


def load_homography_csv(path):
    t = np.genfromtxt(path, delimiter=",", skip_header=1)
    return t[:, 1:10].reshape(-1, 3, 3)


def run(data_dir, work):
    out = os.path.join(work, "target.npz")
    hom = os.path.join(work, "homography.npy")
    if not os.path.exists(out):
        t, info = build_target(data_dir)
        np.savez_compressed(out + ".tmp.npz", **t)
        os.replace(out + ".tmp.npz", out)
        print("prep:", info, flush=True)
    if not os.path.exists(hom):
        M = load_homography_csv(paths(data_dir)["homography"])
        assert len(M) == N_FRAMES
        np.save(hom + ".tmp.npy", M)
        os.replace(hom + ".tmp.npy", hom)
    return out, hom
