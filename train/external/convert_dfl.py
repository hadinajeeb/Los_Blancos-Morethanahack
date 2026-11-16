"""Convert one DFL/Sportec open-data match to the common format (ws/FORMAT.md).

Labels: every ball frame carries BallPossession (1 = home, 2 = guest) and BallStatus (1 = alive,
0 = dead). label = D where BallStatus == 0, else H/A from BallPossession. No derivation beyond that.

Coordinates: DFL is metres, origin at the centre spot, y up. Common format is 0-1, origin top-left:
    x = X / length + 0.5        y = 0.5 - Y / width

Usage: python convert_dfl.py <raw dir> <match id e.g. DFL-MAT-J03WMX> <out.npz>
"""
import glob
import json
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

raw, mid, out = sys.argv[1:4]
info = glob.glob(f"{raw}/DFL_02_01_matchinformation_*_{mid}.xml")[0]
pos = glob.glob(f"{raw}/DFL_04_03_positions_raw_observed_*_{mid}.xml")[0]

g = ET.parse(info).getroot().find("MatchInformation")
gen = g.find("General")
env = g.find("Environment")
home_id, away_id = gen.get("HomeTeamId"), gen.get("GuestTeamId")
length, width = float(env.get("PitchX")), float(env.get("PitchY"))

FS = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)"')
FR = re.compile(r'<Frame N="(\d+)" T="[^"]*" X="([-\d.]+)" Y="([-\d.]+)"')
BALL = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')

# pass 1 would cost a second read of 400 MB; instead collect per-object lists and assemble after
sets = []  # (section, team, person, N array, X array, Y array, [poss, status])
cur = None
with open(pos, encoding="utf-8") as f:
    for line in f:
        if line.startswith("<Frame "):
            m = FR.match(line)
            cur[3].append(int(m.group(1))); cur[4].append(float(m.group(2))); cur[5].append(float(m.group(3)))
            if cur[1] == "BALL":
                b = BALL.search(line)
                cur[6].append(int(b.group(1))); cur[7].append(int(b.group(2)))
        elif line.startswith("<FrameSet"):
            m = FS.match(line)
            cur = [m.group(1), m.group(2), m.group(3), [], [], [], [], []]
            sets.append(cur)

SECT = {"firstHalf": 1, "secondHalf": 2}
sets = [s for s in sets if s[0] in SECT]  # no extra time in this dataset; guard anyway
balls = {SECT[s[0]]: s for s in sets if s[1] == "BALL"}
frames = {h: np.asarray(balls[h][3]) for h in (1, 2)}
for h in (1, 2):
    assert np.all(np.diff(frames[h]) == 1), "ball frames not contiguous"
n1, n2 = len(frames[1]), len(frames[2])
N = n1 + n2
base = {1: (frames[1][0], 0), 2: (frames[2][0], n1)}

persons = {home_id: [], away_id: []}
for s in sets:
    if s[1] in persons and s[2] not in persons[s[1]]:
        persons[s[1]].append(s[2])
arr = {t: np.full((N, len(persons[t]), 2), np.nan, np.float32) for t in persons}
ball = np.full((N, 2), np.nan, np.float32)
poss = np.zeros(N, np.uint8); status = np.zeros(N, np.uint8)
half = np.zeros(N, np.uint8); half[:n1] = 1; half[n1:] = 2
dropped = 0
for sec, team, person, n, x, y, bp, bs in sets:
    h = SECT[sec]
    f0, off = base[h]
    idx = np.asarray(n) - f0
    ok = (idx >= 0) & (idx < len(frames[h]))
    dropped += int((~ok).sum())
    idx = idx[ok] + off
    xn = (np.asarray(x)[ok] / length + 0.5).astype(np.float32)
    yn = (0.5 - np.asarray(y)[ok] / width).astype(np.float32)
    if team == "BALL":
        ball[idx, 0] = xn; ball[idx, 1] = yn
        poss[idx] = np.asarray(bp)[ok]; status[idx] = np.asarray(bs)[ok]
    elif team in persons:
        j = persons[team].index(person)
        arr[team][idx, j, 0] = xn; arr[team][idx, j, 1] = yn

label = np.where(status == 0, 2, np.where(poss == 1, 0, np.where(poss == 2, 1, 3))).astype(np.uint8)
home_dir = np.zeros(N, np.int8)
for h, sl in ((1, slice(0, n1)), (2, slice(n1, N))):
    mh = np.nanmean(arr[home_id][sl, :, 0]); ma = np.nanmean(arr[away_id][sl, :, 0])
    home_dir[sl] = 1 if mh < ma else -1

meta = dict(source="DFL/Sportec open dataset, Bassek et al. 2025, doi:10.6084/m9.figshare.28196177.v1",
            licence="CC BY 4.0", match_id=mid, title=gen.get("MatchTitle"), competition=gen.get("CompetitionName"),
            result=gen.get("Result"),
            labels="direct: D where BallStatus=0, else H if BallPossession=1 (home), A if 2 (guest)",
            caveats="clean optical tracking (TRACAB), persistent identities, ball always present; "
                    "frames cover the two halves only; frames_dropped_outside_ball_span=%d" % dropped,
            frame_n_first_half_start=int(frames[1][0]), frame_n_second_half_start=int(frames[2][0]))
np.savez_compressed(out, fps=np.float32(25.0), pitch=np.asarray([length, width], np.float32),
                    home=arr[home_id], away=arr[away_id], ball=ball, label=label, half=half,
                    home_dir=home_dir, meta=json.dumps(meta),
                    ball_possession_raw=poss, ball_status_raw=status)
print(mid, gen.get("MatchTitle"), "N", N, "halves", n1, n2, "home", arr[home_id].shape, "away", arr[away_id].shape,
      "label counts H/A/D/X", np.bincount(label, minlength=4).tolist(), "home_dir", home_dir[0], home_dir[-1],
      "poss values", np.unique(poss).tolist(), "dropped", dropped)
