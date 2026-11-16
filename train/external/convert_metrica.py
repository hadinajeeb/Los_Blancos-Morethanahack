"""Convert the three Metrica sample games to the common format (ws/FORMAT.md).

The sample data has no per-frame possession or ball-status field. Labels are DERIVED from the
events and from the ball's NaN stretches. See derive_labels() for the exact rule.

Usage: python convert_metrica.py <sample-data dir> <out dir>
"""
import csv
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np

ONBALL = {"PASS", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "BALL OUT", "CARRY"}
CONTROL = {"PASS", "SHOT", "CARRY", "SET PIECE"}
LIVE_MAX = 75        # a NaN run shorter than 3 s with no stoppage evidence is treated as live play
TOUCH_MAX = 12       # label_controlled: a stint that loses the ball within 0.5 s of its first touch


def is_stop(e):
    t, s = e["type"], e["sub"]
    if t in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return True
    if t == "BALL LOST" and any(k in s for k in ("OFFSIDE", "HAND BALL", "END HALF")):
        return True
    if t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s:
        return True
    return False


def is_restart(e):
    return e["type"] == "SET PIECE" or "GOAL KICK" in e["sub"] or "KICK OFF" in e["sub"]


def runs(mask):
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return np.where(d == 1)[0], np.where(d == -1)[0]


def derive_labels(ball_nan, period, events):
    """events: dicts with team in {0 home, 1 away}, type, sub (str), start, end (1-based frames).

    Dead ball. In this data the ball position is NaN (a) through stoppages and (b) for a few frames
    between a BALL LOST and the RECOVERY that follows, which is live play. A NaN run is D when the
    last event before it is a stoppage (ball out, foul, shot out or goal, offside, hand ball, end of
    half), or the first event after it is a restart (set piece, goal kick, kick-off), or it lasts
    3 s or more. Otherwise it is live. NaN runs that touch the start or end of a period are X: they
    are pre-kick-off or post-whistle time, not a stoppage inside play.

    Owner. For live frames the owner is the team of the most recent on-ball event (PASS, SHOT,
    RECOVERY, SET PIECE, BALL LOST, BALL OUT, CARRY), switched at that event's start frame.
    CHALLENGE, FAULT RECEIVED and CARD do not move the owner.

    label_controlled. Same, except that a stint with no PASS/SHOT/CARRY/SET PIECE whose ball is gone
    within 0.5 s of its first touch (an interception that goes straight back) stays with the
    previous owner. This mirrors the challenge text ("a simple touch without control" is not a change).
    """
    N = len(ball_nan)
    ev = [e for e in events if e["type"] != "CARD"]
    order = sorted(range(len(ev)), key=lambda i: (ev[i]["start"], i))
    ev = [ev[i] for i in order]
    starts = np.array([e["start"] for e in ev])

    dead = np.zeros(N, bool); xmask = np.zeros(N, bool)
    stats = dict(nan_runs=0, dead_runs=0, live_runs=0, ambiguous_runs=0)
    for a, b in zip(*runs(ball_nan)):
        stats["nan_runs"] += 1
        edge = a == 0 or b == N or period[a - 1] != period[min(b, N - 1)] or period[a] != period[b - 1]
        if edge:
            xmask[a:b] = True
            continue
        i = np.searchsorted(starts, a + 1, side="right") - 1
        j = np.searchsorted(starts, b + 1, side="left")
        pre = ev[i] if i >= 0 else None
        post = ev[j] if j < len(ev) else None
        evid = (pre is not None and is_stop(pre)) or (post is not None and is_restart(post))
        if evid or (b - a) >= LIVE_MAX:
            dead[a:b] = True; stats["dead_runs"] += 1
            if not evid:
                stats["ambiguous_runs"] += 1
        else:
            stats["live_runs"] += 1
            if (b - a) > 30:
                stats["ambiguous_runs"] += 1

    def owner_array(skip):
        own = np.full(N, -1, np.int8)
        for k, e in enumerate(ev):
            if e["type"] in ONBALL and k not in skip:
                own[e["start"] - 1:] = e["team"]
        first = np.argmax(own >= 0)
        own[:first] = own[first]
        return own

    own = owner_array(set())
    # stints of the on-ball event sequence, split by dead time
    on = [k for k, e in enumerate(ev) if e["type"] in ONBALL]
    skip = set(); stint = []
    def close(stint):
        if not stint:
            return
        es = [ev[k] for k in stint]
        if any(e["type"] in CONTROL for e in es):
            return
        losses = [e["start"] for e in es if e["type"] in ("BALL LOST", "BALL OUT")]
        if losses and max(losses) - es[0]["start"] <= TOUCH_MAX:
            skip.update(stint)
    prev = None
    for k in on:
        e = ev[k]
        new = prev is None or e["team"] != ev[prev]["team"] or dead[ev[prev]["start"] - 1:e["start"]].any()
        if new:
            # a stint that opens a live phase (follows dead time) is never skipped: there is no previous owner
            if stint and not stint_opens_phase:
                close(stint)
            stint = []
            stint_opens_phase = prev is None or dead[ev[prev]["start"] - 1:e["start"]].any()
        stint.append(k); prev = k
    if stint and not stint_opens_phase:
        close(stint)
    own_c = owner_array(skip)

    def lab(o):
        l = o.astype(np.uint8)
        l[dead] = 2
        l[xmask] = 3
        return l
    stats["skipped_touch_events"] = len(skip)
    return lab(own), lab(own_c), stats


def finish(name, out_dir, home, away, ball, period, events, extra_meta):
    N = len(ball)
    label, label_c, stats = derive_labels(np.isnan(ball[:, 0]), period, events)
    half = period.astype(np.uint8)
    home_dir = np.zeros(N, np.int8)
    for h in (1, 2):
        k = half == h
        home_dir[k] = 1 if np.nanmean(home[k, :, 0]) < np.nanmean(away[k, :, 0]) else -1
    meta = dict(source="Metrica Sports sample-data, github.com/metrica-sports/sample-data @ e706dd5",
                licence="no formal licence; README asks for responsible use and acknowledgement",
                match_id=name,
                labels="DERIVED from events + ball NaN runs, see convert_metrica.py derive_labels(); "
                       "label = event-owner variant, label_controlled = one-touch stints kept with previous owner",
                caveats="clean corrected tracking with identities; ball is NaN through stoppages (unlike the "
                        "target's automatic tracking); halftime is not in the data; derivation stats: %s" % json.dumps(stats),
                **extra_meta)
    np.savez_compressed(f"{out_dir}/{name}.npz", fps=np.float32(25.0), pitch=np.asarray([105.0, 68.0], np.float32),
                        home=home.astype(np.float32), away=away.astype(np.float32), ball=ball.astype(np.float32),
                        label=label, label_controlled=label_c, half=half, home_dir=home_dir, meta=json.dumps(meta))
    print(name, "N", N, "home", home.shape, "away", away.shape, "label H/A/D/X", np.bincount(label, minlength=4).tolist(),
          "controlled", np.bincount(label_c, minlength=4).tolist(), stats, "home_dir", home_dir[0], home_dir[-1])


def csv_game(root, g, out_dir):
    d = f"{root}/data/Sample_Game_{g}"
    H = np.genfromtxt(f"{d}/Sample_Game_{g}_RawTrackingData_Home_Team.csv", delimiter=",", skip_header=3)
    A = np.genfromtxt(f"{d}/Sample_Game_{g}_RawTrackingData_Away_Team.csv", delimiter=",", skip_header=3)
    assert np.array_equal(H[:, 1], A[:, 1]) and np.array_equal(H[:, 1], np.arange(1, len(H) + 1))
    period = H[:, 0].astype(int)
    home = H[:, 3:-2].reshape(len(H), -1, 2); away = A[:, 3:-2].reshape(len(A), -1, 2)
    ball = H[:, -2:]
    events = []
    for r in csv.DictReader(open(f"{d}/Sample_Game_{g}_RawEventsData.csv")):
        s, e = int(r["Start Frame"]), int(r["End Frame"])
        events.append(dict(team=0 if r["Team"] == "Home" else 1, type=r["Type"], sub=r["Subtype"], start=s, end=max(e, s)))
    finish(f"metrica_game{g}", out_dir, home, away, ball, period, events, dict(format="CSV"))


def epts_game(root, out_dir):
    d = f"{root}/data/Sample_Game_3"
    root_x = ET.parse(f"{d}/Sample_Game_3_metadata.xml").getroot()
    md = root_x.find("Metadata")
    team_ids = [t.get("id") for t in md.find("Teams")]
    pteam = {p.get("id"): p.get("teamId") for p in md.find("Players")}
    chan_player = {}
    for pc in md.iter("PlayerChannel"):
        chan_player[pc.get("id")] = pc.get("playerId")
    params = {p.findtext("Name"): p.findtext("Value") for p in md.find("GlobalConfig").find("ProviderGlobalParameters")}
    h1 = (int(params["first_half_start"]), int(params["first_half_end"]))
    h2 = (int(params["second_half_start"]), int(params["second_half_end"]))
    specs = []
    for spec in root_x.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        players = [chan_player[c] for c in refs if c.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), players))
    all_players = {t: [p for p, tt in pteam.items() if tt == t] for t in team_ids}
    lines = open(f"{d}/Sample_Game_3_tracking.txt").read().splitlines()
    N = len(lines)
    arr = {t: np.full((N, len(all_players[t]), 2), np.nan) for t in team_ids}
    ball = np.full((N, 2), np.nan)
    si = 0
    for i, line in enumerate(lines):
        fid, pl, bl = line.split(":")
        fid = int(fid); assert fid == i + 1
        while not (specs[si][0] <= fid <= specs[si][1]):
            si += 1
        players = specs[si][2]
        slots = pl.split(";")
        assert len(slots) == len(players), (fid, len(slots), len(players))
        for p, s in zip(players, slots):
            x, y = s.split(",")
            t = pteam[p]
            arr[t][i, all_players[t].index(p)] = (float(x), float(y))
        bx = bl.split(",")
        ball[i] = (float(bx[0]), float(bx[1]))
    period = np.zeros(N, int)
    period[h1[0] - 1:h1[1]] = 1; period[h2[0] - 1:h2[1]] = 2
    events = []
    for e in json.load(open(f"{d}/Sample_Game_3_events.json"))["data"]:
        st = e["subtypes"]
        sub = "" if not st else (st["name"] if isinstance(st, dict) else "-".join(s["name"] for s in st))
        s, en = int(e["start"]["frame"]), int(e["end"]["frame"])
        events.append(dict(team=team_ids.index(e["team"]["id"]), type=e["type"]["name"], sub=sub, start=s, end=max(en, s)))
    finish("metrica_game3", out_dir, arr[team_ids[0]], arr[team_ids[1]], ball, period, events,
           dict(format="EPTS + JSON events", n_format_specs=len(specs)))


if __name__ == "__main__":
    root, out_dir = sys.argv[1:3]
    csv_game(root, 1, out_dir)
    csv_game(root, 2, out_dir)
    epts_game(root, out_dir)
