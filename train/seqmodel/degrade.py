"""Make a clean external match look like the target's automatic tracking.

Everything here is drawn from random distributions (domain randomisation). The ranges were set
from input statistics of the target (ball-gap lengths, players per frame, track lengths, camera
coverage) and from what the video check showed about the tracker. No target label is used.

Why the ball is re-simulated on dead-ball frames: DFL freezes the ball (speed exactly 0) on every
D frame and Metrica sets it to NaN. The target does neither. Left alone, either would be a label
leak that does not exist in the target.
"""
import numpy as np
from scipy.signal import lfilter

from common import D, DT, FPS, X

L_HALF, W_HALF = 52.5, 34.0


def _runs(mask):
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _ou(rng, n, sigma, tau_s, dims=2):
    """Zero-mean AR(1) noise with stationary std sigma and correlation time tau_s seconds."""
    rho = np.exp(-DT / max(tau_s, 1e-3))
    eps = rng.standard_normal((n, dims)) * sigma * np.sqrt(1 - rho ** 2)
    eps[0] = rng.standard_normal(dims) * sigma
    return lfilter([1.0], [1.0, -rho], eps, axis=0).astype(np.float32)


def _lognorm(rng, median, sigma, lo, hi, size=None):
    return np.clip(median * np.exp(sigma * rng.standard_normal(size)), lo, hi)


def _fill_nan_linear(b):
    """Linear interpolation over NaN runs (per column); edges take the nearest valid value."""
    out = b.copy()
    idx = np.arange(len(b))
    for c in range(b.shape[1]):
        ok = ~np.isnan(b[:, c])
        if ok.any():
            out[:, c] = np.interp(idx, idx[ok], b[ok, c])
        else:
            out[:, c] = 0.0
    return out


def simulate_dead_ball(ball, label, rng, rest_scale=1.0):
    """True-ish ball path through every D spell: roll on, get carried to the restart spot, rest.

    Returns a full ball track with no NaN (visibility is decided later)."""
    b = _fill_nan_linear(ball)
    n = len(b)
    s_arr, e_arr = _runs(label == D)
    for s, e in zip(s_arr, e_arr):
        T = e - s
        p0 = b[s - 1] if s > 0 else b[min(e, n - 1)]
        v0 = (b[s - 1] - b[s - 2]) / DT if s > 1 else np.zeros(2, np.float32)
        sp = np.hypot(*v0)
        if sp > 25:
            v0 = v0 * 25 / sp
        p1 = b[e] if e < n else p0
        t = np.arange(T) * DT
        tau = rng.uniform(0.4, 1.2)
        t_roll = min(rng.uniform(0.5, 3.0), T * DT * 0.5)
        k_roll = max(1, int(round(t_roll * FPS)))
        path = np.empty((T, 2), np.float32)
        roll = p0[None] + v0[None] * tau * (1 - np.exp(-t[:k_roll, None] / tau))
        path[:k_roll] = roll
        q0 = roll[-1]
        rest = T - k_roll
        if rest > 0:
            dist = float(np.hypot(*(p1 - q0)))
            k_move = int(np.clip(round(dist / rng.uniform(2.0, 10.0) * FPS), 1, rest))
            k_wait = int(rng.integers(0, rest - k_move + 1) * rest_scale)  # ball lies still before it is fetched
            w = np.zeros(rest, np.float32)
            ramp = (np.arange(k_move) + 1) / k_move
            w[k_wait:k_wait + k_move] = ramp * ramp * (3 - 2 * ramp)  # smoothstep
            w[k_wait + k_move:] = 1.0
            path[k_roll:] = q0[None] * (1 - w[:, None]) + p1[None] * w[:, None]
            # the carrier does not walk a straight line
            path[k_roll:] += _ou(rng, rest, rng.uniform(0.0, 0.6), 2.0) * (w * (1 - w) * 4)[:, None]
        b[s:e] = path
    return b


def ball_visibility(label, rng, cfg, b_true=None, players=None):
    """Boolean mask: True where the simulated tracker reports a ball."""
    n = len(label)
    vis = np.ones(n, bool)
    live = label < D
    s_arr, e_arr = _runs(label == D)
    for s, e in zip(s_arr, e_arr):
        T = (e - s) * DT
        if rng.random() < cfg["p_vis_all"] * np.exp(-T / 8.0) or rng.random() < cfg["p_vis_any"]:
            continue
        loose = cfg.get("gap_loose", False)
        a = s + int(round(rng.uniform(*((-4.0, 8.0) if loose else (-2.0, 4.0))) * FPS))
        if rng.random() < cfg["p_late"]:
            z = e + int(round(rng.uniform(0.5, 8.0 if loose else 6.0) * FPS))       # reacquired after the restart
        else:
            z = e + int(round(rng.uniform(-8.0 if loose else -4.0, 0.5) * FPS))     # reacquired before or at the restart
        if rng.random() < cfg["p_early"]:
            a = s - int(round(rng.uniform(1.0, 6.0) * FPS))       # lost before play stopped
        a, z = max(a, 0), min(z, n)
        if z - a < 3:
            continue
        vis[a:z] = False
        if T >= 15 and rng.random() < cfg["p_island"]:            # ball seen for a while mid-stoppage
            k = int(round(rng.uniform(1.0, 8.0) * FPS))
            c = int(rng.integers(a, max(a + 1, z - k)))
            vis[c:c + k] = True
    # gaps in live play
    n_live_min = live.sum() * DT / 60.0
    for _ in range(rng.poisson(cfg["live_gap_rate"] * n_live_min)):
        c = int(rng.integers(0, n))
        if not live[c]:
            continue
        k = int(round(_lognorm(rng, cfg["live_gap_median"], 1.0, 0.3, 40.0) * FPS))
        vis[max(0, c - k // 2):c + (k + 1) // 2] = False
    if b_true is not None and cfg.get("p_out_drop", 0) > 0:
        # a ball that has left the pitch is rarely tracked
        mg = cfg["out_margin"]
        outside = (np.abs(b_true[:, 0]) > L_HALF + mg) | (np.abs(b_true[:, 1]) > W_HALF + mg)
        for s0, e0 in zip(*_runs(outside)):
            if rng.random() < cfg["p_out_drop"]:
                vis[s0:e0] = False
    if b_true is not None and players is not None and cfg.get("crowd_drop", 0) > 0:
        # the ball is lost in crowds (duels, goalmouth scrambles), whatever the game state
        d = np.hypot(players[..., 0] - b_true[:, None, 0], players[..., 1] - b_true[:, None, 1])
        crowd = (np.where(np.isnan(d), 99.0, d) < 2.5).sum(1) >= 3
        pad = int(round(0.5 * FPS))
        for s0, e0 in zip(*_runs(crowd)):
            if rng.random() < cfg["crowd_drop"]:
                vis[max(0, s0 - int(rng.integers(0, pad + 1))):e0 + int(rng.integers(0, 2 * pad + 1))] = False
    if cfg.get("fill_short", 0) > 0:
        # a smoothing tracker bridges brief losses
        for s0, e0 in zip(*_runs(~vis)):
            if (e0 - s0) * DT < cfg["fill_short"]:
                vis[s0:e0] = True
    # short detector misses anywhere (video profile)
    for _ in range(rng.poisson(cfg.get("flicker_rate", 0.0) * n * DT / 60.0)):
        c = int(rng.integers(0, n))
        k = max(1, int(round(_lognorm(rng, cfg["flicker_median"], 1.0, 0.08, 10.0) * FPS)))
        vis[c:c + k] = False
    return vis


def ball_measurement(b_true, vis, rng, cfg, players):
    """What the tracker reports on visible frames: smoothed, noisy, sometimes far off, sometimes false."""
    n = len(b_true)
    b = b_true.copy()
    w = int(round(cfg["ball_smooth"] * FPS / 12.5)) | 1   # window given in 12.5 fps frames
    if w > 1:
        ker = np.ones(w, np.float32) / w
        b = np.stack([np.convolve(np.pad(b[:, c], (w // 2, w - 1 - w // 2), mode="edge"), ker, mode="valid") for c in range(2)], 1)
    b = b + _ou(rng, n, cfg["ball_sigma"], 1.0)
    if cfg.get("ball_white", 0) > 0:
        b = b + rng.standard_normal((n, 2)).astype(np.float32) * cfg["ball_white"]
    if cfg.get("p_air", 0) > 0:
        # a fast ball is usually in the air; a ground-plane projection then lands metres beyond it,
        # always on the side away from the camera (one side per half in this rotated view)
        sp = np.zeros(n, np.float32)
        sp[1:-1] = np.hypot(*(b_true[2:] - b_true[:-2]).T) / (2 * DT)
        fs, fe = _runs(sp > 12.0)
        side = np.array([rng.uniform(-0.3, 0.3), rng.choice([-1.0, 1.0])], np.float32)
        side /= np.hypot(*side)
        for s0, e0 in zip(fs, fe):
            if (e0 - s0) * DT < 0.4 or rng.random() > cfg["p_air"]:
                continue
            k = e0 - s0
            bump = np.sin(np.linspace(0, np.pi, k, dtype=np.float32)) ** 2
            b[s0:e0] += bump[:, None] * rng.uniform(2.0, 14.0) * side
    if cfg.get("outlier_rate", 0) > 0:
        for _ in range(rng.poisson(cfg["outlier_rate"] * n * DT / 60.0)):
            k = int(rng.integers(1, 4))
            c = int(rng.integers(0, max(1, n - k)))
            b[c:c + k] += rng.uniform(-8, 8, 2).astype(np.float32)
    # bursts of large error (ball in the air seen as a ground point)
    n_min = n * DT / 60.0
    for _ in range(rng.poisson(cfg["burst_rate"] * n_min)):
        k = int(round(rng.uniform(0.5, 3.0) * FPS))
        c = int(rng.integers(0, max(1, n - k)))
        bump = np.sin(np.linspace(0, np.pi, k, dtype=np.float32)) ** 2
        ang = rng.uniform(0, 2 * np.pi)
        mag = rng.uniform(2.0, 9.0)
        b[c:c + k] += bump[:, None] * mag * np.array([np.cos(ang), np.sin(ang)], np.float32)
    out = np.where(vis[:, None], b, np.nan).astype(np.float32)
    # false ball tracks inside gaps: a static point or a point riding on a player
    s_arr, e_arr = _runs(~vis)
    for s, e in zip(s_arr, e_arr):
        if e - s < 2 * FPS or rng.random() > cfg["p_false"]:
            continue
        k = int(round(rng.uniform(1.0, 10.0) * FPS))
        k = min(k, e - s)
        c = int(rng.integers(s, e - k + 1))
        if rng.random() < 0.5:
            pt = np.array([rng.uniform(-L_HALF, L_HALF), rng.uniform(-W_HALF - 3, W_HALF + 3)], np.float32)
            out[c:c + k] = pt[None] + _ou(rng, k, 0.2, 1.0)
        else:
            j = int(rng.integers(0, players.shape[1]))
            seg = players[c:c + k, j]
            ok = ~np.isnan(seg[:, 0])
            seg = seg + _ou(rng, k, 0.4, 1.0)
            out[c:c + k][ok] = seg[ok]
    return out


def degrade_players(home, away, b_true, rng, cfg):
    """Camera window, held keepers, dropouts, duplicates, jitter, rare team swaps."""
    n = len(b_true)
    teams = [home.copy(), away.copy()]
    teams_clean = [home, away]
    # camera: centre follows the ball with a lag; half-width varies slowly
    rho = np.exp(-DT / cfg["cam_tau"])
    cx = lfilter([1 - rho], [1.0, -rho], b_true[:, 0], zi=[b_true[0, 0] * rho])[0].astype(np.float32)
    cx = cx + _ou(rng, n, 2.0, 5.0, dims=1)[:, 0]
    hw = cfg["cam_halfwidth"] * (1.0 + _ou(rng, n, 0.08, 10.0, dims=1)[:, 0])
    cx = np.clip(cx, -L_HALF + hw * 0.7, L_HALF - hw * 0.7)   # the operator does not film the stands
    for ti, p in enumerate(teams):
        pres = ~np.isnan(p[..., 0])
        inview = np.abs(p[..., 0] - cx[:, None]) <= hw[:, None]
        # the deepest player of the team (its keeper) is held when off screen
        mx = np.nanmedian(np.where(pres, p[..., 0], np.nan), axis=0)
        mx = np.where(np.isnan(mx), 0.0, mx)
        gk = int(np.argmin(mx)) if ti == 0 else int(np.argmax(mx))
        out = pres & ~inview
        if rng.random() < cfg["p_hold"]:
            col = p[:, gk].copy()
            hid = out[:, gk]
            # hold the last seen position through each off-screen run
            idx = np.where(~hid & pres[:, gk], np.arange(n), -1)
            np.maximum.accumulate(idx, out=idx)
            held = np.where(idx[:, None] >= 0, col[np.maximum(idx, 0)], np.nan)
            col[hid] = held[hid]
            p[:, gk] = col
            out[:, gk] = False
        p[out] = np.nan
        # random track breaks
        n_min = n * DT / 60.0
        for j in range(p.shape[1]):
            if not pres[:, j].any():
                continue
            for _ in range(rng.poisson(cfg["drop_rate"] * n_min)):
                k = int(round(_lognorm(rng, 1.5, 1.0, 0.2, 30.0) * FPS))
                c = int(rng.integers(0, n))
                p[c:c + k, j] = np.nan
        # jitter
        for j in range(p.shape[1]):
            if pres[:, j].any():
                p[:, j] += _ou(rng, n, cfg["player_sigma"], 0.5)
        if cfg.get("p_occl", 0) > 0:
            # players close to another player are occluded more often, and tracked less precisely
            allp = np.concatenate([home, away], 1)
            for j in range(p.shape[1]):
                if not pres[:, j].any():
                    continue
                d = np.hypot(allp[..., 0] - teams_clean[ti][:, j, None, 0], allp[..., 1] - teams_clean[ti][:, j, None, 1])
                d = np.where(np.isnan(d) | (d < 1e-6), 99.0, d)
                close = d.min(1) < 1.0
                for s0, e0 in zip(*_runs(close)):
                    u = rng.random()
                    if u < cfg["p_occl"]:
                        p[s0:e0 + int(rng.integers(0, int(FPS) + 1)), j] = np.nan
                    elif u < 2 * cfg["p_occl"]:
                        p[s0:e0, j] += _ou(rng, e0 - s0, 3 * cfg["player_sigma"], 0.5)
        teams[ti] = p
    # duplicates, phantoms (referee, staff), and team swaps go into spare slots
    for ti in range(2):
        p = teams[ti]
        spare = [j for j in range(p.shape[1]) if np.isnan(p[:, j, 0]).all()]
        used = [j for j in range(p.shape[1]) if j not in spare]
        n_min = n * DT / 60.0
        for _ in range(rng.poisson(cfg["dup_rate"] * n_min)):
            if not spare or not used:
                break
            k = int(round(_lognorm(rng, 4.0, 1.0, 0.5, 60.0) * FPS))
            c = int(rng.integers(0, n))
            src_team = teams[1 - ti] if rng.random() < cfg["p_teamswap"] else p
            src_used = [j for j in range(src_team.shape[1]) if not np.isnan(src_team[:, j, 0]).all()]
            src = src_team[c:c + k, int(rng.choice(src_used))]
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(3.0, 15.0) if rng.random() < cfg["p_phantom"] else rng.uniform(0.3, 0.9)
            off = r * np.array([np.cos(ang), np.sin(ang)], np.float32)
            j = int(rng.choice(spare))
            seg = src + off + _ou(rng, len(src), 0.15 if r < 1 else 1.0, 0.5 if r < 1 else 3.0)
            free = np.isnan(p[c:c + k, j, 0])
            p[c:c + k, j][free] = seg[free]
    return teams[0], teams[1]


def sample_cfg(rng, strength=1.0, profile="provided", version=1):
    """One random tracker. strength 0 = mild, 1 = full range.

    profile "provided": like the target's provided ball (smoothed, lost for long stretches around
    stoppages, about 41% missing). profile "video": like our video ball detector (unsmoothed,
    jittery, wrong by metres on airborne balls, visible through most stoppages, short gaps, about
    17% missing). Player degradation is the same in both.
    """
    u = rng.uniform
    cfg = dict(
        p_vis_all=u(0.05, 0.6), p_vis_any=u(0.0, 0.3), p_late=u(0.1, 0.6), p_early=u(0.05, 0.4), p_island=u(0.1, 0.6),
        live_gap_rate=u(0.0, 2.0) * strength, live_gap_median=u(1.5, 6.0),
        ball_smooth=int(rng.choice([1, 3, 5, 9, 13])), ball_sigma=u(0.05, 0.8), burst_rate=u(0.0, 2.5) * strength,
        ball_white=0.0, p_air=0.0, outlier_rate=0.0,
        p_false=u(0.0, 0.25) * strength,
        cam_tau=u(0.5, 2.5), cam_halfwidth=u(25.0, 42.0), p_hold=u(0.3, 0.95),
        drop_rate=u(0.02, 0.3) * strength, player_sigma=u(0.03, 0.35), dup_rate=u(0.3, 4.0) * strength, p_teamswap=u(0.0, 0.15), p_phantom=u(0.2, 0.7),
    )
    if version >= 2:
        # tightened on further input statistics of the target (check_degrade2.py): both keepers almost
        # always present, fewer close pairs, ball rarely tracked off the pitch or in crowds, looser gap timing
        cfg.update(cam_halfwidth=u(25.0, 38.0), p_hold=u(0.85, 1.0), p_phantom=u(0.6, 1.0), dup_rate=u(1.5, 5.5) * strength,
                   p_out_drop=u(0.5, 0.95), out_margin=u(0.5, 3.0), crowd_drop=u(0.0, 0.3), p_occl=u(0.0, 0.2), gap_loose=True,
                   rest_scale=u(0.0, 1.0) ** 2, fill_short=u(0.5, 2.0))
    if profile == "video":
        cfg.update(p_vis_all=u(0.5, 1.0), p_vis_any=u(0.45, 0.95), p_late=u(0.1, 0.5), p_early=u(0.0, 0.3),
                   live_gap_rate=u(0.0, 2.0), live_gap_median=u(0.6, 2.5), flicker_rate=u(1.0, 6.0), flicker_median=u(0.3, 1.5),
                   ball_smooth=1, ball_sigma=u(0.0, 0.3), ball_white=u(0.01, 0.15), burst_rate=u(0.0, 1.0) * strength,
                   p_air=u(0.3, 1.0), outlier_rate=u(0.0, 4.0) * strength, p_false=u(0.0, 0.15) * strength)
        if version >= 2:
            cfg.update(crowd_drop=u(0.3, 0.9), fill_short=0.0)
    return cfg


def degrade(m, rng, cfg=None, resim_ball=True, do_players=True, do_ball_drop=True):
    """Clean canonical match -> degraded copy (same keys). Labels are untouched."""
    cfg = cfg or sample_cfg(rng)
    lab = m["label"]
    b_true = simulate_dead_ball(m["ball"], lab, rng, cfg.get("rest_scale", 1.0)) if resim_ball else _fill_nan_linear(m["ball"])
    home, away = (degrade_players(m["home"], m["away"], b_true, rng, cfg) if do_players else (m["home"].copy(), m["away"].copy()))
    vis = ball_visibility(lab, rng, cfg, b_true, np.concatenate([m["home"], m["away"]], 1)) if do_ball_drop else np.ones(len(lab), bool)
    if not resim_ball:
        vis &= ~np.isnan(m["ball"][:, 0])
    ball = ball_measurement(b_true, vis, rng, cfg, np.concatenate([home, away], 1))
    out = dict(m)
    out.update(home=home, away=away, ball=ball, cfg=cfg)
    return out


def gap_aux(ball):
    """Per frame: seconds since the ball was last seen, seconds until it is next seen (capped at 60),
    and the last-seen and next-seen ball positions. Zero offsets when the ball is present."""
    n = len(ball)
    ok = ~np.isnan(ball[:, 0])
    idx = np.arange(n)
    last = np.where(ok, idx, -1)
    np.maximum.accumulate(last, out=last)
    nxt = np.where(ok, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    dt_last = np.where(last >= 0, (idx - last) * DT, 60.0)
    dt_next = np.where(nxt < n, (nxt - idx) * DT, 60.0)
    p_last = np.where((last >= 0)[:, None], ball[np.maximum(last, 0)], 0.0)
    p_next = np.where((nxt < n)[:, None], ball[np.minimum(nxt, n - 1)], 0.0)
    aux = np.concatenate([np.minimum(dt_last, 60.0)[:, None], np.minimum(dt_next, 60.0)[:, None], p_last, p_next], 1)
    return aux.astype(np.float32)


def time_warp(m, rate):
    """Play the match `rate` times faster (rate > 1) or slower by resampling. Positions are linearly
    interpolated where both neighbours are valid; labels and halves take the nearest frame."""
    n = len(m["label"])
    t = np.arange(0, n - 1, rate)
    i0 = np.floor(t).astype(int); i1 = np.minimum(i0 + 1, n - 1)
    w = (t - i0).astype(np.float32)
    near = np.minimum(np.rint(t).astype(int), n - 1)
    out = dict(m)
    for k in ("home", "away", "ball"):
        v = m[k]
        ww = w.reshape((-1,) + (1,) * (v.ndim - 1))
        out[k] = v[i0] * (1 - ww) + v[i1] * ww
    out["label"] = m["label"][near]; out["half"] = m["half"][near]
    return out


def spell_weights(label, alpha, ref_s=4.0, lo=0.5, hi=6.0):
    """Per-frame loss weight (ref / spell length)^alpha for H and A spells, 1 on D. With alpha = 1
    every possession carries the same total weight, as in spellF1; alpha = 0 turns it off."""
    w = np.ones(len(label), np.float32)
    if alpha <= 0:
        return w
    e = np.flatnonzero(np.diff(label) != 0) + 1
    st = np.concatenate(([0], e)); en = np.concatenate((e, [len(label)]))
    for a, b in zip(st, en):
        if label[a] < D:
            w[a:b] = np.clip((ref_s / ((b - a) * DT)) ** alpha, lo, hi)
    return w
