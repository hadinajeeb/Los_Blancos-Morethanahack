"""Stage 4: posterior P(H), P(A), P(D) for every graded frame from the sequence-model ensemble.

The model's view: metres, origin at the centre spot, the pitch rotated so that the home team always
attacks toward +x, at the checkpoints' frame rate (12.5 fps: the even and the odd 25 fps frames are
run separately, so every 25 fps frame gets its own posterior). Each half carries 60 s of ungraded
context on both sides. Test-time augmentation averages team swap and left-right flip.
"""
import os

import numpy as np
import torch

from . import seq_model as SM
from .prep import HALVES, N_FRAMES

S_MAX = 26  # slots per team after padding


def _pad_slots(a, s=S_MAX):
    n, k, _ = a.shape
    if k >= s:
        keep = np.argsort(-(~np.isnan(a[..., 0])).sum(0), kind="stable")[:s]
        return a[:, np.sort(keep)]
    out = np.full((n, s, 2), np.nan, np.float32)
    out[:, :k] = a
    return out


def canonical(home, away, ball, home_dir, pitch):
    """Normalised 0-1 arrays -> metres, centred, home attacking +x."""
    L, W = float(pitch[0]), float(pitch[1])
    s = np.where(home_dir >= 0, 1.0, -1.0).astype(np.float32)

    def conv(a, sgn):
        out = np.empty_like(a, dtype=np.float32)
        out[..., 0] = (a[..., 0] - 0.5) * L * sgn
        out[..., 1] = (a[..., 1] - 0.5) * W * sgn
        return out

    return conv(home, s[:, None]), conv(away, s[:, None]), conv(ball, s)


def gap_aux(ball, dt):
    """Per frame: seconds since the ball was last seen, seconds until it is next seen (capped at 60),
    and the last-seen and next-seen ball positions."""
    n = len(ball)
    ok = ~np.isnan(ball[:, 0])
    idx = np.arange(n)
    last = np.where(ok, idx, -1)
    np.maximum.accumulate(last, out=last)
    nxt = np.where(ok, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    dt_last = np.where(last >= 0, (idx - last) * dt, 60.0)
    dt_next = np.where(nxt < n, (nxt - idx) * dt, 60.0)
    p_last = np.where((last >= 0)[:, None], ball[np.maximum(last, 0)], 0.0)
    p_next = np.where((nxt < n)[:, None], ball[np.minimum(nxt, n - 1)], 0.0)
    aux = np.concatenate([np.minimum(dt_last, 60.0)[:, None], np.minimum(dt_next, 60.0)[:, None], p_last, p_next], 1)
    return aux.astype(np.float32)


def load_target(target, ball_all, phase, stride, margin_s=60.0):
    """One dict per half in the model's view. `frames` holds the video frame of every row.

    Goalkeeper slots (the last slot of each team): a keeper cannot stand in the opponent's
    defensive third in open play, so a keeper slot found there is dropped. A general rule.
    """
    pitch = target["pitch"]
    out = []
    mar = int(round(margin_s * 25))
    for a, b in HALVES:
        d = int(target["home_dir"][a])
        lo, hi = max(0, a - mar), min(N_FRAMES, b + 1 + mar)
        fr = np.arange(lo + ((phase - lo) % stride), hi, stride)
        hdir = np.full(len(fr), d, np.int8)
        home, away, ball_c = canonical(target["home"][fr], target["away"][fr], ball_all[fr], hdir, pitch)
        third = float(pitch[0]) / 6.0
        gk_h, gk_a = home.shape[1] - 1, away.shape[1] - 1
        bad_h = home[:, gk_h, 0] > third
        bad_a = away[:, gk_a, 0] < -third
        home[bad_h, gk_h] = np.nan
        away[bad_a, gk_a] = np.nan
        out.append(dict(home=_pad_slots(home), away=_pad_slots(away), ball=ball_c, frames=fr))
    return out


def posterior(ckpts, target, ball_all, fps=12.5, dom=0, device="cuda", tta=True, chunk_s=None, overlap_s=60.0):
    """chunk_s None: each half goes through the network in one pass (how the v1 checkpoints were applied).
    A number: windows of chunk_s seconds with overlap_s seconds of context at inner edges."""
    SM.configure(fps)
    SM.set_deterministic()
    torch.manual_seed(0)
    stride = int(round(25.0 / fps))
    models = []
    for ck in ckpts:
        m, margs = SM.load_model(ck, device)
        assert abs(margs.get("fps", 12.5) - fps) < 1e-6, f"{ck} was trained at {margs.get('fps')} fps, config says {fps}"
        models.append(m)
    post = np.zeros((N_FRAMES, 3), np.float32)
    have = np.zeros(N_FRAMES, bool)
    for phase in range(stride):
        for t in load_target(target, ball_all, phase, stride):
            args = [torch.from_numpy(x).to(device) for x in (t["home"], t["away"], t["ball"], gap_aux(t["ball"], 1.0 / fps))]
            kw = dict(chunk_s=1e9) if chunk_s is None else dict(chunk_s=float(chunk_s), overlap_s=float(overlap_s))
            ps = [SM.predict_full(m, *args, tta=tta, dom=dom, **kw)[0] for m in models]
            post[t["frames"]] = torch.stack(ps).mean(0).cpu().numpy()
            have[t["frames"]] = True
    return post, have


def run(work, ckpts, ball_key, fps, dom, device, chunk_s=None, overlap_s=60.0, tag=""):
    """ball_key: "provided", or a key of ball_video.npz. `tag` names the checkpoint set, so that
    posteriors of different ensembles do not share a cache file."""
    out = os.path.join(work, f"post_{'provided' if ball_key == 'provided' else 'video'}{tag}.npz")
    if os.path.exists(out):
        return out
    target = dict(np.load(os.path.join(work, "target.npz")))
    ball_all = target["ball"][:, :2] if ball_key == "provided" else np.load(os.path.join(work, "ball_video.npz"))[ball_key]
    post, have = posterior(ckpts, target, ball_all, fps=fps, dom=dom, device=device, chunk_s=chunk_s, overlap_s=overlap_s)
    np.savez_compressed(out + ".tmp.npz", post=post, have=have)
    os.replace(out + ".tmp.npz", out)
    return out
