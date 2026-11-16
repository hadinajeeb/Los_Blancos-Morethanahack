"""Identity-free possession model: per-frame set encoder over players + ball, then TCN + BiGRU.

Input tensors are in the canonical view (metres, home attacks +x, 12.5 fps):
    home, away: (B, T, S, 2) with NaN for missing players
    ball:       (B, T, 2) with NaN for a missing ball
    aux:        (B, T, 6) from degrade.gap_aux (time since / until ball seen, last / next ball position)
Output: logits (B, T, 3) for H, A, D.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DT, FPS

N_TOKEN = 16
N_GLOBAL = 30


def _velocity(pz, pres):
    """Central difference, zero where a neighbour is missing. pz: (B, T, ..., 2) NaN-free; pres: (B, T, ...)."""
    v = torch.zeros_like(pz)
    ok = torch.zeros_like(pres)
    v[:, 1:-1] = (pz[:, 2:] - pz[:, :-2]) / (2 * DT)
    ok[:, 1:-1] = pres[:, 2:] & pres[:, :-2] & pres[:, 1:-1]
    v = v * ok.unsqueeze(-1)
    return v, ok


def featurize(home, away, ball, aux, use_ball=True, kink=False):
    B, T, S, _ = home.shape
    P = torch.cat([home, away], 2)                      # (B, T, 2S, 2)
    pres = ~torch.isnan(P[..., 0])
    Pz = torch.nan_to_num(P, nan=0.0)
    v, vok = _velocity(Pz, pres)
    v = v.clamp(-12, 12)
    speed = v.norm(dim=-1)
    team = torch.cat([torch.ones(S, device=P.device), -torch.ones(S, device=P.device)]).view(1, 1, 2 * S).expand(B, T, -1)

    if not use_ball:
        ball = torch.full_like(ball, float("nan"))
        aux = torch.zeros_like(aux)
        aux[..., 0:2] = 60.0
    bp = ~torch.isnan(ball[..., 0])                     # (B, T)
    bz = torch.nan_to_num(ball, nan=0.0)
    bv, bvok = _velocity(bz, bp)
    bv = bv.clamp(-40, 40)
    bacc = torch.zeros_like(bv)
    bacc[:, 1:-1] = (bv[:, 2:] - bv[:, :-2]) / (2 * DT)
    aok = torch.zeros_like(bvok)
    aok[:, 1:-1] = bvok[:, 2:] & bvok[:, :-2]
    bacc = (bacc * aok.unsqueeze(-1)).clamp(-150, 150)
    bpf = bp.float()

    rel = (Pz - bz.unsqueeze(2)) * bpf[..., None, None]
    dist = rel.norm(dim=-1)
    relv = (v - bv.unsqueeze(2)) * (vok & bvok.unsqueeze(2)).unsqueeze(-1)
    dt_last, dt_next = aux[..., 0], aux[..., 1]
    p_last, p_next = aux[..., 2:4], aux[..., 4:6]
    w_last = torch.exp(-dt_last / 4.0) * (dt_last < 59).float()
    w_next = torch.exp(-dt_next / 4.0) * (dt_next < 59).float()
    d_last = (Pz - p_last.unsqueeze(2)).norm(dim=-1)
    d_next = (Pz - p_next.unsqueeze(2)).norm(dim=-1)
    bpe = bpf.unsqueeze(-1).expand(-1, -1, 2 * S)
    tok = torch.stack([
        team, Pz[..., 0] / 52.5, Pz[..., 1] / 34.0, v[..., 0] / 8, v[..., 1] / 8, speed / 8, vok.float(),
        torch.asinh(rel[..., 0] / 3), torch.asinh(rel[..., 1] / 3), torch.asinh(dist / 3) * bpe, torch.exp(-dist / 2.0) * bpe,
        relv[..., 0] / 15, relv[..., 1] / 15,
        torch.exp(-d_last / 3.0) * w_last.unsqueeze(-1), torch.exp(-d_next / 3.0) * w_next.unsqueeze(-1), bpe,
    ], -1)                                              # (B, T, 2S, N_TOKEN)
    tok = tok * pres.unsqueeze(-1)

    big = 1e3
    dh = torch.where(pres[..., :S] & bp.unsqueeze(-1), dist[..., :S], torch.full_like(dist[..., :S], big))
    da = torch.where(pres[..., S:] & bp.unsqueeze(-1), dist[..., S:], torch.full_like(dist[..., S:], big))
    dh2 = dh.topk(2, dim=-1, largest=False).values
    da2 = da.topk(2, dim=-1, largest=False).values
    near = torch.cat([dh2, da2], -1)                    # (B, T, 4)
    near_ok = (near < big).float()
    near = near.clamp(max=80)
    nh = pres[..., :S].sum(-1).float()
    na = pres[..., S:].sum(-1).float()

    def tmean(x, m):
        return (x * m).sum(-1) / m.sum(-1).clamp(min=1)

    mh, ma = pres[..., :S].float(), pres[..., S:].float()
    vokh, voka = vok[..., :S].float(), vok[..., S:].float()
    cxh, cxa = tmean(Pz[..., :S, 0], mh), tmean(Pz[..., S:, 0], ma)
    cyh, cya = tmean(Pz[..., :S, 1], mh), tmean(Pz[..., S:, 1], ma)
    sxh = torch.sqrt(tmean((Pz[..., :S, 0] - cxh.unsqueeze(-1)) ** 2, mh) + 1e-6)
    sxa = torch.sqrt(tmean((Pz[..., S:, 0] - cxa.unsqueeze(-1)) ** 2, ma) + 1e-6)
    g = torch.stack([
        bpf, bz[..., 0] / 52.5, bz[..., 1] / 34.0, bv[..., 0] / 15, bv[..., 1] / 15, bv.norm(dim=-1) / 15, bvok.float(),
        bacc.norm(dim=-1) / 50, F.relu(bz[..., 0].abs() - 52.5) / 5 * bpf, F.relu(bz[..., 1].abs() - 34.0) / 5 * bpf,
        torch.asinh(near[..., 0] / 3) * near_ok[..., 0], torch.asinh(near[..., 1] / 3) * near_ok[..., 1],
        torch.asinh(near[..., 2] / 3) * near_ok[..., 2], torch.asinh(near[..., 3] / 3) * near_ok[..., 3],
        torch.tanh((near[..., 2] - near[..., 0]) / 3) * near_ok[..., 0] * near_ok[..., 2],
        nh / 11, na / 11, tmean(speed[..., :S], vokh) / 5, tmean(speed[..., S:], voka) / 5,
        cxh / 52.5, cxa / 52.5, cyh / 34, cya / 34, sxh / 20, sxa / 20,
        torch.log1p(dt_last) / 4, torch.log1p(dt_next) / 4,
        p_last[..., 0] / 52.5 * (1 - bpf), p_last[..., 1] / 34 * (1 - bpf), (p_next - p_last).norm(dim=-1) / 30 * (1 - bpf),
    ], -1)                                              # (B, T, N_GLOBAL)
    if kink:
        # ball velocity change over 0.64 s (a touch that survives light smoothing), and over 0.16-0.32 s
        feats = []
        for span_s in (0.32, 0.16):
            k = max(1, int(round(span_s * FPS)))
            dv = torch.zeros_like(bv)
            ok = torch.zeros_like(bvok)
            dv[:, k:-k] = bv[:, 2 * k:] - bv[:, :-2 * k]
            ok[:, k:-k] = bvok[:, 2 * k:] & bvok[:, :-2 * k]
            feats.append(dv.norm(dim=-1) / 15 * ok.float())
        g = torch.cat([g, torch.stack(feats, -1)], -1)
    return tok, pres, g, dh, da


class Block(nn.Module):
    def __init__(self, ch, dil, drop):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 5, padding=2 * dil, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, 1)
        self.n = nn.GroupNorm(8, ch)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.c2(F.gelu(self.n(self.c1(x)))))


class PossNet(nn.Module):
    def __init__(self, d_tok=96, ch=192, gru=128, drop=0.15, dilations=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64), use_ball=True, bnd=False, kink=False):
        super().__init__()
        self.use_ball = use_ball
        self.kink = kink
        self.tok = nn.Sequential(nn.Linear(N_TOKEN, d_tok), nn.GELU(), nn.Linear(d_tok, d_tok), nn.GELU())
        self.inp = nn.Linear(6 * d_tok + N_GLOBAL + (2 if kink else 0), ch)
        self.dom = nn.Embedding(3, ch)   # label-convention token: 0 DFL, 1 Metrica, 2 SkillCorner
        nn.init.zeros_(self.dom.weight)
        self.blocks = nn.ModuleList([Block(ch, d, drop) for d in dilations])
        self.gru = nn.GRU(ch, gru, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(ch + 2 * gru, ch), nn.GELU(), nn.Dropout(drop), nn.Linear(ch, 3))
        # boundary head: [direct H<->A change here, any label change here]
        self.bhead = nn.Sequential(nn.Linear(ch + 2 * gru, ch), nn.GELU(), nn.Linear(ch, 2)) if bnd else None

    def forward(self, home, away, ball, aux, dom=None, with_bnd=False):
        tok, pres, g, dh, da = featurize(home, away, ball, aux, self.use_ball, self.kink)
        B, T, S2, _ = tok.shape
        S = S2 // 2
        e = self.tok(tok)                                # (B, T, 2S, d)
        neg = torch.finfo(e.dtype).min

        def pool(ee, m):
            mx = ee.masked_fill(~m.unsqueeze(-1), neg).max(dim=2).values
            mx = torch.where(m.any(-1, keepdim=True), mx, torch.zeros_like(mx))
            mean = (ee * m.unsqueeze(-1)).sum(2) / m.sum(-1, keepdim=True).clamp(min=1)
            return mx, mean

        hx, hm = pool(e[:, :, :S], pres[:, :, :S])
        ax, am = pool(e[:, :, S:], pres[:, :, S:])
        # embedding of the player nearest the ball, per team (zero when no ball or no player)
        ih = dh.argmin(-1)
        ia = da.argmin(-1)
        okh = (dh.gather(-1, ih.unsqueeze(-1)) < 1e3).to(e.dtype)
        oka = (da.gather(-1, ia.unsqueeze(-1)) < 1e3).to(e.dtype)
        d = e.shape[-1]
        nh = e[:, :, :S].gather(2, ih.view(B, T, 1, 1).expand(-1, -1, 1, d)).squeeze(2) * okh
        na = e[:, :, S:].gather(2, ia.view(B, T, 1, 1).expand(-1, -1, 1, d)).squeeze(2) * oka
        x = self.inp(torch.cat([hx, hm, ax, am, nh, na, g.to(e.dtype)], -1))
        if dom is not None:
            x = x + self.dom(dom).unsqueeze(1)
        y = x.transpose(1, 2)
        for b in self.blocks:
            y = b(y)
        y = y.transpose(1, 2)
        r, _ = self.gru(y)
        z = torch.cat([y, r], -1)
        if with_bnd:
            return self.head(z), (self.bhead(z) if self.bhead is not None else None)
        return self.head(z)


def swap_teams(home, away, ball, aux):
    """Exchange the teams and rotate the pitch by 180 degrees, so the new home still attacks +x."""
    aux2 = aux.clone()
    aux2[..., 2:6] = -aux[..., 2:6]
    return -away, -home, -ball, aux2


def flip_y(home, away, ball, aux):
    def fy(a):
        a = a.clone()
        a[..., 1] = -a[..., 1]
        return a
    aux2 = aux.clone()
    aux2[..., 3] = -aux[..., 3]
    aux2[..., 5] = -aux[..., 5]
    return fy(home), fy(away), fy(ball), aux2


def set_deterministic():
    """Inference settings for a reproducible file: no TF32, deterministic cuDNN, no autotuning."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(ck, dev="cuda"):
    """Rebuild a PossNet from a checkpoint written by train.py."""
    sd = torch.load(ck, map_location=dev, weights_only=True)
    a = sd["args"]
    m = PossNet(use_ball=not a.get("no_ball", False), drop=a.get("drop", 0.15), ch=a.get("ch", 192),
                dilations=tuple(a.get("dilations", (1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64))),
                bnd=a.get("bnd", 0) > 0, kink=a.get("kink", False)).to(dev)
    m.load_state_dict(sd["model"], strict=False)
    m.eval()
    return m, a


def _chunks(T, ch, ov):
    """(start, end, keep_start, keep_end) windows covering 0..T; each inner edge has `ov` frames of context."""
    if T <= ch:
        return [(0, T, 0, T)]
    out, s = [], 0
    while True:
        e = min(s + ch, T)
        out.append((s, e, s if s == 0 else s + ov, e if e == T else e - ov))
        if e == T:
            return out
        s = e - 2 * ov


@torch.no_grad()
def predict_full(model, home, away, ball, aux, tta=True, dom=0, chunk_s=480.0, overlap_s=60.0):
    """Posterior (T, 3) and boundary probabilities (T, 2) or None, for one long sequence.

    The sequence is processed in windows of chunk_s seconds whose inner edges carry overlap_s
    seconds of context on each side (the convolution stack sees about 40 s). Results are averaged
    over team swap and y flip when tta is on. dom selects the label convention (0 = DFL).
    """
    model.eval()
    T = home.shape[0]
    post = torch.zeros(T, 3, device=home.device)
    bnd = torch.zeros(T, 2, device=home.device) if model.bhead is not None else None
    d = torch.full((1,), dom, dtype=torch.long, device=home.device)
    for s, e, ks, ke in _chunks(T, int(chunk_s * FPS), int(overlap_s * FPS)):
        args = [t[s:e].unsqueeze(0) for t in (home, away, ball, aux)]
        ps, bs = [], []
        for sw in ((False, True) if tta else (False,)):
            for fl in ((False, True) if tta else (False,)):
                a = args
                if sw:
                    a = swap_teams(*a)
                if fl:
                    a = flip_y(*a)
                lg, bl = model(*a, dom=d, with_bnd=True)
                p = lg.float().softmax(-1)[0]
                ps.append(p[:, [1, 0, 2]] if sw else p)
                if bl is not None:
                    bs.append(torch.sigmoid(bl.float())[0])
        post[ks:ke] = torch.stack(ps).mean(0)[ks - s:ke - s]
        if bnd is not None:
            bnd[ks:ke] = torch.stack(bs).mean(0)[ks - s:ke - s]
    return post, bnd


def predict(model, home, away, ball, aux, tta=True, dom=0):
    return predict_full(model, home, away, ball, aux, tta=tta, dom=dom)[0]
