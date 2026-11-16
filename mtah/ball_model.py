"""Small fully-convolutional heatmap net: 3 stacked RGB frames (t-2, t, t+2) -> ball heatmap at stride 4."""
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4


def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = cbr(c, c), nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(x + self.b(self.a(x)))


class BallNet(nn.Module):
    def __init__(self, cin=9, w=(32, 64, 128, 192)):
        super().__init__()
        self.s1 = nn.Sequential(cbr(cin, w[0], 2), cbr(w[0], w[0]))
        self.s2 = nn.Sequential(cbr(w[0], w[1], 2), Res(w[1]), Res(w[1]))
        self.s3 = nn.Sequential(cbr(w[1], w[2], 2), Res(w[2]), Res(w[2]))
        self.s4 = nn.Sequential(cbr(w[2], w[3], 2), Res(w[3]), Res(w[3]))
        self.u3 = nn.Sequential(cbr(w[3] + w[2], w[2]), Res(w[2]))
        self.u2 = nn.Sequential(cbr(w[2] + w[1], w[1]), Res(w[1]))
        self.head = nn.Sequential(cbr(w[1], w[1]), nn.Conv2d(w[1], 1, 1))
        self.head[-1].bias.data.fill_(-4.6)  # start at p ~ 0.01

    def forward(self, x):
        a = self.s1(x); b = self.s2(a); c = self.s3(b); d = self.s4(c)
        c = self.u3(torch.cat([F.interpolate(d, size=c.shape[-2:], mode="bilinear", align_corners=False), c], 1))
        b = self.u2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1))
        return self.head(b)[:, 0]  # logits (B, H/4, W/4)


def peaks(logits, k=3, thr=0.05):
    """Top-k local maxima per image. Returns (B,k,3): x_px, y_px, conf (conf 0 where none)."""
    p = torch.sigmoid(logits.float())
    m = (F.max_pool2d(p[:, None], 5, 1, 2)[:, 0] == p) & (p > thr)
    B, Hh, Ww = p.shape
    s = (p * m).flatten(1)
    conf, idx = s.topk(k, dim=1)
    iy, ix = idx // Ww, idx % Ww
    # sub-cell refinement: probability-weighted centroid in a 3x3 neighbourhood
    pp = F.pad(p, (1, 1, 1, 1))
    ox = torch.zeros_like(conf); oy = torch.zeros_like(conf); tot = torch.zeros_like(conf)
    bi = torch.arange(B, device=p.device)[:, None].expand(-1, k)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            v = pp[bi, iy + 1 + dy, ix + 1 + dx]
            ox += v * dx; oy += v * dy; tot += v
    x = (ix + ox / tot.clamp_min(1e-6) + 0.5) * STRIDE
    y = (iy + oy / tot.clamp_min(1e-6) + 0.5) * STRIDE
    return torch.stack([x, y, conf], -1)
