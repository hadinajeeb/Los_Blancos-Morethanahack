"""Stage 2: run the ball heatmap network on the tactical video.

Each process decodes its chunks of the video in order, stacks frames (t-2, t, t+2), and stores the
top-K heatmap peaks of every frame: x_px, y_px, confidence. Several processes can share one GPU.

    python -m mtah.ball_infer --video V --ckpt W --out DIR --shard i --nshards n [--device cuda|cpu] [--amp]

Reference mode is float32 with TF32 off and deterministic cuDNN. `--amp` (bfloat16 autocast) is
about twice as fast on GPU; it is how the first candidate file was made and it is not portable
across hardware.
"""
import argparse
import os
from collections import deque

import av
import numpy as np
import torch
import torch.nn.functional as F

from .ball_model import BallNet, peaks
from .prep import FPS, HALVES, N_FRAMES

W, H = 1920, 1080
DT, CHUNK, MARGIN = 2, 2000, 50


class Reader:
    """Sequential frame reader with accurate seek. Yields (frame_index, rgb uint8 HxWx3)."""

    def __init__(self, path, threads=2):
        self.c = av.open(path)
        self.s = self.c.streams.video[0]
        self.s.thread_type = "AUTO"
        self.s.codec_context.thread_count = threads
        self.tb = float(self.s.time_base)
        self.start = self.s.start_time or 0

    def idx(self, frame):
        return int(round((frame.pts - self.start) * self.tb * FPS))

    def read(self, start, stop, fmt="rgb24"):
        ts = int(max(start - 1, 0) / FPS / self.tb) + self.start
        self.c.seek(ts, stream=self.s, backward=True, any_frame=False)
        for fr in self.c.decode(self.s):
            i = self.idx(fr)
            if i < start:
                continue
            if i >= stop:
                break
            yield i, fr.to_ndarray(format=fmt)

    def close(self):
        self.c.close()


def chunks():
    """Frame chunks covering the graded halves plus a margin. The same list for every shard count."""
    rng = [(s - MARGIN, e + MARGIN + 1) for s, e in HALVES]
    return [(c, min(c + CHUNK, e)) for s, e in rng for c in range(s, e, CHUNK)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--amp", action="store_true")
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--only", default="", help="comma-separated chunk start frames (for tests)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if a.threads:
        torch.set_num_threads(a.threads)
    todo = chunks()
    if a.only:
        keep = {int(v) for v in a.only.split(",")}
        todo = [c for c in todo if c[0] in keep]
    mine = todo[a.shard::a.nshards]
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    net = BallNet().to(dev).to(memory_format=torch.channels_last).eval()
    net.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True))
    pad_h = (16 - H % 16) % 16

    @torch.no_grad()
    def run(batch):
        x = torch.from_numpy(np.stack(batch)).to(dev)            # B,3,H,W,3 uint8
        B = x.shape[0]
        x = x.permute(0, 1, 4, 2, 3).reshape(B, 9, H, W).float() / 255.0
        x = F.pad((x - 0.45) / 0.25, (0, 0, 0, pad_h)).contiguous(memory_format=torch.channels_last)
        if a.amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
        else:
            out = net(x)
        out = out[:, : H // 4]
        return peaks(out, k=a.k, thr=0.05).cpu().numpy()

    for (s, e) in mine:
        path = f"{a.out}/pk_{s:06d}.npy"
        if os.path.exists(path):
            continue
        res = np.zeros((e - s, a.k, 3), np.float32)
        r = Reader(a.video, threads=3)
        buf = deque(maxlen=2 * DT + 1); batch, ids = [], []
        for i, img in r.read(max(s - DT, 0), min(e + DT, N_FRAMES)):
            buf.append((i, img))
            c = i - DT
            if s <= c < e and len(buf) == 2 * DT + 1 and buf[0][0] == c - DT:
                batch.append(np.stack([buf[0][1], buf[DT][1], buf[2 * DT][1]])); ids.append(c)
                if len(batch) == a.bs:
                    res[np.array(ids) - s] = run(batch); batch, ids = [], []
        if batch:
            res[np.array(ids) - s] = run(batch)
        r.close()
        np.save(path + ".tmp.npy", res); os.replace(path + ".tmp.npy", path)
    print("shard", a.shard, "done", len(mine), "chunks", flush=True)


if __name__ == "__main__":
    main()
