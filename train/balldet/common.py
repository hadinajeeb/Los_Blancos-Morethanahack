"""Shared helpers for the ball-detector workstream: paths, homography, tracking, video decoding."""
import os

import av
import numpy as np

ROOT = os.environ.get("MTAH_ROOT", ".")
VIDEO = f"{ROOT}/data/tactical/BAR-ATH_tactical.mp4"
HOMOG = f"{ROOT}/data/tactical/BAR-ATH_tactical_homography.csv"
ATD = f"{ROOT}/derived/tactical_atd.npz"
SCRATCH = os.environ.get("MTAH_SCRATCH", "./scratch")
RUNS = f"{ROOT}/runs/balldet"
N_FRAMES = 199319
FPS = 25
HALVES = [(11185, 90651), (110808, 190197)]  # graded video frames, inclusive
W, H = 1920, 1080


def graded_mask():
    g = np.zeros(N_FRAMES, bool)
    for a, b in HALVES:
        g[a:b + 1] = True
    return g


def load_homography():
    """(N,3,3) matrices mapping normalised pitch (0-1) -> pixels; cached as npy."""
    cache = f"{RUNS}/homography.npy"
    if os.path.exists(cache):
        return np.load(cache)
    t = np.genfromtxt(HOMOG, delimiter=",", skip_header=1)
    M = t[:, 1:10].reshape(-1, 3, 3)
    os.makedirs(RUNS, exist_ok=True)
    np.save(cache, M)
    return M


def project(M, x, y):
    """pitch (0-1) -> pixels. M (...,3,3), x,y broadcastable."""
    d = M[..., 2, 0] * x + M[..., 2, 1] * y + M[..., 2, 2]
    return ((M[..., 0, 0] * x + M[..., 0, 1] * y + M[..., 0, 2]) / d,
            (M[..., 1, 0] * x + M[..., 1, 1] * y + M[..., 1, 2]) / d)


def unproject(M, px, py):
    return project(np.linalg.inv(M), px, py)


def load_ball():
    """Provided ball track: (N,2) normalised pitch, NaN missing. Index = video frame."""
    return np.load(ATD)["ball"][:, :2].astype(np.float64)


def ball_pixels():
    """Provided ball projected to pixels, (N,2), NaN where missing."""
    B = load_ball()
    M = load_homography()
    px, py = project(M, B[:, 0], B[:, 1])
    return np.stack([px, py], 1)


class Reader:
    """Sequential frame reader with accurate seek. Yields (frame_index, rgb uint8 HxWx3)."""

    def __init__(self, path=VIDEO, threads=2):
        self.c = av.open(path)
        self.s = self.c.streams.video[0]
        self.s.thread_type = "AUTO"
        self.s.codec_context.thread_count = threads
        self.tb = float(self.s.time_base)
        self.start = self.s.start_time or 0

    def idx(self, frame):
        return int(round((frame.pts - self.start) * self.tb * FPS))

    def read(self, start, stop, fmt="rgb24"):
        """Yield frames start..stop-1."""
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
