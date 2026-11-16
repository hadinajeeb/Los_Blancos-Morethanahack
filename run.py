"""Regenerate the submission from the provided data bundle.

    python run.py --data <MTAH_DATA_DIR> --out <team_name>.csv

Stages (each cached in --work; delete the directory to recompute):
    1 prep        ATD text + XML + homography CSV -> arrays
    2 ball_infer  ball heatmap network on the tactical video -> peaks per frame   (ball sources "video", "avg")
    3 ball_link   peaks -> one ball track                                          (ball sources "video", "avg")
    4 seq         sequence-model ensemble -> P(H), P(A), P(D) per frame
    5 decode      Viterbi + dead-ball rules -> CSV

config.json selects the weights, the ball source, and the decode settings:
    "ball_source": "video"     our ball track from the video; the provided ball where it has none
                   "provided"  the provided ball only; no video and no ball detector needed
                   "avg"       mean of the two posteriors
--ball-source overrides it. --fallback is the same as --ball-source provided.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR"), help="data bundle root (holds tactical/)")
    ap.add_argument("--out", required=True, help="submission CSV to write")
    ap.add_argument("--work", default=os.path.join(HERE, "work"), help="cache directory for the stages")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--ball-source", choices=["video", "provided", "avg"], help="override config.json")
    ap.add_argument("--fallback", action="store_true", help="same as --ball-source provided")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--shards", type=int, default=0, help="ball_infer processes (default 8 on cuda, 16 on cpu)")
    ap.add_argument("--ball-amp", action="store_true", help="bfloat16 autocast in ball_infer (faster, not portable; overrides config)")
    ap.add_argument("--cpu-threads", type=int, default=8, help="torch threads per ball_infer process on cpu")
    a = ap.parse_args()
    assert a.data, "pass --data or set MTAH_DATA_DIR"
    cfg = json.load(open(a.config))
    source = "provided" if a.fallback else (a.ball_source or cfg["ball_source"])
    res = lambda p: p if os.path.isabs(p) else os.path.join(HERE, p)
    os.makedirs(a.work, exist_ok=True)
    hashes, times = {}, {}

    from mtah import prep
    t0 = time.time()
    prep.run(a.data, a.work)
    times["prep"] = time.time() - t0
    target = dict(np.load(os.path.join(a.work, "target.npz")))

    if source in ("video", "avg"):
        from mtah import ball_infer, ball_link
        t0 = time.time()
        pdir = os.path.join(a.work, "peaks")
        todo = [c for c in ball_infer.chunks() if not os.path.exists(f"{pdir}/pk_{c[0]:06d}.npy")]
        if todo:
            n = a.shards or (8 if a.device == "cuda" else 16)
            video = prep.paths(a.data)["video"]
            procs = []
            for i in range(n):
                cmd = [sys.executable, "-m", "mtah.ball_infer", "--video", video, "--ckpt", res(cfg["ball_ckpt"]), "--out", pdir,
                       "--shard", str(i), "--nshards", str(n), "--device", a.device]
                if cfg.get("ball_amp") or a.ball_amp:
                    cmd.append("--amp")
                if a.device == "cpu":
                    cmd += ["--threads", str(a.cpu_threads)]
                procs.append(subprocess.Popen(cmd, cwd=HERE))
            rc = [p.wait() for p in procs]
            assert not any(rc), f"ball_infer failed: {rc}"
        times["ball_infer"] = time.time() - t0
        t0 = time.time()
        ball_link.run(a.work)
        times["ball_link"] = time.time() - t0
        PK, _ = ball_link.load_peaks(pdir)
        bv = np.load(os.path.join(a.work, "ball_video.npz"))
        hashes["peaks"] = sha_arr(PK)
        hashes["ball_track"] = sha_arr(np.nan_to_num(bv[cfg["ball_key"]], nan=-9.0))

    from mtah import seq, decode
    t0 = time.time()
    ckpts = [res(c) for c in cfg["seq_ckpts"]]
    tag = "_" + cfg.get("name", "x")
    posts = {}
    for key in {"video": [cfg["ball_key"]], "provided": ["provided"], "avg": [cfg["ball_key"], "provided"]}[source]:
        p = seq.run(a.work, ckpts, key, cfg["seq_fps"], cfg["convention"], a.device, cfg.get("seq_chunk_s"), cfg.get("seq_overlap_s", 60.0), tag)
        posts[key] = np.load(p)["post"]
        hashes["posterior_" + ("provided" if key == "provided" else "video")] = sha_arr(posts[key])
    post = posts[cfg["ball_key"]] if source == "video" else posts["provided"] if source == "provided" else (posts[cfg["ball_key"]] + posts["provided"]) / 2
    times["seq"] = time.time() - t0

    t0 = time.time()
    lab, rinfo = decode.labels_from_posterior(cfg, post, target)
    decode.write_csv(lab, a.out)
    decode.check_csv(a.out)
    times["decode"] = time.time() - t0
    hashes["csv"] = sha_file(a.out)
    counts = {c: int(n) for c, n in zip("HADX", np.bincount(lab, minlength=4))}
    report = dict(out=a.out, ball_source=source, device=a.device, dead_rules=rinfo, label_counts=counts,
                  sha256=hashes, seconds={k: round(v, 1) for k, v in times.items()}, config=cfg)
    json.dump(report, open(os.path.join(a.work, f"report_{cfg.get('name', 'x')}_{source}.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
