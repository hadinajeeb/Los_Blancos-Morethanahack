"""Compare input statistics of degraded DFL matches with the target (inputs only, no labels of the target)."""
import sys
import numpy as np
import common as C
from degrade import degrade, sample_cfg, _runs

def stats(home, away, ball, tag):
    bp = ~np.isnan(ball[:, 0])
    s, e = _runs(~bp); gl = (e - s) * C.DT
    s2, e2 = _runs(bp); vl = (e2 - s2) * C.DT
    nh = (~np.isnan(home[..., 0])).sum(1); na = (~np.isnan(away[..., 0])).sum(1)
    mins = len(ball) * C.DT / 60
    print(f"{tag:28s} miss {1-bp.mean():.3f} gaps/100min {len(gl)/mins*100:5.0f} gap p10/50/90 {np.percentile(gl,[10,50,90]).round(1) if len(gl) else None}"
          f" vis p50 {np.median(vl):5.1f}  players h {nh.mean():.2f} a {na.mean():.2f}  P(11) {np.mean(nh==11):.2f} P(10) {np.mean(nh==10):.2f} P(<=8) {np.mean(nh<=8):.2f} P(>=12) {np.mean(nh>=12):.2f}")

for t in C.load_target():
    g = t["half"] > 0
    stats(t["home"][g], t["away"][g], t["ball"][g], t["name"] + f" gkdrop{t['gk_dropped']}")
m = C.load_external(C.list_matches("dfl")[0])
for seed in range(8):
    rng = np.random.default_rng(seed)
    d = degrade(m, rng, cfg=sample_cfg(rng, 1.0, sys.argv[1] if len(sys.argv) > 1 else "provided", int(sys.argv[2]) if len(sys.argv) > 2 else 1))
    stats(d["home"], d["away"], d["ball"], f"dfl {sys.argv[1:]} seed {seed}")
    lab = m["label"]; bp = ~np.isnan(d["ball"][:, 0])
    print("      P(miss|live) %.3f P(miss|D) %.3f  D share %.3f" % ((~bp)[lab < 2].mean(), (~bp)[lab == 2].mean(), (lab == 2).mean()))
