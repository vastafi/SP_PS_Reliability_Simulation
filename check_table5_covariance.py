# -*- coding: utf-8 -*-
"""Block-level check of the independence assumption behind z in Table 5.
Blocks of 594 digits hold 2 inverse-CDF and 3 structural replications that share digits."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sp_ps_core import (FILES, C3, C5, K_TRUE, load_digits, randu_digits, sim_inverse, sim_structural,
                        mle_complete, mle_censored)
import sp_ps_core as core
R, n = 1000, 33
DPF = getattr(core, "DIGITS_PER_FILE", 2_614_690)
src = {g: load_digits(f[0]) for g, f in FILES.items()}
src["RANDU"] = randu_digits(DPF)
est = lambda X, net: {"complete": mle_complete(X, net=net), "C3": mle_censored(X, C3, net=net),
                      "C5": mle_censored(X, C5, net=net)}
res = json.load(open(HERE / "results" / "results_v2.json"))["E2_delta"]   # run run_experiments.py first
out = {}
for g, d in src.items():
    for net in ("SP", "PS"):
        Ei = est(sim_inverse(d, R, n, net=net), net); Es = est(sim_structural(d, R, n, net=net), net)
        for s in ("complete", "C3", "C5"):
            a = (Ei[s] - K_TRUE) ** 2.0; b = (Es[s] - K_TRUE) ** 2.0
            A = np.zeros(500); B = np.zeros(500)
            np.add.at(A, np.arange(R) // 2, a); np.add.at(B, np.arange(R) // 3, b)
            Dm = (A - B) / R                      # block contributions; ΔMSE = sum(Dm)
            g1 = slice(0, 333); g3 = slice(334, 500)
            var = 333 * Dm[g1].var(ddof=1) + 166 * Dm[g3].var(ddof=1) + Dm[g1].var(ddof=1)
            se_b = np.sqrt(var); dm = Dm.sum()
            r = np.corrcoef(A[g1], B[g1])[0, 1]
            k = f"{g}|{net}|{s}"
            out[k] = (res[k]["z"], dm / se_b, r)
zi = np.array([v[0] for v in out.values()]); zb = np.array([v[1] for v in out.values()])
rr = np.array([v[2] for v in out.values()])
for k, v in out.items():
    print(f"{k:40s} z_indep={v[0]:+.2f} z_block={v[1]:+.2f} corr={v[2]:+.3f}")
print("max|z_indep-z_block|=%.3f  corr mean=%.3f range=[%.3f,%.3f]" % (abs(zi - zb).max(), rr.mean(), rr.min(), rr.max()))
print("|z|>1.96 indep:", int((abs(zi) > 1.96).sum()), " block:", int((abs(zb) > 1.96).sum()))
