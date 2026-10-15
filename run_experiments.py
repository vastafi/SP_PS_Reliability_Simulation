"""
run_experiments.py  (revision v2)
Reproduces every number in the revised manuscript and writes results/*.json|csv.

E0  theoretical characteristics (Table 1, Section 2.1, Eqs. (15)-(16))
E1  verification of simulated means with the CLT volume (Figure 4, Section 3.2)
E2  main MLE study, R = 1000, n = 33 (Tables 3-5) + exact benchmark (PCG64)
E3  replication over the six independent files of each generator (Section 3.4, text)
E4  sensitivity to n (Figure 6), K and (N, M) (Table 6), and pooled Java check over six files
E5  controlled-defect (dose-response) generators (Figure 7)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sp_ps_core import (C3, C5, DIGITS_PER_FILE, FILES, K_TRUE, M_SUB, N_UNITS,
                        cdf, clt_volume, load_concat, load_digits, mle_censored,
                        mle_complete, pmf, randu_digits, sim_exact, sim_inverse,
                        sim_structural, summary, theory)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
R, n = 1000, 33
SEED = 2026
R_BENCH = 200_000


def est_all(X, net, N=N_UNITS, M=M_SUB):
    return {"complete": mle_complete(X, N, M, net),
            "C3": mle_censored(X, C3, N, M, net),
            "C5": mle_censored(X, C5, N, M, net)}


def to_py(o):
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


res = {}

# ---------------------------------------------------------------- E0
th = theory()
nSP, epsSP, z = clt_volume(th["SP"]["mean"], th["SP"]["sd"])
nPS, epsPS, _ = clt_volume(th["PS"]["mean"], th["PS"]["sd"])
res["E0"] = dict(theory=th, z=z, n_SP=nSP, n_PS=nPS, eps_SP=epsSP, eps_PS=epsPS)
print("E0", th["SP"]["mean"], th["PS"]["mean"], th["SP"]["var"], nSP, nPS)

# ---------------------------------------------------------------- E1
E1 = {}
for g, files in FILES.items():
    E1[g] = {}
    for net, nn in (("SP", nSP), ("PS", nPS)):
        d = load_concat(files, nn * N_UNITS * M_SUB)
        X = sim_structural(d, 1, nn, net=net)
        mu = th[net]["mean"]
        F_emp = np.array([(X <= k).mean() for k in range(K_TRUE + 1)])
        E1[g][net] = dict(n=nn, mean=float(X.mean()), dev=float(X.mean() - mu),
                          eps=0.001 * mu, F_emp=F_emp,
                          dF=F_emp - th[net]["F"],
                          khat={k: int(v[0]) for k, v in est_all(X, net).items()})
        print("E1", g, net, E1[g][net]["mean"], E1[g][net]["dev"], E1[g][net]["khat"])
res["E1"] = E1

# ---------------------------------------------------------------- E2
rng = np.random.default_rng(SEED)
bench = {}
for net in ("SP", "PS"):
    Xb = sim_exact(rng, R_BENCH, n, net=net)
    eb = est_all(Xb, net)
    bench[net] = {s: summary(v) for s, v in eb.items()}
    bench[net]["complete"]["P_eq10"] = float((eb["complete"] == 10).mean())
    bench[net]["complete"]["P_le8"] = float((eb["complete"] <= 8).mean())
    bench[net]["max_khat"] = {s: int(v.max()) for s, v in eb.items()}
    bench[net]["n_khat_1"] = {s: int((v == 1).sum()) for s, v in eb.items()}
    # 95 % band for a sample of R = 1000 replications (normal approximation)
    for s, v in ((k, x) for k, x in bench[net].items() if k in ("complete", "C3", "C5")):
        p = v["exact"] / 100
        v["band_exact"] = [100 * (p - 1.96 * np.sqrt(p * (1 - p) / R)),
                           100 * (p + 1.96 * np.sqrt(p * (1 - p) / R))]
res["benchmark"] = bench

sources = {g: load_digits(f[0]) for g, f in FILES.items()}
sources["RANDU"] = randu_digits(DIGITS_PER_FILE)

E2 = {}
sq = {}          # squared errors, for the significance of ΔMSE
first_samples = {}
for g, d in sources.items():
    E2[g] = {}
    for meth, sim in (("inverse", sim_inverse), ("structural", sim_structural)):
        E2[g][meth] = {}
        for net in ("SP", "PS"):
            X = sim(d, R, n, net=net)
            est = est_all(X, net)
            E2[g][meth][net] = {s: summary(v) for s, v in est.items()}
            E2[g][meth][net]["mean_x"] = float(X.mean())
            for s, v in est.items():
                sq[(g, meth, net, s)] = (v - K_TRUE) ** 2
            if meth == "inverse":
                x1 = X[0]
                ll = {}
                for K in (9, 10):
                    p = pmf(K, net=net, support=31)
                    ll[K] = float(np.sum(np.log(p[x1])))
                first_samples[(g, net)] = dict(sample=x1.tolist(), mean=float(x1.mean()),
                                               max=int(x1.max()), khat=int(est["complete"][0]),
                                               logL9=ll[9], logL10=ll[10])
res["E2"] = E2
res["first_samples"] = {f"{g}|{net}": v for (g, net), v in first_samples.items()}

# ΔMSE = MSE(inverse) - MSE(structural) and its standard error
D = {}
for g in sources:
    for net in ("SP", "PS"):
        for s in ("complete", "C3", "C5"):
            a, b = sq[(g, "inverse", net, s)], sq[(g, "structural", net, s)]
            dm = a.mean() - b.mean()
            se = np.sqrt(a.var(ddof=1) / R + b.var(ddof=1) / R)
            dex = E2[g]["inverse"][net][s]["exact"] - E2[g]["structural"][net][s]["exact"]
            D[f"{g}|{net}|{s}"] = dict(dMSE=dm, se=se, z=dm / se, dExact=dex)
res["E2_delta"] = D

# ---------------------------------------------------------------- E3
E3 = {}
for g, files in FILES.items():
    E3[g] = {}
    for net in ("SP", "PS"):
        per_file = {s: [] for s in ("complete", "C3", "C5")}
        mse_file = {s: [] for s in ("complete", "C3", "C5")}
        for f in files:
            X = sim_inverse(load_digits(f), R, n, net=net)
            for s, v in est_all(X, net).items():
                sm = summary(v)
                per_file[s].append(sm["exact"])
                mse_file[s].append(sm["mse"])
        E3[g][net] = {s: dict(exact=per_file[s], mse=mse_file[s],
                              mean_exact=float(np.mean(per_file[s])),
                              sd_exact=float(np.std(per_file[s], ddof=1)),
                              mean_mse=float(np.mean(mse_file[s])),
                              sd_mse=float(np.std(mse_file[s], ddof=1)))
                      for s in per_file}
        print("E3", g, net, {s: round(E3[g][net][s]["mean_exact"], 1) for s in per_file})
res["E3"] = E3

# ---------------------------------------------------------------- E4
E4 = {"n": {}, "config": {}}
gens = {g: sources[g] for g in FILES}
for nn in (10, 20, 33, 50, 100):
    E4["n"][nn] = {}
    for net in ("SP", "PS"):
        row = {}
        for g, d in gens.items():
            row[g] = summary(mle_complete(sim_inverse(d, R, nn, net=net), net=net))
        Xb = sim_exact(rng, 100_000, nn, net=net)
        row["benchmark"] = summary(mle_complete(Xb, net=net))
        E4["n"][nn][net] = row
    print("E4 n", nn, {net: {g: round(v["exact"], 1) for g, v in E4["n"][nn][net].items()}
                       for net in ("SP", "PS")})
for K in (5, 9, 15):
    for (N, M) in ((3, 2), (2, 3), (2, 2)):
        key = f"K={K}|N={N}|M={M}"
        E4["config"][key] = {}
        for net in ("SP", "PS"):
            row = {}
            for g, d in gens.items():
                X = sim_inverse(d, R, n, K=K, N=N, M=M, net=net)
                row[g] = summary(mle_complete(X, N, M, net, k_max=max(30, 3 * K)), K=K)
            Xb = sim_exact(rng, 100_000, n, K=K, N=N, M=M, net=net)
            row["benchmark"] = summary(mle_complete(Xb, N, M, net, k_max=max(30, 3 * K)), K=K)
            E4["config"][key][net] = row
        print("E4 cfg", key, {net: {g: round(v["exact"], 1) for g, v in E4["config"][key][net].items()}
                               for net in ("SP", "PS")})
res["E4"] = E4


# ---------------------------------------------------------------- E5
def sticky_digits(rng, size, rho):
    """Serially dependent digits: repeat previous digit with probability rho."""
    fresh = rng.integers(0, 10, size)
    rep = rng.random(size) < rho
    idx = np.where(~rep, np.arange(size), 0)
    np.maximum.accumulate(idx, out=idx)
    return fresh[idx].astype(np.uint8)


def biased_digits(rng, size, delta):
    """Non-uniform marginal: P(9) = 0.1 - delta, others share the excess."""
    p = np.full(10, 0.1 + delta / 9)
    p[9] = 0.1 - delta
    return rng.choice(10, size=size, p=p).astype(np.uint8)


E5 = {"sticky": {}, "biased": {}}
R5 = 5000
size = R5 * n * 9
for rho in (0.0, 0.05, 0.10, 0.20, 0.30):
    E5["sticky"][rho] = {}
    for meth, sim in (("inverse", sim_inverse), ("structural", sim_structural)):
        for net in ("SP", "PS"):
            X = sim(sticky_digits(rng, size, rho), R5, n, net=net)
            E5["sticky"][rho][f"{meth}|{net}"] = summary(mle_complete(X, net=net))
for delta in (0.0, 0.01, 0.02, 0.03, 0.05):
    E5["biased"][delta] = {}
    for meth, sim in (("inverse", sim_inverse), ("structural", sim_structural)):
        for net in ("SP", "PS"):
            X = sim(biased_digits(rng, size, delta), R5, n, net=net)
            E5["biased"][delta][f"{meth}|{net}"] = summary(mle_complete(X, net=net))
for fam in E5:
    for lev, v in E5[fam].items():
        print("E5", fam, lev, {k: (round(s["exact"], 1), round(s["mse"], 3)) for k, s in v.items()})
res["E5"] = E5

# pooled check of Java ThreadLocalRandom over the six files for the configurations of Table 6
res["E4"]["pooled_java"] = {}
for (K, N, M, net) in ((9, 3, 2, "SP"), (9, 2, 3, "PS"), (9, 2, 2, "SP"), (15, 3, 2, "SP"),
                       (15, 2, 3, "SP"), (15, 2, 3, "PS")):
    ex = [float((mle_complete(sim_inverse(load_digits(f), R, n, K=K, N=N, M=M, net=net), N, M, net,
                              k_max=max(30, 3 * K)) == K).mean()) for f in FILES["Java ThreadLocalRandom"]]
    b = float((mle_complete(sim_exact(rng, R_BENCH, n, K=K, N=N, M=M, net=net), N, M, net,
                            k_max=max(30, 3 * K)) == K).mean())
    p = float(np.mean(ex))
    res["E4"]["pooled_java"][f"K={K}|N={N}|M={M}|{net}"] = dict(per_file=ex, pooled=p, bench=b,
                                                         z=(p - b) / np.sqrt(b * (1 - b) / (6 * R)))
    print("E4 pooled Java", K, N, M, net, round(100 * p, 2), round(100 * b, 2))
(OUT / "results_v2.json").write_text(json.dumps(to_py(res), indent=1, ensure_ascii=False,
                                                default=str))
print("saved", OUT / "results_v2.json")
