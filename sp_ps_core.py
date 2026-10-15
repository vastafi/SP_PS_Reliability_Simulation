"""
sp_ps_core.py  (revision v2)
Core model, data loading, simulation and maximum likelihood estimation (MLE)
for serial-parallel (SP) and parallel-serial (PS) networks whose unit lifetimes
are i.i.d. discrete uniform on {0, 1, ..., K}.

Model (unified, used everywhere in the paper):
    SP:  U = min_{1<=i<=M} max_{1<=j<=N} X_ij
    PS:  V = max_{1<=i<=M} min_{1<=j<=N} X_ij
    with N = 3 units per sub-network and M = 2 sub-networks (Figure 1).

Changes with respect to the first submission
  * PS network simulated with the same (N, M) = (3, 2) as in the theory
    (the first version of the MLE scripts used (N, M) = (2, 3) for PS);
  * censored-data likelihood: an interval with zero observations contributes
    the factor p^0 = 1 and therefore does NOT constrain K (the first version
    rejected every K for which an empty interval had probability 0, which
    implicitly imposed K >= 9 under scheme C5);
  * fully vectorised (NumPy), relative paths, explicit parameter space for K.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# files used in the first submission (sample no. 1 = first listed file)
FILES = {
    "Java ThreadLocalRandom": [DATA / "java_threadlocal" / f"threadlocal_data{s}.csv"
                               for s in ["1", "2", "3", "4", "5", ""]],
    "Python secrets": [DATA / "python_secrets" / f"secrets_data{s}.csv"
                       for s in ["1", "2", "3", "4", "5", ""]],
    "Digits of π": [DATA / "pi" / f"pi_digits_part{s}.csv"
                    for s in ["", "1", "2", "3", "4", "5"]],
}
DIGITS_PER_FILE = 2_614_690

# ---------------------------------------------------------------- model
K_TRUE, N_UNITS, M_SUB = 9, 3, 2
K_MAX = 30                     # upper bound of the parameter space for K


def cdf(k, K, N=N_UNITS, M=M_SUB, net="SP"):
    """CDF of the network lifetime at integer point(s) k, Eqs. (2)-(3)."""
    k = np.asarray(k, dtype=float)
    p = np.clip((k + 1) / (K + 1), 0.0, 1.0)
    q = np.clip((K - k) / (K + 1), 0.0, 1.0)
    F = 1.0 - (1.0 - p ** N) ** M if net == "SP" else (1.0 - q ** N) ** M
    return np.where(k < 0, 0.0, np.where(k >= K, 1.0, F))


def pmf(K, N=N_UNITS, M=M_SUB, net="SP", support=None):
    """P(X = k), k = 0..support-1 (zero outside {0..K}), Eq. (4)."""
    support = K + 1 if support is None else support
    k = np.arange(support)
    return cdf(k, K, N, M, net) - cdf(k - 1, K, N, M, net)


def theory(K=K_TRUE, N=N_UNITS, M=M_SUB):
    """Theoretical characteristics used in Section 2.1 / Table 1."""
    out = {}
    k = np.arange(K + 1)
    for net in ("SP", "PS"):
        F = cdf(k, K, N, M, net)
        P = pmf(K, N, M, net)
        R = 1 - F
        Rm1 = np.concatenate([[1.0], R[:-1]])
        h = P / Rm1
        mu = float((k * P).sum())
        var = float(((k - mu) ** 2 * P).sum())
        out[net] = dict(F=F, P=P, R=R, h=h, mean=mu, var=var, sd=math.sqrt(var))
    return out


def clt_volume(mu, sigma, eps_rel=1e-3, alpha=0.05):
    """Eq. (13)-(14): n >= floor((z * sigma / eps)^2) + 1, eps = eps_rel * mu."""
    z = norm.ppf(1 - alpha / 2)
    eps = eps_rel * mu
    return int(math.floor((z * sigma / eps) ** 2)) + 1, eps, z


# ---------------------------------------------------------------- data
def load_digits(path) -> np.ndarray:
    """Read a CSV with one digit per line (LF or CRLF) into a uint8 array."""
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    d = raw[(raw >= 48) & (raw <= 57)] - 48
    return d.astype(np.uint8)


def load_concat(files, n_digits):
    parts, total = [], 0
    for f in files:
        d = load_digits(f)
        parts.append(d)
        total += d.size
        if total >= n_digits:
            break
    d = np.concatenate(parts)
    if d.size < n_digits:
        raise ValueError(f"need {n_digits} digits, have {d.size}")
    return d[:n_digits]


def randu_digits(n, seed=1):
    """Digits floor(10*u) from RANDU, x_{k+1} = 65539 x_k mod 2^31 (Knuth [11]).
    Used as a deliberately defective reference generator."""
    out = np.empty(n, dtype=np.uint8)
    x = seed
    a, m = 65539, 2 ** 31
    for i in range(n):
        x = (a * x) % m
        out[i] = (10 * x) // m
    return out


def lcg_lowdigit(n, seed=1):
    """Digits x mod 10 of the LCG x_{k+1} = (1103515245 x_k + 12345) mod 2^31:
    the classic 'rand() % 10' misuse of the low-order bits (period of the last
    bit equals 2), a second, strongly defective reference generator."""
    out = np.empty(n, dtype=np.uint8)
    x = seed
    for i in range(n):
        x = (1103515245 * x + 12345) % 2147483648
        out[i] = x % 10
    return out


# ---------------------------------------------------------------- simulation
def sim_inverse(digits, R, n, K=K_TRUE, N=N_UNITS, M=M_SUB, net="SP", offset=0):
    """Method 1: 9 consecutive digits -> u in [0,1); lifetime = min{k: F(k) >= u}."""
    need = R * n * 9
    blk = digits[offset:offset + need].reshape(R * n, 9).astype(np.int64)
    u = blk @ (10 ** np.arange(8, -1, -1, dtype=np.int64)) / 1e9
    F = cdf(np.arange(K + 1), K, N, M, net)
    x = np.searchsorted(F, u, side="left")
    return x.reshape(R, n)


def sim_structural(digits, R, n, N=N_UNITS, M=M_SUB, net="SP", offset=0):
    """Method 2: N*M digits per network (valid for K = 9 only, digits 0..9)."""
    need = R * n * N * M
    blk = digits[offset:offset + need].reshape(R * n, M, N)
    x = blk.max(axis=2).min(axis=1) if net == "SP" else blk.min(axis=2).max(axis=1)
    return x.reshape(R, n).astype(np.int64)


def sim_exact(rng, R, n, K=K_TRUE, N=N_UNITS, M=M_SUB, net="SP"):
    """Benchmark: lifetimes drawn from the exact PMF with NumPy PCG64."""
    return rng.choice(K + 1, size=(R, n), p=pmf(K, N, M, net))


# ---------------------------------------------------------------- MLE
def mle_complete(X, N=N_UNITS, M=M_SUB, net="SP", k_max=K_MAX):
    """Exhaustive maximisation of Eq. (10) over the parameter space
    Theta = {1, ..., k_max}; values K < max(x_i) have zero likelihood.
    Ties are resolved in favour of the smallest K."""
    X = np.asarray(X)
    V = k_max + 1
    counts = np.stack([(X == v).sum(axis=1) for v in range(V)], axis=1)  # (R, V)
    Ks = np.arange(1, k_max + 1)
    P = np.stack([pmf(K, N, M, net, support=V) for K in Ks])            # (nK, V)
    with np.errstate(divide="ignore"):
        logP = np.where(P > 0, np.log(np.where(P > 0, P, 1.0)), 0.0)
    LL = counts @ logP.T                                                  # (R, nK)
    impossible = (counts[:, None, :] > 0) & (P[None, :, :] <= 0)
    LL[impossible.any(axis=2)] = -np.inf
    return Ks[np.argmax(LL, axis=1)]


def mle_censored(X, bounds, N=N_UNITS, M=M_SUB, net="SP", k_max=K_MAX):
    """Maximisation of Eq. (11) for interval-grouped data.
    bounds = [a_0=0, a_1, ..., a_{m-1}] ; last interval is [a_{m-1}, +inf).
    Only intervals with n_j > 0 enter the likelihood."""
    X = np.asarray(X)
    edges = list(bounds) + [np.inf]
    counts = np.stack([((X >= a) & (X < b)).sum(axis=1)
                       for a, b in zip(edges[:-1], edges[1:])], axis=1)   # (R, m)
    Ks = np.arange(1, k_max + 1)
    P = []
    for K in Ks:
        Fa = np.array([cdf(a - 1, K, N, M, net) for a in edges[:-1]])
        Fb = np.array([1.0 if np.isinf(b) else cdf(b - 1, K, N, M, net) for b in edges[1:]])
        P.append(Fb - Fa)
    P = np.array(P)                                                       # (nK, m)
    with np.errstate(divide="ignore"):
        logP = np.where(P > 0, np.log(np.where(P > 0, P, 1.0)), 0.0)
    LL = counts @ logP.T
    impossible = (counts[:, None, :] > 0) & (P[None, :, :] <= 0)
    LL[impossible.any(axis=2)] = -np.inf
    return Ks[np.argmax(LL, axis=1)]


C3 = [0, 4, 7]            # [0,4), [4,7), [7,+inf)
C5 = [0, 5, 7, 8, 9]      # [0,5), [5,7), [7,8), [8,9), [9,+inf)


def summary(Khat, K=K_TRUE):
    Khat = np.asarray(Khat, dtype=float)
    m = Khat.mean()
    return dict(mean=m, var=Khat.var(), mse=((Khat - K) ** 2).mean(),
                exact=100 * (Khat == K).mean(),
                under=100 * (Khat < K).mean(), over=100 * (Khat > K).mean())
