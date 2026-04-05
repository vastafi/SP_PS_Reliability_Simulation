"""
Configurare centralizată: parametri model, căi fișiere, funcții teoretice.
Importat de toate celelalte scripturi.
"""

import math
from scipy.stats import norm

# ── Căi proiect ───────────────────────────────────────────────
DATA    = "./data/"
FIGURES = "./figures/"

# ── Parametri model ───────────────────────────────────────────
K = 9    # distribuție uniformă pe {0, 1, ..., K}
N = 3    # unități per subrețea
M = 2    # subretele

# ── Fișiere de date ───────────────────────────────────────────
FILES = {
    "Java ThreadLocalRandom": [
        DATA + "java_threadlocal/threadlocal_data1.csv",
        DATA + "java_threadlocal/threadlocal_data2.csv",
        DATA + "java_threadlocal/threadlocal_data3.csv",
        DATA + "java_threadlocal/threadlocal_data4.csv",
        DATA + "java_threadlocal/threadlocal_data5.csv",
    ],
    "Python secrets": [
        DATA + "python_secrets/secrets_data1.csv",
        DATA + "python_secrets/secrets_data2.csv",
        DATA + "python_secrets/secrets_data3.csv",
        DATA + "python_secrets/secrets_data4.csv",
        DATA + "python_secrets/secrets_data5.csv",
    ],
    "Cifrele lui pi": [
        DATA + "pi/pi_digits_part.csv",
        DATA + "pi/pi_digits_part1.csv",
        DATA + "pi/pi_digits_part2.csv",
        DATA + "pi/pi_digits_part3.csv",
        DATA + "pi/pi_digits_part4.csv",
        DATA + "pi/pi_digits_part5.csv",
    ],
}

# Ordinea de afișare în grafice
GNPA_ORDER  = ["Java ThreadLocalRandom", "Python secrets", "Cifrele lui pi"]
GNPA_LABELS = ["Java\nThreadLocalRandom", "Python\nsecrets", "Cifrele\nlui π"]
COLORS      = ['#1f77b4', '#ff7f0e', '#2ca02c']

# ── Scheme censurare ──────────────────────────────────────────
C3_SP = [0, 4, 7, K + 1]        # [0,4), [4,7), [7,+inf)
C3_PS = [0, 3, 6, K + 1]        # [0,3), [3,6), [6,+inf)

C5_SP = [0, 4, 6, 7, 8, K + 1]  # [0,4), [4,6), [6,7), [7,8), [8,+inf)
C5_PS = [0, 3, 5, 7, 8, K + 1]  # [0,3), [3,5), [5,7), [7,8), [8,+inf)


# ── Funcții model teoretic ────────────────────────────────────
def pmf_sp(k, n=N, m=M, kmax=K):
    """Distribuția punctuală P(U=k) pentru rețeaua Serial-Paralelă."""
    pk  = (k + 1) / (kmax + 1)
    pk0 = k / (kmax + 1)
    if k == 0:
        return 1 - (1 - pk ** n) ** m
    return (1 - pk0 ** n) ** m - (1 - pk ** n) ** m

def pmf_ps(k, n=N, m=M, kmax=K):
    """Distribuția punctuală P(V=k) pentru rețeaua Paralel-Serială."""
    qk  = (kmax - k) / (kmax + 1)
    qk0 = (kmax - k + 1) / (kmax + 1)
    if k == 0:
        return (1 - qk ** n) ** m
    return (1 - qk ** n) ** m - (1 - qk0 ** n) ** m

def F_sp(k, n=N, m=M, kmax=K):
    """CDF rețea SP."""
    return 1 - (1 - ((k + 1) / (kmax + 1)) ** n) ** m

def F_ps(k, n=N, m=M, kmax=K):
    """CDF rețea PS."""
    return (1 - ((kmax - k) / (kmax + 1)) ** n) ** m

def R_sp(k, n=N, m=M, kmax=K):
    """Funcția de fiabilitate SP."""
    return 1 - F_sp(k, n, m, kmax)

def R_ps(k, n=N, m=M, kmax=K):
    """Funcția de fiabilitate PS."""
    return 1 - F_ps(k, n, m, kmax)

def h_sp(k, n=N, m=M, kmax=K):
    """Rata de hazard SP."""
    p = pmf_sp(k, n, m, kmax)
    r = R_sp(k - 1, n, m, kmax) if k > 0 else 1.0
    return p / r if r > 1e-15 else 0.0

def h_ps(k, n=N, m=M, kmax=K):
    """Rata de hazard PS."""
    p = pmf_ps(k, n, m, kmax)
    r = R_ps(k - 1, n, m, kmax) if k > 0 else 1.0
    return p / r if r > 1e-15 else 0.0

def mean_theo(pmf_fn, kmax=K):
    return sum(k * pmf_fn(k) for k in range(kmax + 1))

def var_theo(pmf_fn, kmax=K):
    mu = mean_theo(pmf_fn, kmax)
    return sum((k - mu) ** 2 * pmf_fn(k) for k in range(kmax + 1))


# ── Calcul volum TLC ──────────────────────────────────────────
def calc_n_tlc(mu, sigma, eps_rel=0.001, alpha=0.05):
    """
    Returnează (n, eps, z) prin Teorema Limită Centrală.
    n >= floor((z_{1-alpha/2} * sigma / eps)^2) + 1
    cu eps = eps_rel * mu.
    """
    z   = norm.ppf(1 - alpha / 2)
    eps = eps_rel * mu
    n   = int(math.floor((z * sigma / eps) ** 2)) + 1
    return n, eps, z


# ── Valori globale precalculate (disponibile la import) ───────
MU_SP  = mean_theo(pmf_sp)
MU_PS  = mean_theo(pmf_ps)
VAR_SP = var_theo(pmf_sp)    # VAR_SP == VAR_PS prin simetrie
SIGMA  = math.sqrt(VAR_SP)

N_SP, EPS_SP, Z975 = calc_n_tlc(MU_SP, SIGMA)
N_PS, EPS_PS, _    = calc_n_tlc(MU_PS, SIGMA)


if __name__ == "__main__":
    print("=" * 55)
    print("CONFIG — Valori teoretice")
    print("=" * 55)
    print(f"  K={K}, N={N}, M={M}")
    print(f"  E[U] SP  = {MU_SP:.10f}")
    print(f"  E[V] PS  = {MU_PS:.10f}")
    print(f"  E[U]+E[V]= {MU_SP + MU_PS:.10f}  (trebuie = {K})")
    print(f"  D[U]=D[V]= {VAR_SP:.10f}")
    print(f"  sigma    = {SIGMA:.10f}")
    print(f"  z_0.975  = {Z975:.10f}")
    print(f"  n_SP     = {N_SP:>10,}  eps_SP = {EPS_SP:.10f}")
    print(f"  n_PS     = {N_PS:>10,}  eps_PS = {EPS_PS:.10f}")
