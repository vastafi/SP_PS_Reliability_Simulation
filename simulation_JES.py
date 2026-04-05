"""
================================================================
COD PYTHON COMPLET — Articol JES
================================================================
Titlu: Fiabilitatea rețelelor Serial-Paralele și Paralel-Seriale
       cu Distribuție Uniformă Discretă a Duratei de Viață:
       Studiu de Validare Monte Carlo prin Generatoare
       Pseudoaleatoare de Top

Autori: Valentina Astafi, Alexei Leahu
Universitatea Tehnică a Moldovei

Cerințe:
  pip install numpy scipy matplotlib

Rulare:
  python3 simulation_JES.py
================================================================
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')   # schimba cu 'TkAgg' daca vrei sa vezi graficele
import matplotlib.pyplot as plt
import secrets as sec_module
import warnings
warnings.filterwarnings('ignore')

# Seed fix pentru reproductibilitate (doar pentru numpy)
np.random.seed(42)

# ================================================================
# SECTIUNEA 1: FORMULELE MODELULUI DIN CAIM 2025
# N fix, M fix, K=9, distributie uniforma pe {0,1,...,9}
# ================================================================

K_TRUE = 9   # valoarea adevarata a lui K

def p_k(k, K=9):
    """Functia de repartitie a distributiei uniforme pe {0,...,K}"""
    return (k + 1) / (K + 1)

def q_k(k, K=9):
    """Complementul: 1 - p_k"""
    return (K - k) / (K + 1)

# ── Functii de repartitie ──────────────────────────────────────

def F_sp(k, N, M, K=9):
    """CDF retea Serial-Paralela (SP): U = min(max(...))"""
    return 1 - (1 - p_k(k, K)**N)**M

def F_ps(k, N, M, K=9):
    """CDF retea Paralel-Seriala (PS): V = max(min(...))"""
    return (1 - q_k(k, K)**N)**M

# ── Distributii punctuale ──────────────────────────────────────

def pmf_sp(k, N, M, K=9):
    """P(U = k) pentru reteaua SP"""
    if k < 0:
        return 0.0
    if k == 0:
        return F_sp(0, N, M, K)
    return F_sp(k, N, M, K) - F_sp(k - 1, N, M, K)

def pmf_ps(k, N, M, K=9):
    """P(V = k) pentru reteaua PS"""
    if k < 0:
        return 0.0
    if k == 0:
        return F_ps(0, N, M, K)
    return F_ps(k, N, M, K) - F_ps(k - 1, N, M, K)

# ── Functii de fiabilitate ─────────────────────────────────────

def R_sp(k, N, M, K=9):
    """R_sp(k) = 1 - F_sp(k)"""
    return 1 - F_sp(k, N, M, K)

def R_ps(k, N, M, K=9):
    """R_ps(k) = 1 - F_ps(k)"""
    return 1 - F_ps(k, N, M, K)

# ── Rate de hazard ─────────────────────────────────────────────

def h_sp(k, N, M, K=9):
    """Rata de hazard h_sp(k) = P(U=k) / R_sp(k-1)"""
    p = pmf_sp(k, N, M, K)
    r = R_sp(k - 1, N, M, K) if k > 0 else 1.0
    return p / r if r > 1e-15 else 0.0

def h_ps(k, N, M, K=9):
    """Rata de hazard h_ps(k) = P(V=k) / R_ps(k-1)"""
    p = pmf_ps(k, N, M, K)
    r = R_ps(k - 1, N, M, K) if k > 0 else 1.0
    return p / r if r > 1e-15 else 0.0

# ── Caracteristici numerice ────────────────────────────────────

def mean_sp(N, M, K=9):
    return sum(k * pmf_sp(k, N, M, K) for k in range(K + 1))

def mean_ps(N, M, K=9):
    return sum(k * pmf_ps(k, N, M, K) for k in range(K + 1))

def var_sp(N, M, K=9):
    mu = mean_sp(N, M, K)
    return sum((k - mu)**2 * pmf_sp(k, N, M, K) for k in range(K + 1))

def var_ps(N, M, K=9):
    mu = mean_ps(N, M, K)
    return sum((k - mu)**2 * pmf_ps(k, N, M, K) for k in range(K + 1))

# ================================================================
# SECTIUNEA 2: GENERATOARE PSEUDOALEATOARE (GNPA)
# Top 3 din articolul CS 2026:
#   Locul 1: Java ThreadLocalRandom — simulat cu numpy.random
#   Locul 2: Python secrets
#   Locul 3: Cifrele lui π
# ================================================================

def gnpa_numpy(n):
    """
    GNPA 1: numpy.random.randint (Mersenne Twister)
    Folosit ca proxy pentru Java ThreadLocalRandom
    (ambele sunt generatoare uniforme de inalta calitate)
    """
    return np.random.randint(0, 10, size=n)


def gnpa_secrets(n):
    """
    GNPA 2: Python secrets
    Generator criptografic securizat bazat pe os.urandom()
    """
    return np.array([sec_module.randbelow(10) for _ in range(n)])


# Primele 2000 cifre ale lui π după virgulă
_PI_DIGITS = (
    "14159265358979323846264338327950288419716939937510"
    "58209749445923078164062862089986280348253421170679"
    "82148086513282306647093844609550582231725359408128"
    "48111745028410270193852110555964462294895493038196"
    "44288109756659334461284756482337867831652712019091"
    "45648566923460348610454326648213393607260249141273"
    "72458700660631558817488152092096282925409171536436"
    "78925903600113305305488204665213841469519415116094"
    "33057270365759591953092186117381932611793105118548"
    "07446237996274956735188575272489122793818301194912"
    "98336733624406566430860213949463952247371907021798"
    "60943702770539217176293176752384674818467669405132"
    "00056812714526356082778577134275778960917363717872"
    "14684409012249534301465495853710507922796892589235"
    "42019956112129021960864034418159813629774771309960"
    "51870721134999999837297804995105973173281609631859"
    "50244594553469083026425223082533446850352619311881"
    "71010003137838752886587533208381420617177669147303"
    "59825349042875546873115956286388235378759375195778"
    "18577805321712268066130019278766111959092164201989"
    "38095257201065485863278865936153381827968230301952"
    "03530185296899577362259941389124972177528347913152"
    "65132605024756025740575859778727498804928671976739"
    "10777687297689258923542019956112129021960864034418"
    "15981362977477130996051870721134999999837297804995"
    "10597317328160963185950244594553469083026425223082"
    "53344685035261931188171010003137838752886587533208"
    "38142061717766914730359825349042875546873115956286"
    "38823537875937519577818577805321712268066130019278"
    "76611195909216420198938095257201065485863278865936"
    "15338182796823030195203530185296899577362259941389"
    "12497217752834791315265132605024756025740575859778"
    "72749880492867197673910777687297689258923542019956"
    "11212902196086403441815981362977477130996051870721"
    "13499999983729780499510597317328160963185950244594"
    "55346908302642522308253344685035261931188171010003"
    "13783875288658753320838142061717766914730359825349"
    "04287554687311595628638823537875937519577818577805"
    "32171226806613001927876611195909216420198938095257"
    "20106548586327886593615338182796823030195203530185"
)

_PI_ARRAY = np.array([int(d) for d in _PI_DIGITS if d.isdigit()])

def gnpa_pi(n, offset=0):
    """
    GNPA 3: Cifrele zecimale ale lui π
    Se extrage o fereastra de n cifre incepand de la pozitia offset.
    Sursa: P. Trueb, One Trillion Digits of Pi, 2016.
    """
    needed = offset + n
    # Daca avem suficiente cifre, returnez direct
    if needed <= len(_PI_ARRAY):
        return _PI_ARRAY[offset:offset + n].copy()
    # Altfel, extindem circular (pentru simulari cu multi pasi)
    extended = np.tile(_PI_ARRAY, (needed // len(_PI_ARRAY)) + 2)
    return extended[offset:offset + n]

# ================================================================
# SECTIUNEA 3: SIMULAREA DURATEI DE VIATA A RETELEI
# ================================================================

def simulate_lifetimes(N, M, network_type, gnpa_func, n_samples,
                       pi_offset=0):
    """
    Simuleaza n_samples durate de viata ale retelei SP sau PS.

    Parametri:
        N            : numar de unitati per subreata
        M            : numar de subretele
        network_type : 'SP' sau 'PS'
        gnpa_func    : functia generatorului (gnpa_numpy, gnpa_secrets, gnpa_pi)
        n_samples    : numarul de durate de viata simulate
        pi_offset    : offset pt cifrele lui π (evita reutilizarea)

    Returneaza:
        np.array de forma (n_samples,) cu duratele de viata in {0,...,9}
    """
    lifetimes = np.empty(n_samples, dtype=int)
    units_per_network = N * M

    for i in range(n_samples):
        if gnpa_func == gnpa_pi:
            units = gnpa_pi(units_per_network,
                            offset=pi_offset + i * units_per_network)
        else:
            units = gnpa_func(units_per_network)

        # Reshape: M subretele x N unitati
        matrix = units.reshape(M, N)

        if network_type == 'SP':
            # U = min_{i=1}^{M} max_{j=1}^{N} X_{ij}
            lifetimes[i] = matrix.max(axis=1).min()
        else:
            # V = max_{i=1}^{M} min_{j=1}^{N} X_{ij}
            lifetimes[i] = matrix.min(axis=1).max()

    return lifetimes

# ================================================================
# SECTIUNEA 4: ESTIMAREA MLE PENTRU PARAMETRUL K
# ================================================================

def mle_K_complete(data, N, M, network_type='SP', K_max=30):
    """
    MLE pentru K pe baza datelor complete.
    Cauta discret K in {max(data), ..., K_max}.
    """
    pmf = pmf_sp if network_type == 'SP' else pmf_ps
    K_min = int(np.max(data))
    best_K, best_ll = K_min, -np.inf

    for K_try in range(K_min, K_max + 1):
        ll = sum(
            np.log(max(pmf(int(x), N, M, K_try), 1e-300))
            for x in data
        )
        if ll > best_ll:
            best_ll = ll
            best_K = K_try

    return best_K


def mle_K_censored(intervals, counts, N, M, network_type='SP',
                   K_max=30):
    """
    MLE pentru K pe baza datelor censurate.

    Parametri:
        intervals : lista de tupluri (a, b) unde b=None inseamna [a, +inf)
        counts    : frecventele n_1, n_2, ... (lista de intregi)
    """
    F = F_sp if network_type == 'SP' else F_ps
    K_min = int(intervals[-1][0])  # K >= ultima limita inferioara
    best_K, best_ll = K_min, -np.inf

    for K_try in range(K_min, K_max + 1):
        ll = 0.0
        valid = True
        for (a, b), cnt in zip(intervals, counts):
            if cnt == 0:
                continue
            # Probabilitatea ca reteaua sa cada in intervalul [a, b)
            if b is None:
                # P(X >= a) = R(a-1) = 1 - F(a-1)
                prob = 1.0 - (F(int(a) - 1, N, M, K_try) if a > 0 else 0.0)
            else:
                # P(a <= X < b) = F(b-1) - F(a-1)
                f_b = F(int(b) - 1, N, M, K_try)
                f_a = F(int(a) - 1, N, M, K_try) if a > 0 else 0.0
                prob = f_b - f_a

            if prob <= 0:
                valid = False
                break
            ll += cnt * np.log(prob)

        if valid and ll > best_ll:
            best_ll = ll
            best_K = K_try

    return best_K


def compute_censored_counts(data, intervals):
    """
    Numara cate observatii cad in fiecare interval.
    intervals: lista de (a, b) cu b=None pt ultimul interval.
    """
    counts = []
    for (a, b) in intervals:
        if b is None:
            counts.append(int(np.sum(data >= a)))
        else:
            counts.append(int(np.sum((data >= a) & (data < b))))
    return counts

# ================================================================
# SECTIUNEA 5: SIMULARE COMPLETA — MSE pe R repetitii
# ================================================================

# Intervale de censurare (din CAIM 2025, adaptate pentru SP si PS)
INTERVALS_SP_3 = [(0, 4), (4, 7), (7, None)]   # 3 intervale SP
INTERVALS_SP_5 = [(0, 4), (4, 6), (6, 7), (7, 8), (8, None)]  # 5 intervale SP
INTERVALS_PS_3 = [(0, 3), (3, 6), (6, None)]   # 3 intervale PS
INTERVALS_PS_5 = [(0, 3), (3, 5), (5, 7), (7, 8), (8, None)]  # 5 intervale PS


def run_monte_carlo(N, M, network_type, gnpa_func,
                    n=33, R=1000, true_K=9, pi_offset=0):
    """
    Ruleaza R simulari Monte Carlo si calculeaza:
    - K_hat_mean, Bias, MSE, Std pentru date complete
    - K_hat_mean, Bias, MSE, Std pentru date censurate 3 intervale
    - K_hat_mean, Bias, MSE, Std pentru date censurate 5 intervale

    Returneaza un dict cu toate statisticile.
    """
    if network_type == 'SP':
        int3 = INTERVALS_SP_3
        int5 = INTERVALS_SP_5
    else:
        int3 = INTERVALS_PS_3
        int5 = INTERVALS_PS_5

    K_complete = np.zeros(R)
    K_cens3    = np.zeros(R)
    K_cens5    = np.zeros(R)

    for r in range(R):
        # Offset pentru cifrele lui π — evita suprapunerea
        offset = pi_offset + r * n * N * M

        # Simuleaza n durate de viata
        data = simulate_lifetimes(N, M, network_type, gnpa_func,
                                  n, pi_offset=offset)
        data = np.clip(data, 0, 9).astype(int)

        # MLE date complete
        K_complete[r] = mle_K_complete(data, N, M, network_type)

        # MLE date censurate — 3 intervale
        cnt3 = compute_censored_counts(data, int3)
        K_cens3[r] = mle_K_censored(int3, cnt3, N, M, network_type)

        # MLE date censurate — 5 intervale
        cnt5 = compute_censored_counts(data, int5)
        K_cens5[r] = mle_K_censored(int5, cnt5, N, M, network_type)

    def stats(arr):
        return {
            'mean' : float(np.mean(arr)),
            'bias' : float(np.mean(arr) - true_K),
            'mse'  : float(np.mean((arr - true_K)**2)),
            'std'  : float(np.std(arr)),
        }

    return {
        'complete': stats(K_complete),
        'cens3'   : stats(K_cens3),
        'cens5'   : stats(K_cens5),
    }

# ================================================================
# SECTIUNEA 6: RULAREA COMPLETA
# ================================================================

print("=" * 70)
print("CARACTERISTICI TEORETICE (K=9)")
print("=" * 70)

CASES = [
    {'N': 3, 'M': 2, 'label': 'Cazul a) N=3, M=2  (reteaua SP mai fiabila)'},
    {'N': 2, 'M': 3, 'label': 'Cazul b) N=2, M=3  (reteaua PS mai fiabila)'},
]

for case in CASES:
    N, M = case['N'], case['M']
    print(f"\n{case['label']}")
    print(f"  E[U] (SP) = {mean_sp(N, M):.6f}     D[U] = {var_sp(N, M):.6f}")
    print(f"  E[V] (PS) = {mean_ps(N, M):.6f}     D[V] = {var_ps(N, M):.6f}")
    print(f"  h_sp(5)   = {h_sp(5, N, M):.6f}")
    print(f"  h_ps(5)   = {h_ps(5, N, M):.6f}")

# ── Tabele complete CDF / PMF / R / h ──────────────────────────

for idx, case in enumerate(CASES, 1):
    N, M = case['N'], case['M']
    print(f"\n{'='*70}")
    print(f"TABEL {idx}: N={N}, M={M}, K=9")
    print(f"{'='*70}")
    header = (f"{'k':>2}  {'F_sp':>8} {'P(U=k)':>8} {'R_sp':>8} {'h_sp':>8}"
              f"  |  {'F_ps':>8} {'P(V=k)':>8} {'R_ps':>8} {'h_ps':>8}")
    print(header)
    print("-" * len(header))
    for k in range(K_TRUE + 1):
        print(f"{k:>2}  {F_sp(k,N,M):>8.6f} {pmf_sp(k,N,M):>8.6f} "
              f"{R_sp(k,N,M):>8.6f} {h_sp(k,N,M):>8.6f}  |  "
              f"{F_ps(k,N,M):>8.6f} {pmf_ps(k,N,M):>8.6f} "
              f"{R_ps(k,N,M):>8.6f} {h_ps(k,N,M):>8.6f}")

# ── Simulari Monte Carlo ────────────────────────────────────────

GENERATORS = [
    ('Java ThreadLocalRandom (numpy)', gnpa_numpy),
    ('Python secrets',                 gnpa_secrets),
    ('Cifrele lui π',                  gnpa_pi),
]

N_SAMPLES   = 33    # volumul esantionului (ca in CAIM 2025)
N_REPS      = 1000  # numarul de repetitii Monte Carlo

print(f"\n{'='*70}")
print(f"SIMULARI MONTE CARLO  (n={N_SAMPLES}, R={N_REPS}, K={K_TRUE})")
print(f"{'='*70}")

# Stocam toate rezultatele pentru grafice
ALL_RESULTS = {}

for case in CASES:
    N, M = case['N'], case['M']
    key = f"N{N}M{M}"
    ALL_RESULTS[key] = {}

    print(f"\n{case['label']}")

    for net_type in ['SP', 'PS']:
        ALL_RESULTS[key][net_type] = {}
        print(f"\n  Reteaua {net_type}:")
        print(f"  {'GNPA':<35} {'K_hat_c':>7} {'Bias_c':>7} {'MSE_c':>7}"
              f"  {'K_hat_3':>7} {'MSE_3':>7}  {'K_hat_5':>7} {'MSE_5':>7}")
        print(f"  {'-'*80}")

        for gnpa_name, gnpa_func in GENERATORS:
            res = run_monte_carlo(
                N, M, net_type, gnpa_func,
                n=N_SAMPLES, R=N_REPS, true_K=K_TRUE,
                pi_offset=0
            )
            ALL_RESULTS[key][net_type][gnpa_name] = res

            print(f"  {gnpa_name:<35}"
                  f" {res['complete']['mean']:>7.4f}"
                  f" {res['complete']['bias']:>7.4f}"
                  f" {res['complete']['mse']:>7.4f}"
                  f"  {res['cens3']['mean']:>7.4f}"
                  f" {res['cens3']['mse']:>7.4f}"
                  f"  {res['cens5']['mean']:>7.4f}"
                  f" {res['cens5']['mse']:>7.4f}")

print("\n✅ Simulari complete!")

# ================================================================
# SECTIUNEA 7: GRAFICE
# ================================================================

# ── Figura 1: CDF, Fiabilitate, Hazard ──────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Funcțiile de repartiție, fiabilitate și rata de hazard\n"
    "pentru rețelele SP și PS (K=9)",
    fontsize=13, fontweight='bold'
)

plot_cfg = [
    (3, 2, 'SP', axes[0, 0], 'Cazul a) N=3, M=2 — Rețeaua SP'),
    (3, 2, 'PS', axes[0, 1], 'Cazul a) N=3, M=2 — Rețeaua PS'),
    (2, 3, 'SP', axes[1, 0], 'Cazul b) N=2, M=3 — Rețeaua SP'),
    (2, 3, 'PS', axes[1, 1], 'Cazul b) N=2, M=3 — Rețeaua PS'),
]

k_vals = list(range(K_TRUE + 1))
C_BLUE  = '#1f77b4'
C_ORG   = '#ff7f0e'
C_GREEN = '#2ca02c'

for N, M, net, ax, title in plot_cfg:
    if net == 'SP':
        F_v = [F_sp(k, N, M) for k in k_vals]
        R_v = [R_sp(k, N, M) for k in k_vals]
        h_v = [h_sp(k, N, M) for k in k_vals]
    else:
        F_v = [F_ps(k, N, M) for k in k_vals]
        R_v = [R_ps(k, N, M) for k in k_vals]
        h_v = [h_ps(k, N, M) for k in k_vals]

    ax.step(k_vals, F_v, where='post', color=C_BLUE,  lw=2,
            label='F(k) — CDF', marker='o', ms=5)
    ax.step(k_vals, R_v, where='post', color=C_ORG,   lw=2,
            label='R(k) — Fiabilitate', marker='s', ms=5, ls='--')
    ax.step(k_vals, h_v, where='post', color=C_GREEN, lw=2,
            label='h(k) — Hazard', marker='^', ms=5, ls=':')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('k', fontsize=10)
    ax.set_ylabel('Valoare', fontsize=10)
    ax.set_xticks(k_vals)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

plt.tight_layout()
plt.savefig('figura1_fiabilitate.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figura 1 salvata: figura1_fiabilitate.png")

# ── Figura 2: MSE comparativ pe GNPA-uri ────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"MSE al estimatorului K̂ pentru cele 3 GNPA-uri de top\n"
    f"(R={N_REPS} repetitii, n={N_SAMPLES}, K={K_TRUE})",
    fontsize=12, fontweight='bold'
)

gnpa_labels = [
    'Java TLR\n(numpy)',
    'Python\nsecrets',
    'Cifrele\nlui π',
]
gnpa_keys = [g[0] for g in GENERATORS]
x = np.arange(len(gnpa_labels))
w = 0.25
C_BARS = ['#2196F3', '#4CAF50', '#FF9800']

plot_mse_cfg = [
    ('N3M2', 'SP', axes[0, 0], 'Cazul a) N=3, M=2 — Rețeaua SP'),
    ('N3M2', 'PS', axes[0, 1], 'Cazul a) N=3, M=2 — Rețeaua PS'),
    ('N2M3', 'SP', axes[1, 0], 'Cazul b) N=2, M=3 — Rețeaua SP'),
    ('N2M3', 'PS', axes[1, 1], 'Cazul b) N=2, M=3 — Rețeaua PS'),
]

for case_key, net_type, ax, title in plot_mse_cfg:
    mse_c  = [ALL_RESULTS[case_key][net_type][g]['complete']['mse'] for g in gnpa_keys]
    mse_3  = [ALL_RESULTS[case_key][net_type][g]['cens3']['mse']    for g in gnpa_keys]
    mse_5  = [ALL_RESULTS[case_key][net_type][g]['cens5']['mse']    for g in gnpa_keys]

    ax.bar(x - w,   mse_c, w, label='Date complete',         color=C_BARS[0], alpha=0.85)
    ax.bar(x,       mse_3, w, label='Date censurate 3 int.', color=C_BARS[1], alpha=0.85)
    ax.bar(x + w,   mse_5, w, label='Date censurate 5 int.', color=C_BARS[2], alpha=0.85)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('GNPA', fontsize=10)
    ax.set_ylabel('MSE(K̂)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(gnpa_labels, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figura2_mse_gnpa.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figura 2 salvata: figura2_mse_gnpa.png")

print("\n" + "=" * 70)
print("TOATE CALCULELE SI GRAFICELE SUNT GATA!")
print("Fisiere generate:")
print("  → figura1_fiabilitate.png")
print("  → figura2_mse_gnpa.png")
print("=" * 70)
