"""
Simulări Monte Carlo și estimatori MLE pentru rețelele SP și PS.
Citește date reale din fișierele CSV ale proiectului.
"""

import os
import math
import numpy as np
from config import (
    K, N, M, FILES, GNPA_ORDER,
    C3_SP, C3_PS, C5_SP, C5_PS,
    pmf_sp, pmf_ps,
    MU_SP, MU_PS, SIGMA, Z975,
    N_SP, N_PS, EPS_SP, EPS_PS,
)

# ════════════════════════════════════════════════════════════════
# 1. CITIRE DATE CSV
# ════════════════════════════════════════════════════════════════

def load_digits(file_list):
    """
    Citește cifre din fișierele CSV și returnează array NumPy 1D.
    Fiecare linie poate conține una sau mai multe cifre (separate prin
    virgulă sau spațiu).
    """
    digits = []
    for fpath in file_list:
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"\n  Fișier negăsit: {fpath}\n"
                f"  Verifică că BASE din config.py este corect."
            )
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for part in line.replace(',', ' ').split():
                    if part.isdigit():
                        digits.append(int(part))
    return np.array(digits, dtype=np.int32)


# ════════════════════════════════════════════════════════════════
# 2. SIMULARE REȚELE
# ════════════════════════════════════════════════════════════════

def simulate_network(digits, net_type, n_samples):
    """
    Calculează n_samples durate de viață simulate din cifre GNPA reale.

    digits    : array 1D de cifre pe {0,...,9}
    net_type  : 'SP'  → U = min_i(max_j(X_ij))
                'PS'  → V = max_i(min_j(X_ij))
    n_samples : număr de rețele de simulat

    Returnează array 1D de durate de viață.
    """
    cells_per_net = N * M
    n_available   = len(digits) // cells_per_net

    if n_samples > n_available:
        print(f"  ⚠  Cifre disponibile: {n_available:,}, "
              f"necesare: {n_samples:,}. Se folosesc {n_available:,}.")
        n_samples = n_available

    # Reorganizăm cifrele: (n_samples, M, N)
    data = digits[:n_samples * cells_per_net].reshape(n_samples, M, N)

    if net_type == 'SP':
        # max pe N unități (axis=2), min pe M subretele (axis=1)
        return np.min(np.max(data, axis=2), axis=1)
    elif net_type == 'PS':
        # min pe N unități (axis=2), max pe M subretele (axis=1)
        return np.max(np.min(data, axis=2), axis=1)
    else:
        raise ValueError(f"net_type trebuie 'SP' sau 'PS', nu '{net_type}'")


# ════════════════════════════════════════════════════════════════
# 3. ESTIMATORI MLE
# ════════════════════════════════════════════════════════════════

def mle_complete(lifetimes):
    """
    MLE pentru date complete.
    K̂ = max(observații) — estimator consistent pentru dist. uniformă discretă.
    """
    return int(np.max(lifetimes))


def mle_censored(lifetimes, scheme_bounds, net_type, k_max_search=30):
    """
    MLE pentru date censurate (date grupate în intervale).

    scheme_bounds : [a0, a1, ..., am] — limitele intervalelor
                    [a0,a1), [a1,a2), ..., [am-1,+inf)
    net_type      : 'SP' sau 'PS'
    """
    pmf_fn = pmf_sp if net_type == 'SP' else pmf_ps
    bounds = scheme_bounds

    # Frecvențele pe intervale
    freqs = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if i < len(bounds) - 2:
            cnt = int(np.sum((lifetimes >= lo) & (lifetimes < hi)))
        else:
            cnt = int(np.sum(lifetimes >= lo))
        freqs.append(cnt)

    # Căutare K̂ care maximizează log-verosimilitatea multinomială
    best_k  = int(np.max(lifetimes))
    best_ll = -math.inf

    for k_try in range(best_k, k_max_search + 1):
        probs = []
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            p = sum(pmf_fn(x, kmax=k_try)
                    for x in range(lo, min(hi, k_try + 1)))
            probs.append(p)

        # Log-verosimilitate
        ll = 0.0
        valid = True
        for f, p in zip(freqs, probs):
            if f > 0:
                if p <= 0:
                    valid = False
                    break
                ll += f * math.log(p)
        if not valid:
            continue

        if ll > best_ll:
            best_ll = ll
            best_k  = k_try

    return best_k


def mse(k_hat, k_true=K):
    return (k_hat - k_true) ** 2


# ════════════════════════════════════════════════════════════════
# 4. RULARE PRINCIPALĂ
# ════════════════════════════════════════════════════════════════

def run_simulations():
    """
    Rulează toate simulările și returnează dicționarul de rezultate.
    Structură: results[gnpa_name][net_type] = {cheie: valoare}
    """
    print("=" * 60)
    print("SIMULĂRI MONTE CARLO — Fiabilitate SP/PS")
    print(f"Parametri: K={K}, N={N}, M={M}")
    print("=" * 60)
    print(f"\nVolume TLC (eps_rel=0.1%, alpha=0.05, z={Z975:.4f}):")
    print(f"  n_SP = {N_SP:>10,}   eps_SP = {EPS_SP:.10f}")
    print(f"  n_PS = {N_PS:>10,}   eps_PS = {EPS_PS:.10f}")

    results = {}

    for gnpa in GNPA_ORDER:
        print(f"\n{'─'*60}")
        print(f"GNPA: {gnpa}")
        print(f"{'─'*60}")

        digits = load_digits(FILES[gnpa])
        print(f"  Cifre citite: {len(digits):,}")

        results[gnpa] = {}

        for net_type, n_needed, mu_t, eps_t, c3, c5 in [
            ('SP', N_SP, MU_SP, EPS_SP, C3_SP, C5_SP),
            ('PS', N_PS, MU_PS, EPS_PS, C3_PS, C5_PS),
        ]:
            lt = simulate_network(digits, net_type, n_needed)

            mean_sim = float(np.mean(lt))
            diff     = abs(mean_sim - mu_t)
            in_eps   = diff <= eps_t

            k_c  = mle_complete(lt)
            k_c3 = mle_censored(lt, c3, net_type)
            k_c5 = mle_censored(lt, c5, net_type)

            results[gnpa][net_type] = {
                'n_sim':     len(lt),
                'mean_sim':  mean_sim,
                'mu_t':      mu_t,
                'eps_t':     eps_t,
                'diff':      diff,
                'in_eps':    in_eps,
                'k_hat_c':   k_c,
                'k_hat_c3':  k_c3,
                'k_hat_c5':  k_c5,
                'mse_c':     mse(k_c),
                'mse_c3':    mse(k_c3),
                'mse_c5':    mse(k_c5),
            }

            ok = "✅" if in_eps else "❌"
            print(f"  {net_type}:  n={len(lt):,}  "
                  f"mean={mean_sim:.10f}  "
                  f"|diff|={diff:.10f} {ok}")
            print(f"       K̂(C)={k_c}  "
                  f"K̂(C3)={k_c3}  "
                  f"K̂(C5)={k_c5}  "
                  f"MSE={mse(k_c):.4f}")

    return results


def print_table_medii(results):
    print("\n" + "=" * 70)
    print("TABELUL 3 — Medii simulate și verificarea TLC")
    print("=" * 70)
    print(f"{'GNPA':<26} {'Ret.':<4} "
          f"{'x̄ simulat':<14} {'E[X] teoretic':<14} "
          f"{'|diff|':<14} {'În ε?'}")
    print("─" * 76)
    for gnpa in GNPA_ORDER:
        for net in ['SP', 'PS']:
            r  = results[gnpa][net]
            ok = "✅ Da" if r['in_eps'] else "❌ Nu"
            print(f"{gnpa:<26} {net:<4} "
                  f"{r['mean_sim']:.10f}  "
                  f"{r['mu_t']:.10f}  "
                  f"{r['diff']:.10f}  {ok}")
    print("─" * 76)


def print_table_mle(results):
    print("\n" + "=" * 70)
    print("TABELUL 4 — Estimatorii MLE K̂  (K adevărat = 9)")
    print("=" * 70)
    print(f"{'GNPA':<26} {'Ret.':<4} {'n':<12} "
          f"{'K̂(C)':<8} {'K̂(C3)':<8} {'K̂(C5)':<8} {'MSE'}")
    print("─" * 70)
    for gnpa in GNPA_ORDER:
        for net in ['SP', 'PS']:
            r = results[gnpa][net]
            print(f"{gnpa:<26} {net:<4} {r['n_sim']:<12,} "
                  f"{r['k_hat_c']:<8} "
                  f"{r['k_hat_c3']:<8} "
                  f"{r['k_hat_c5']:<8} "
                  f"{r['mse_c']:.4f}")
    print("─" * 70)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = run_simulations()
    print_table_medii(results)
    print_table_mle(results)
    print("\nGata!")
