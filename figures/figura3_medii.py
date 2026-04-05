"""
Figura 3 — Medii simulate vs valori teoretice cu banda TLC.
Citește date reale din fișierele CSV ale proiectului.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── Parametri ─────────────────────────────────────────────────
K = 9
N = 3
M = 2

BASE    = "/Users/astafivalentina/PycharmProjects/SP_PS_Reliability_Simulation/"
FIGURES = BASE + "figures/"

FILES = {
    "Java ThreadLocalRandom": [
        BASE + "data/java_threadlocal/threadlocal_data1.csv",
        BASE + "data/java_threadlocal/threadlocal_data2.csv",
        BASE + "data/java_threadlocal/threadlocal_data3.csv",
        BASE + "data/java_threadlocal/threadlocal_data4.csv",
        BASE + "data/java_threadlocal/threadlocal_data5.csv",
    ],
    "Python secrets": [
        BASE + "data/python_secrets/secrets_data1.csv",
        BASE + "data/python_secrets/secrets_data2.csv",
        BASE + "data/python_secrets/secrets_data3.csv",
        BASE + "data/python_secrets/secrets_data4.csv",
        BASE + "data/python_secrets/secrets_data5.csv",
    ],
    "Cifrele lui pi": [
        BASE + "data/pi/pi_digits_part.csv",
        BASE + "data/pi/pi_digits_part1.csv",
        BASE + "data/pi/pi_digits_part2.csv",
        BASE + "data/pi/pi_digits_part3.csv",
        BASE + "data/pi/pi_digits_part4.csv",
        BASE + "data/pi/pi_digits_part5.csv",
    ],
}

GNPA_ORDER  = ["Java ThreadLocalRandom", "Python secrets", "Cifrele lui pi"]
GNPA_LABELS = ["Java\nThreadLocalRandom", "Python\nsecrets", "Cifrele\nlui π"]
COLORS      = ['#1f77b4', '#ff7f0e', '#2ca02c']


# ── Funcții model ─────────────────────────────────────────────
def pmf_sp(k):
    pk  = (k + 1) / (K + 1)
    pk0 = k / (K + 1)
    if k == 0:
        return 1 - (1 - pk ** N) ** M
    return (1 - pk0 ** N) ** M - (1 - pk ** N) ** M

def pmf_ps(k):
    qk  = (K - k) / (K + 1)
    qk0 = (K - k + 1) / (K + 1)
    if k == 0:
        return (1 - qk ** N) ** M
    return (1 - qk ** N) ** M - (1 - qk0 ** N) ** M


# ── Valori teoretice și TLC ───────────────────────────────────
mu_sp  = sum(k * pmf_sp(k) for k in range(K + 1))
mu_ps  = sum(k * pmf_ps(k) for k in range(K + 1))
var_sp = sum((k - mu_sp) ** 2 * pmf_sp(k) for k in range(K + 1))
sigma  = math.sqrt(var_sp)
z      = norm.ppf(0.975)
eps_sp = 0.001 * mu_sp
eps_ps = 0.001 * mu_ps
n_sp   = int(math.floor((z * sigma / eps_sp) ** 2)) + 1
n_ps   = int(math.floor((z * sigma / eps_ps) ** 2)) + 1

print(f"E[U] SP  = {mu_sp:.10f}")
print(f"E[V] PS  = {mu_ps:.10f}")
print(f"sigma    = {sigma:.10f}")
print(f"n_SP     = {n_sp:,}   eps_SP = {eps_sp:.10f}")
print(f"n_PS     = {n_ps:,}   eps_PS = {eps_ps:.10f}")


# ── Citire CSV ────────────────────────────────────────────────
def load_digits(file_list):
    digits = []
    for fpath in file_list:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Fișier negăsit: {fpath}")
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for part in line.replace(',', ' ').split():
                    if part.isdigit():
                        digits.append(int(part))
    return np.array(digits, dtype=np.int32)


# ── Simulare ─────────────────────────────────────────────────
def simulate(digits, net_type, n_samples):
    cells = N * M
    n_use = min(n_samples, len(digits) // cells)
    data  = digits[:n_use * cells].reshape(n_use, M, N)
    if net_type == 'SP':
        return np.min(np.max(data, axis=2), axis=1)
    else:
        return np.max(np.min(data, axis=2), axis=1)


# ── Calcul medii simulate ─────────────────────────────────────
means_sp = {}
means_ps = {}

for gnpa in GNPA_ORDER:
    print(f"\nGNPA: {gnpa}")
    digits = load_digits(FILES[gnpa])
    print(f"  Cifre disponibile: {len(digits):,}")

    lt_sp = simulate(digits, 'SP', n_sp)
    lt_ps = simulate(digits, 'PS', n_ps)

    means_sp[gnpa] = float(np.mean(lt_sp))
    means_ps[gnpa] = float(np.mean(lt_ps))

    ok_sp = "✅" if abs(means_sp[gnpa] - mu_sp) <= eps_sp else "❌"
    ok_ps = "✅" if abs(means_ps[gnpa] - mu_ps) <= eps_ps else "❌"
    print(f"  SP: {means_sp[gnpa]:.10f}  "
          f"|diff|={abs(means_sp[gnpa]-mu_sp):.10f} {ok_sp}")
    print(f"  PS: {means_ps[gnpa]:.10f}  "
          f"|diff|={abs(means_ps[gnpa]-mu_ps):.10f} {ok_ps}")

SP_SIM = [means_sp[g] for g in GNPA_ORDER]
PS_SIM = [means_ps[g] for g in GNPA_ORDER]


# ── Figură ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 12, 'axes.linewidth': 0.8,
    'grid.alpha': 0.3, 'grid.linestyle': '--',
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Figura 3. Mediile simulate comparate cu valorile teoretice "
    "și limitele erorii TLC\n"
    "pentru cele 3 GNPA-uri ($N=3$, $M=2$, $K=9$, "
    "$\\varepsilon_{rel}=0.1\\%$)",
    fontsize=12, fontweight='bold', y=1.02,
)

x = np.arange(len(GNPA_ORDER))

for ax, sim_vals, mu_t, eps_t, title, unit, ypad in [
    (axes[0], SP_SIM, mu_sp, eps_sp, "Rețeaua SP", "E[U]", 0.0006),
    (axes[1], PS_SIM, mu_ps, eps_ps, "Rețeaua PS", "E[V]", 0.0003),
]:
    bars = ax.bar(x, sim_vals, color=COLORS, alpha=0.80,
                  width=0.5, edgecolor='black', linewidth=0.7, zorder=3)

    ax.axhline(y=mu_t, color='red', lw=2.0, ls='-', zorder=4,
               label=f'Val. teoretică {unit}={mu_t:.4f}')
    ax.axhline(y=mu_t + eps_t, color='red', lw=1.0,
               ls='--', alpha=0.5, zorder=3)
    ax.axhline(y=mu_t - eps_t, color='red', lw=1.0,
               ls='--', alpha=0.5, zorder=3,
               label=f'Limita TLC $\\pm\\varepsilon={eps_t:.4f}$')
    ax.fill_between([-0.5, len(GNPA_ORDER) - 0.5],
                    [mu_t - eps_t] * 2, [mu_t + eps_t] * 2,
                    alpha=0.07, color='red', zorder=2)

    for bar, val in zip(bars, sim_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + ypad, f'{val:.6f}',
                ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(GNPA_LABELS, fontsize=10)
    ax.set_ylabel('Media simulată', fontsize=11)
    ax.legend(fontsize=9, loc='upper right',
              framealpha=0.92, edgecolor='#cccccc')
    ax.grid(True, alpha=0.3, axis='y', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(min(sim_vals) - 7 * eps_t,
                max(sim_vals) + 8 * eps_t)

plt.tight_layout()
out = FIGURES + "figura3_medii.png"
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSalvat: {out}")