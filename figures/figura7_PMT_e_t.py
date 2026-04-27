"""
Figure 7 — Empirical vs theoretical PMF for the SP and PS networks
for the three PRNG sources.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────
K = 9
N = 3
M = 2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)

FILES = {
    "Java ThreadLocalRandom": [
        os.path.join(BASE, "data", "java_threadlocal", "threadlocal_data1.csv"),
        os.path.join(BASE, "data", "java_threadlocal", "threadlocal_data2.csv"),
        os.path.join(BASE, "data", "java_threadlocal", "threadlocal_data3.csv"),
        os.path.join(BASE, "data", "java_threadlocal", "threadlocal_data4.csv"),
        os.path.join(BASE, "data", "java_threadlocal", "threadlocal_data5.csv"),
    ],
    "Python secrets": [
        os.path.join(BASE, "data", "python_secrets", "secrets_data1.csv"),
        os.path.join(BASE, "data", "python_secrets", "secrets_data2.csv"),
        os.path.join(BASE, "data", "python_secrets", "secrets_data3.csv"),
        os.path.join(BASE, "data", "python_secrets", "secrets_data4.csv"),
        os.path.join(BASE, "data", "python_secrets", "secrets_data5.csv"),
    ],
    "Digits of π": [
        os.path.join(BASE, "data", "pi", "pi_digits_part.csv"),
        os.path.join(BASE, "data", "pi", "pi_digits_part1.csv"),
        os.path.join(BASE, "data", "pi", "pi_digits_part2.csv"),
        os.path.join(BASE, "data", "pi", "pi_digits_part3.csv"),
        os.path.join(BASE, "data", "pi", "pi_digits_part4.csv"),
        os.path.join(BASE, "data", "pi", "pi_digits_part5.csv"),
    ],
}

LABEL_COLORS = {
    "Java ThreadLocalRandom": "#1f77b4",
    "Python secrets": "#ff7f0e",
    "Digits of π": "#2ca02c",
}

LABEL_MARKERS = {
    "Java ThreadLocalRandom": "o",
    "Python secrets": "s",
    "Digits of π": "^",
}


# ── Theoretical PMF ───────────────────────────────────────────
def pmf_sp(k):
    pk = (k + 1) / (K + 1)
    pk0 = k / (K + 1)
    if k == 0:
        return 1 - (1 - pk ** N) ** M
    return (1 - pk0 ** N) ** M - (1 - pk ** N) ** M


def pmf_ps(k):
    qk = (K - k) / (K + 1)
    qk0 = (K - k + 1) / (K + 1)
    if k == 0:
        return (1 - qk ** N) ** M
    return (1 - qk ** N) ** M - (1 - qk0 ** N) ** M


ks = np.arange(K + 1)
pmf_sp_theoretical = np.array([pmf_sp(k) for k in ks], dtype=float)
pmf_ps_theoretical = np.array([pmf_ps(k) for k in ks], dtype=float)


# ── CSV loading ───────────────────────────────────────────────
def load_digits(file_list):
    digits = []
    for fpath in file_list:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for part in line.replace(',', ' ').split():
                    if part.isdigit():
                        val = int(part)
                        if 0 <= val <= 9:
                            digits.append(val)
    return np.array(digits, dtype=np.int32)


# ── Simulation from digits ────────────────────────────────────
def simulate_network(digits, net_type):
    cells = N * M
    n_use = len(digits) // cells
    if n_use == 0:
        raise ValueError("Not enough digits to build at least one simulated sample.")

    data = digits[:n_use * cells].reshape(n_use, M, N)

    if net_type == 'SP':
        return np.min(np.max(data, axis=2), axis=1)
    elif net_type == 'PS':
        return np.max(np.min(data, axis=2), axis=1)
    else:
        raise ValueError("net_type must be 'SP' or 'PS'")


# ── Empirical PMF ─────────────────────────────────────────────
def empirical_pmf(values):
    counts = np.bincount(values, minlength=K + 1)
    return counts / len(values)


# ── Plot ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

fig.suptitle(
    "Figure 7. Empirical vs theoretical PMF for the SP and PS networks\n"
    "for the three PRNG sources ($N=3$, $M=2$, $K=9$)",
    fontsize=12,
    fontweight='bold',
    y=1.02
)

plot_specs = [
    (axes[0], 'SP', pmf_sp_theoretical, "Serial–Parallel Network (SP)", "$P(U=k)$"),
    (axes[1], 'PS', pmf_ps_theoretical, "Parallel–Serial Network (PS)", "$P(V=k)$")
]

for ax, net_type, pmf_theoretical, title, prob_label in plot_specs:
    # Theoretical PMF as background bars
    ax.bar(
        ks,
        pmf_theoretical,
        width=0.62,
        color='lightgray',
        edgecolor='gray',
        linewidth=0.8,
        alpha=0.9,
        label=f'Theoretical {prob_label}',
        zorder=2
    )

    # Empirical PMFs
    for source_name, file_list in FILES.items():
        digits = load_digits(file_list)
        sim_values = simulate_network(digits, net_type)
        pmf_emp = empirical_pmf(sim_values)

        ax.plot(
            ks,
            pmf_emp,
            linestyle='--',
            linewidth=1.7,
            marker=LABEL_MARKERS[source_name],
            markersize=5,
            color=LABEL_COLORS[source_name],
            label=f'{source_name} empirical',
            zorder=4
        )

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('$k$', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xticks(ks)
    ax.set_ylim(0, max(np.max(pmf_sp_theoretical), np.max(pmf_ps_theoretical)) * 1.15)
    ax.grid(True, zorder=0)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.92, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(
    'figure7_PMF_e_vs_t.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()

print("figure7_PMF_e_vs_t.png saved")