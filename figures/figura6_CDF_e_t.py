"""
Figure 6 — Empirical vs theoretical CDF for the SP and PS networks
for the three PRNG sources, with deviation panels.
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

# ── Theoretical PMF/CDF ───────────────────────────────────────
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
cdf_sp_theoretical = np.cumsum([pmf_sp(k) for k in ks])
cdf_ps_theoretical = np.cumsum([pmf_ps(k) for k in ks])

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

# ── Empirical CDF ─────────────────────────────────────────────
def empirical_cdf(values):
    counts = np.bincount(values, minlength=K + 1)
    return np.cumsum(counts) / len(values)

# ── Plot ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 8),
    gridspec_kw={'height_ratios': [3, 1]}
)

fig.suptitle(
    "Figure 6. Empirical vs theoretical CDF for the SP and PS networks\n"
    "for the three PRNG sources ($N=3$, $M=2$, $K=9$)",
    fontsize=12,
    fontweight='bold',
    y=0.98
)

plot_specs = [
    (axes[0, 0], axes[1, 0], 'SP', cdf_sp_theoretical, "Serial–Parallel Network (SP)"),
    (axes[0, 1], axes[1, 1], 'PS', cdf_ps_theoretical, "Parallel–Serial Network (PS)")
]

for ax_main, ax_diff, net_type, cdf_theoretical, title in plot_specs:
    ax_main.step(
        ks, cdf_theoretical,
        where='post',
        color='black',
        linewidth=2.4,
        label='Theoretical CDF',
        zorder=6
    )

    max_abs_diff = 0.0

    for source_name, file_list in FILES.items():
        digits = load_digits(file_list)
        sim_values = simulate_network(digits, net_type)
        cdf_emp = empirical_cdf(sim_values)
        diff = cdf_emp - cdf_theoretical

        max_abs_diff = max(max_abs_diff, np.max(np.abs(diff)))

        # main panel
        ax_main.step(
            ks, cdf_emp,
            where='post',
            linewidth=1.6,
            linestyle='--',
            color=LABEL_COLORS[source_name],
            label=f'{source_name} empirical',
            zorder=4
        )

        ax_main.plot(
            ks, cdf_emp,
            marker='o',
            linestyle='None',
            markersize=4,
            color=LABEL_COLORS[source_name],
            zorder=5
        )

        # deviation panel
        ax_diff.step(
            ks, diff,
            where='post',
            linewidth=1.6,
            linestyle='--',
            color=LABEL_COLORS[source_name],
            zorder=3
        )

        ax_diff.plot(
            ks, diff,
            marker='o',
            linestyle='None',
            markersize=4,
            color=LABEL_COLORS[source_name],
            zorder=4
        )

    ax_main.set_title(title, fontsize=11, fontweight='bold')
    ax_main.set_ylabel('$F(k)$', fontsize=12)
    ax_main.set_xticks(ks)
    ax_main.set_ylim(-0.02, 1.02)
    ax_main.grid(True, zorder=0)
    ax_main.legend(fontsize=8.5, loc='lower right', framealpha=0.92, edgecolor='#cccccc')
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    ax_diff.axhline(0, color='black', linewidth=1.0)
    lim = max(1e-4, 1.15 * max_abs_diff)
    ax_diff.set_ylim(-lim, lim)
    ax_diff.set_xticks(ks)
    ax_diff.set_xlabel('$k$', fontsize=12)
    ax_diff.set_ylabel('$\\Delta F$', fontsize=11)
    ax_diff.grid(True, zorder=0)
    ax_diff.spines['top'].set_visible(False)
    ax_diff.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(
    'figure6_CDF_e_vs_t.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()

print("figure6_CDF_e_vs_t.png saved")