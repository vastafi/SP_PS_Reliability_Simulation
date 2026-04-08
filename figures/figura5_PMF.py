"""
Figure 5 — Point mass functions P(X=k) for the SP and PS networks.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────
K = 9
N = 3
M = 2

# ── PMF functions ─────────────────────────────────────────────
def pmf_sp(k):
    pk = (k + 1) / (K + 1)
    pk0 = k / (K + 1)
    if k == 0:
        return 1 - (1 - pk**N)**M
    return (1 - pk0**N)**M - (1 - pk**N)**M

def pmf_ps(k):
    qk = (K - k) / (K + 1)
    qk0 = (K - k + 1) / (K + 1)
    if k == 0:
        return (1 - qk**N)**M
    return (1 - qk**N)**M - (1 - qk0**N)**M

# ── Data ──────────────────────────────────────────────────────
ks = list(range(K + 1))
Psp = [pmf_sp(k) for k in ks]
Pps = [pmf_ps(k) for k in ks]

# ── Figure ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fig.suptitle(
    "Figure 5. Point mass functions $P(X=k)$ for the SP and PS networks\n"
    "($N=3$, $M=2$, $K=9$)",
    fontsize=12,
    fontweight='bold',
    y=1.02
)

for ax, Pv, title, color, label in [
    (axes[0], Psp, "Serial–Parallel Network (SP)", '#66bb6a', '$P(X=k)$'),
    (axes[1], Pps, "Parallel–Serial Network (PS)", '#bcaaa4', '$P(X=k)$'),
]:
    ax.bar(
        ks,
        Pv,
        alpha=0.75,
        color=color,
        width=0.6,
        label=f'{label} - PMF',
        zorder=3
    )

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('$k$', fontsize=12)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_xticks(ks)
    ax.set_ylim(0, max(Psp + Pps) * 1.15)

    ax.legend(
        fontsize=9.5,
        loc='upper right',
        framealpha=0.92,
        edgecolor='#cccccc'
    )

    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(
    'figure5_PMF.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.close()

print("figure5_PMF.png saved")