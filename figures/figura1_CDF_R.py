"""
Figure 1 — Distribution functions F(k), reliability functions R(k),
and point mass functions P(X=k) for the SP and PS networks.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────
K = 9
N = 3
M = 2

# ── Model functions ───────────────────────────────────────────
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

def F_sp(k):
    return 1 - (1 - ((k + 1) / (K + 1))**N)**M

def F_ps(k):
    return (1 - ((K - k) / (K + 1))**N)**M

def R_sp(k):
    return 1 - F_sp(k)

def R_ps(k):
    return 1 - F_ps(k)

# ── Data ──────────────────────────────────────────────────────
ks = list(range(K + 1))

Fsp = [F_sp(k) for k in ks]
Rsp = [R_sp(k) for k in ks]
Psp = [pmf_sp(k) for k in ks]

Fps = [F_ps(k) for k in ks]
Rps = [R_ps(k) for k in ks]
Pps = [pmf_ps(k) for k in ks]

mu_sp = sum(k * pmf_sp(k) for k in ks)
mu_ps = sum(k * pmf_ps(k) for k in ks)

# ── Figure ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Figure 1. Distribution functions $F(k)$, reliability functions $R(k)$, and point mass functions $P(X=k)$\n"
    "for the SP and PS networks ($N=3$, $M=2$, $K=9$)",
    fontsize=12,
    fontweight='bold',
    y=1.02
)

for ax, Fv, Rv, Pv, title, lF, lR, lP, mu_t, cF, cR, cP, mu_lbl in [
    (
        axes[0], Fsp, Rsp, Psp,
        "Serial–Parallel Network (SP)",
        '$F_{SP}(k)$', '$R_{SP}(k)$', '$P(U=k)$',
        mu_sp, '#1f77b4', '#ff7f0e', '#2ca02c', '$E[U]$'
    ),
    (
        axes[1], Fps, Rps, Pps,
        "Parallel–Serial Network (PS)",
        '$F_{PS}(k)$', '$R_{PS}(k)$', '$P(V=k)$',
        mu_ps, '#d62728', '#9467bd', '#8c564b', '$E[V]$'
    ),
]:
    ax.bar(
        ks, Pv, alpha=0.22, color=cP, width=0.55,
        label=f'{lP} - PMF', zorder=2
    )
    ax.step(
        ks, Fv, where='post', color=cF, lw=2.2,
        marker='o', ms=5, label=f'{lF} - CDF', zorder=4
    )
    ax.step(
        ks, Rv, where='post', color=cR, lw=2.2,
        marker='s', ms=5, ls='--', label=f'{lR} - Reliability', zorder=4
    )
    ax.axvline(x=mu_t, color='gray', ls=':', lw=1.2, alpha=0.6)

    ax.text(
        0.87, 0.45, f'{mu_lbl}={mu_t:.4f}',
        transform=ax.transAxes,
        fontsize=9,
        color='gray',
        ha='right',
        va='center'
    )

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('$k$', fontsize=12)
    ax.set_ylabel('Function value', fontsize=11)
    ax.set_xticks(ks)
    ax.set_ylim(-0.02, 1.08)

    # Legend in the upper-left corner — does not cover the data
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.92, edgecolor='#cccccc')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figure1_CDF_R.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("figure1_CDF_R.png saved")