"""
Figure 2 — Hazard rate h(k) for the SP and PS networks.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

def h_sp(k):
    p = pmf_sp(k)
    r = R_sp(k - 1) if k > 0 else 1.0
    return p / r if r > 1e-15 else 0.0

def h_ps(k):
    p = pmf_ps(k)
    r = R_ps(k - 1) if k > 0 else 1.0
    return p / r if r > 1e-15 else 0.0

# ── Data ──────────────────────────────────────────────────────
ks = list(range(K + 1))
Hsp = [h_sp(k) for k in ks]
Hps = [h_ps(k) for k in ks]

hsp5 = h_sp(5)
hps5 = h_ps(5)
diff = hps5 - hsp5

# ── Figure ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.28,
    'grid.linestyle': '--'
})

fig, ax = plt.subplots(figsize=(10, 6))

# Step curves
ax.step(ks, Hsp, where='post',
        color='#1f77b4', lw=2.8, zorder=2)

ax.step(ks, Hps, where='post',
        color='#d62728', lw=2.8, ls='--', zorder=2)

# Markers
ax.scatter(ks, Hsp, s=55, color='#1f77b4', marker='o', zorder=3)
ax.scatter(ks, Hps, s=55, color='#d62728', marker='s', zorder=3)

# Highlight k = 5
ax.scatter([5], [hsp5], s=120, color='#1f77b4', marker='o', zorder=5)
ax.scatter([5], [hps5], s=120, color='#d62728', marker='s', zorder=5)

# Reference line
ax.axvline(x=5, color='gray', ls=':', lw=1.4, alpha=0.35, zorder=1)

# Delta annotation
x_delta = 5.18
ax.annotate(
    '',
    xy=(x_delta, hps5),
    xytext=(x_delta, hsp5),
    arrowprops=dict(arrowstyle='<->', color='#2ca02c', lw=2.0),
    zorder=5
)

ax.text(
    x_delta - 0.28, (hsp5 + hps5) / 2,
    rf'$\Delta = {diff:.4f}$',
    fontsize=11,
    color='#2ca02c',
    ha='right',
    va='center',
    bbox=dict(
        boxstyle='round,pad=0.22',
        facecolor='white',
        edgecolor='#2ca02c',
        alpha=0.95
    ),
    zorder=6
)

# Annotation h_ps(5)
ax.annotate(
    rf'$h_{{PS}}(5) = {hps5:.4f}$',
    xy=(5, hps5),
    xytext=(6.20, 0.50),
    fontsize=11,
    color='#d62728',
    arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.22',
              facecolor='white',
              edgecolor='#d62728',
              alpha=0.95),
    ha='left',
    va='center'
)

# Annotation h_sp(5)
ax.annotate(
    rf'$h_{{SP}}(5) = {hsp5:.4f}$',
    xy=(5, hsp5),
    xytext=(6.20, 0.17),
    fontsize=11,
    color='#1f77b4',
    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.22',
              facecolor='white',
              edgecolor='#1f77b4',
              alpha=0.95),
    ha='left',
    va='center'
)

# Title and axes
ax.set_title(
    "Figure 2. Hazard rate $h(k)$ for the SP and PS networks\n"
    "($N=3$, $M=2$, $K=9$)",
    fontsize=16,
    fontweight='bold'
)

ax.set_xlabel(r'$k$', fontsize=14)
ax.set_ylabel(r'$h(k)$', fontsize=14)

ax.set_xticks(ks)
ax.set_xlim(-0.3, 9.7)
ax.set_ylim(-0.02, 1.10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend_handles = [
    Line2D([0], [0], color='#1f77b4', lw=2.8,
           marker='o', markersize=8,
           linestyle='-',
           label=r'$h_{SP}(k)$ — Serial–Parallel Network (SP)'),

    Line2D([0], [0], color='#d62728', lw=2.8,
           marker='s', markersize=8,
           linestyle='--',
           label=r'$h_{PS}(k)$ — Parallel–Serial Network (PS)')
]

ax.legend(handles=legend_handles,
          fontsize=11,
          loc='upper left',
          framealpha=0.95,
          edgecolor='#cccccc')

ax.grid(True)

plt.tight_layout()
plt.savefig(
    'figure2_hazard.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.close()

print(
    f"figure2_hazard.png saved | "
    f"h_sp(5)={hsp5:.10f} | h_ps(5)={hps5:.10f} | Delta={diff:.10f}"
)