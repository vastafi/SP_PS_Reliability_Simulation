"""
Figure 7 (NOU) — Percentage of exact estimates K_hat = 9 for generators with controlled defects
(n = 33, R = 5000, complete data): (a) non-uniform marginal, P(digit 9) = 0.1 - delta;
(b) serial dependence, each digit repeats the previous one with probability rho.
Datele provin din run_experiments.py (results/results_v2.json, cheia E5).
Acelasi stil ca figurile existente.
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(OUT_DIR)
E5 = json.load(open(os.path.join(BASE, 'results', 'results_v2.json'), encoding='utf-8'))['E5']

STYLES = {  # culori SP/PS ca in figura hazard; linie plina = inversa CDF, punctata = structurala
    'inverse|SP':    ('#1f77b4', '-',  'o', 'SP, inverse CDF'),
    'structural|SP': ('#1f77b4', '--', 's', 'SP, structural'),
    'inverse|PS':    ('#d62728', '-',  'o', 'PS, inverse CDF'),
    'structural|PS': ('#d62728', '--', 's', 'PS, structural'),
}

plt.rcParams.update({'font.size': 12, 'axes.linewidth': 0.8,
                     'grid.alpha': 0.3, 'grid.linestyle': '--'})
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), sharey=True)

for ax, fam, title, xlab in [
        (axes[0], 'biased', '(a) Non-uniform marginal', 'Deficit $\\delta$ of $P$(digit 9)'),
        (axes[1], 'sticky', '(b) Serially dependent digits', 'Serial dependence $\\rho$')]:
    levels = sorted(E5[fam], key=float)
    x = [float(v) for v in levels]
    for key, (c, ls, mk, lab) in STYLES.items():
        ax.plot(x, [E5[fam][lv][key]['exact'] for lv in levels], color=c, ls=ls, lw=2.0,
                marker=mk, ms=6, label=lab, zorder=3)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_xticks(x)
    ax.grid(True, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Exact estimates $\\hat{K}=9$, %', fontsize=11)
axes[0].set_ylim(0, 100)
axes[0].legend(fontsize=8.5, loc='lower left', framealpha=0.92, edgecolor='#cccccc')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Figure7_defects.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print('Figure7_defects.png saved')
