"""
Figure 6 (NOU) — Percentage of exact estimates K_hat = 9 versus the sample size n
(complete data, inverse CDF method, R = 1000), with the reference benchmark and its 95% band.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(OUT_DIR)
res = json.load(open(os.path.join(BASE, 'results', 'results_v2.json'), encoding='utf-8'))
E4n = res['E4']['n']
ns = sorted(int(x) for x in E4n)

GNPA = ["Java ThreadLocalRandom", "Python secrets", "Digits of π"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
MARKERS = ['o', 's', '^']

plt.rcParams.update({'font.size': 12, 'axes.linewidth': 0.8,
                     'grid.alpha': 0.3, 'grid.linestyle': '--'})
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), sharey=True)

for ax, net, title in [(axes[0], 'SP', "Serial–Parallel Network (SP)"),
                       (axes[1], 'PS', "Parallel–Serial Network (PS)")]:
    b = np.array([E4n[str(m)][net]['benchmark']['exact'] for m in ns]) / 100
    half = 1.96 * np.sqrt(b * (1 - b) / 1000)
    ax.fill_between(ns, 100 * (b - half), 100 * np.minimum(b + half, 1), color='gray', alpha=0.18,
                    label='Benchmark ± 95% band', zorder=1)
    ax.plot(ns, 100 * b, color='black', lw=2.0, zorder=2)
    for j, (g, c, mk) in enumerate(zip(GNPA, COLORS, MARKERS)):
        ax.plot(np.array(ns) * (1 + (j - 1) * 0.03),
                [E4n[str(m)][net][g]['exact'] for m in ns],
                marker=mk, ms=6, ls='none', color=c, label=g, zorder=4)
    ax.axvline(x=33, color='gray', ls=':', lw=1.2, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(m) for m in ns])
    ax.minorticks_off()
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Sample size $n$', fontsize=12)
    ax.grid(True, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Exact estimates $\\hat{K}=9$, %', fontsize=11)
axes[0].set_ylim(0, 103)
axes[0].legend(fontsize=8.5, loc='lower right', framealpha=0.92, edgecolor='#cccccc')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Figure6_sample_size.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print('Figure6_sample_size.png saved')
