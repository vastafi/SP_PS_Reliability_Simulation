import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, math
from scipy.stats import norm

K=9; N=3; M=2

def pmf_sp(k):
    pk=(k+1)/(K+1); pk0=k/(K+1)
    if k==0: return 1-(1-pk**N)**M
    return (1-pk0**N)**M-(1-pk**N)**M

def pmf_ps(k):
    qk=(K-k)/(K+1); qk0=(K-k+1)/(K+1)
    if k==0: return (1-qk**N)**M
    return (1-qk**N)**M-(1-qk0**N)**M

mu_sp=sum(k*pmf_sp(k) for k in range(K+1))
mu_ps=sum(k*pmf_ps(k) for k in range(K+1))
var_sp=sum((k-mu_sp)**2*pmf_sp(k) for k in range(K+1))
sig=math.sqrt(var_sp)
z=norm.ppf(0.975)
eps_SP=0.001*mu_sp; eps_PS=0.001*mu_ps

GNPAS=["Java\nThreadLocalRandom","Python\nsecrets","Cifrele\nlui \u03C0"]
SP_SIM=[5.9270523902,5.9277858871,5.9317750382]
PS_SIM=[3.0710961178,3.0712825535,3.0712256415]
COLORS=['#1f77b4','#ff7f0e','#2ca02c']

plt.rcParams.update({'font.size':12,'axes.linewidth':0.8,'grid.alpha':0.3,'grid.linestyle':'--'})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Figura 3. Mediile simulate comparate cu valorile teoretice și limitele erorii TLC\n"
    "pentru cele 3 GNPA-uri ($N=3$, $M=2$, $K=9$, $\\varepsilon_{rel}=0.1\\%$)",
    fontsize=12, fontweight='bold', y=1.02)

x=np.arange(3)
for ax, sim_means, mu_t, eps_t, title, unit, ypad in [
    (axes[0], SP_SIM, mu_sp, eps_SP, "Rețeaua SP", "E[U]", 0.0006),
    (axes[1], PS_SIM, mu_ps, eps_PS, "Rețeaua PS", "E[V]", 0.0003),
]:
    bars=ax.bar(x, sim_means, color=COLORS, alpha=0.80,
                width=0.5, edgecolor='black', linewidth=0.7, zorder=3)

    # Linie teoretica
    ax.axhline(y=mu_t, color='red', lw=2.0, ls='-', zorder=4,
               label=f'Val. teoretică {unit}={mu_t:.4f}')
    # Banda TLC
    ax.axhline(y=mu_t+eps_t, color='red', lw=1.0, ls='--', alpha=0.5, zorder=3)
    ax.axhline(y=mu_t-eps_t, color='red', lw=1.0, ls='--', alpha=0.5, zorder=3,
               label=f'Limita TLC $\\pm\\varepsilon={eps_t:.4f}$')
    ax.fill_between([-0.5,2.5],[mu_t-eps_t]*2,[mu_t+eps_t]*2,
                    alpha=0.07, color='red', zorder=2)

    # Valori deasupra barelor
    for bar, val in zip(bars, sim_means):
        ax.text(bar.get_x()+bar.get_width()/2, val+ypad,
                f'{val:.6f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(GNPAS, fontsize=10)
    ax.set_ylabel('Media simulată', fontsize=11)
    # Legenda sus dreapta
    ax.legend(fontsize=9, loc='upper right', framealpha=0.92, edgecolor='#cccccc')
    ax.grid(True, alpha=0.3, axis='y', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ymin=min(sim_means)-7*eps_t; ymax=max(sim_means)+8*eps_t
    ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig('figura3_medii.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("figura3_medii.png salvata")
