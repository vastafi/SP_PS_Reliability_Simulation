import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------- Parametri ----------------
K = 9
N = 3
M = 2

# ---------------- Functii ----------------
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

# ---------------- Date ----------------
ks = list(range(K + 1))
Hsp = [h_sp(k) for k in ks]
Hps = [h_ps(k) for k in ks]

hsp5 = h_sp(5)
hps5 = h_ps(5)
diff = hps5 - hsp5

# Extindere pentru afisarea corecta a primei trepte
ks_ext = [-1] + ks
Hsp_ext = [Hsp[0]] + Hsp
Hps_ext = [Hps[0]] + Hps

# ---------------- Stil ----------------
plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 0.9,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'figure.dpi': 300
})

fig, ax = plt.subplots(figsize=(8.2, 4.8))

# ---------------- Curbe principale ----------------
ax.step(
    ks_ext, Hsp_ext,
    where='post',
    color='#1f77b4',
    lw=2.0,
    zorder=2
)

ax.step(
    ks_ext, Hps_ext,
    where='post',
    color='#d62728',
    lw=2.0,
    ls='--',
    zorder=2
)

# Markeri discreti
ax.scatter(ks, Hsp, s=28, color='#1f77b4', marker='o', zorder=3)
ax.scatter(ks, Hps, s=28, color='#d62728', marker='s', zorder=3)

# Evidentiere discreta la k = 5
ax.scatter([5], [hsp5], s=52, color='#1f77b4', marker='o', zorder=4)
ax.scatter([5], [hps5], s=52, color='#d62728', marker='s', zorder=4)

# Diferenta Delta pe treapta lui k = 5
x_delta = 5.14
ax.annotate(
    '',
    xy=(x_delta, hps5),
    xytext=(x_delta, hsp5),
    arrowprops=dict(arrowstyle='<->', color='black', lw=1.0),
    zorder=4
)

ax.text(
    x_delta - 0.10,
    (hsp5 + hps5) / 2,
    rf'$\Delta={diff:.4f}$',
    fontsize=10,
    ha='right',
    va='center',
    bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='black', lw=0.8),
    zorder=5
)

# Adnotari compacte
ax.annotate(
    rf'$h_{{ps}}(5)={hps5:.4f}$',
    xy=(5, hps5),
    xytext=(6.05, 0.49),
    fontsize=10,
    color='#d62728',
    arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.0),
    bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='#d62728', lw=0.8),
    ha='left',
    va='center',
    zorder=5
)

ax.annotate(
    rf'$h_{{sp}}(5)={hsp5:.4f}$',
    xy=(5, hsp5),
    xytext=(6.05, 0.16),
    fontsize=10,
    color='#1f77b4',
    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.0),
    bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='#1f77b4', lw=0.8),
    ha='left',
    va='center',
    zorder=5
)

# ---------------- Axe si etichete ----------------
ax.set_xlabel(r'$k$', fontsize=12)
ax.set_ylabel(r'$h(k)$', fontsize=12)

ax.set_xticks(ks)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

ax.set_xlim(-0.30, 9.45)
ax.set_ylim(-0.02, 1.05)

# Axa Y in x=0
ax.spines['left'].set_position(('data', 0))

# Axa X putin sub 0, sa nu taie markerul de la baza
ax.spines['bottom'].set_position(('data', -0.02))

# Fara spine-uri inutile
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# Fara grid pentru paper-ready
ax.grid(False)

# ---------------- Legenda ----------------
legend_handles = [
    Line2D(
        [0], [0],
        color='#1f77b4',
        lw=2.0,
        linestyle='-',
        marker='o',
        markersize=5,
        label=r'$h_{sp}(k)$ — SP'
    ),
    Line2D(
        [0], [0],
        color='#d62728',
        lw=2.0,
        linestyle='--',
        marker='s',
        markersize=5,
        label=r'$h_{ps}(k)$ — PS'
    )
]

ax.legend(
    handles=legend_handles,
    loc='upper left',
    bbox_to_anchor=(0.06, 0.98),
    fontsize=10,
    frameon=True,
    framealpha=1.0,
    edgecolor='black',
    fancybox=False,
    borderpad=0.3,
    handlelength=2.2
)

plt.tight_layout(pad=0.6)
plt.savefig('figura2_hazard_paper_ready.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(
    f"figura2_hazard_paper_ready.png salvata | "
    f"h_sp(5)={hsp5:.4f} | h_ps(5)={hps5:.4f} | Delta={diff:.4f}"
)