"""
Figura 4 — Schema rețelelor Serial-Paralelă (SP) și Paralel-Serială (PS).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

CLR_UNIT='#AED6F1'; CLR_EDGE='#1A5276'; CLR_LINE='#2C3E50'; CLR_LBL='#1A5276'

def draw_unit(ax, x, y, label, w=1.0, h=0.55):
    rect=FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle="round,pad=0.06",facecolor=CLR_UNIT,
        edgecolor=CLR_EDGE,linewidth=1.8,zorder=5)
    ax.add_patch(rect)
    ax.text(x,y,label,ha='center',va='center',fontsize=10,
            fontweight='bold',color=CLR_EDGE,zorder=6)

def wire(ax,x1,y1,x2,y2):
    ax.plot([x1,x2],[y1,y2],color=CLR_LINE,lw=1.8,solid_capstyle='round',zorder=3)

def junction(ax,x,y,r=0.05):
    ax.add_patch(plt.Circle((x,y),r,color=CLR_LINE,zorder=4))

def lbl(ax,x,y,text,ha='center',fontsize=9,style='italic',color=CLR_LBL):
    ax.text(x,y,text,ha=ha,va='center',fontsize=fontsize,
            color=color,style=style,fontweight='bold')

plt.rcParams.update({'font.size':11,'axes.linewidth':0})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Figura 4. Schema rețelelor Serial-Paralelă (SP) și Paralel-Serială (PS)\n"
    "($N=3$ unități per subrețea, $M=2$ subretele)",
    fontsize=12, fontweight='bold', y=1.03)

# ═══ PANOUL STANGA — SP ═════════════════════════════════════
ax=axes[0]
ax.set_xlim(-1.0, 9.5); ax.set_ylim(-0.8, 5.8)
ax.axis('off')
ax.set_title("Rețeaua Serial-Paralelă (SP)\n"
             "$U = \\min(\\max(X_{1j}),\\ \\max(X_{2j}))$",
             fontsize=10.5, fontweight='bold', pad=12)

# IN / OUT
ax.text(0.0,2.5,'IN',ha='center',va='center',fontsize=11,fontweight='bold',color=CLR_LINE,
        bbox=dict(boxstyle='round,pad=0.3',facecolor='#EBF5FB',edgecolor=CLR_EDGE,lw=1.5))
ax.text(9.2,2.5,'OUT',ha='center',va='center',fontsize=11,fontweight='bold',color=CLR_LINE,
        bbox=dict(boxstyle='round,pad=0.3',facecolor='#EBF5FB',edgecolor=CLR_EDGE,lw=1.5))

# Fir intrare
wire(ax,0.4,2.5,1.2,2.5); junction(ax,1.2,2.5)
wire(ax,1.2,2.5,1.2,4.0); wire(ax,1.2,2.5,1.2,1.0)

# Subreteaua 1
for i,y in enumerate([4.0,2.5,1.0]):
    wire(ax,1.2,y,1.7,y)
    draw_unit(ax,2.2,y,f'$X_{{1{i+1}}}$')
    wire(ax,2.7,y,3.5,y)
wire(ax,3.5,4.0,3.5,1.0)
[junction(ax,3.5,y) for y in [4.0,2.5,1.0]]
wire(ax,3.5,2.5,4.3,2.5)

# Eticheta subreteaua 1 — deasupra, spatiu suficient
lbl(ax,2.2,5.2,'Subrețeaua 1',fontsize=10)
lbl(ax,2.2,4.75,'N=3 unități în paralel',fontsize=8.5)
ax.annotate('',xy=(1.2,4.55),xytext=(3.5,4.55),
            arrowprops=dict(arrowstyle='<->',color=CLR_LBL,lw=1.0))

# Fir legatura
wire(ax,4.3,2.5,5.0,2.5); junction(ax,5.0,2.5)
wire(ax,5.0,2.5,5.0,4.0); wire(ax,5.0,2.5,5.0,1.0)

# Subreteaua 2
for i,y in enumerate([4.0,2.5,1.0]):
    wire(ax,5.0,y,5.5,y)
    draw_unit(ax,6.0,y,f'$X_{{2{i+1}}}$')
    wire(ax,6.5,y,7.3,y)
wire(ax,7.3,4.0,7.3,1.0)
[junction(ax,7.3,y) for y in [4.0,2.5,1.0]]
wire(ax,7.3,2.5,8.0,2.5)

# Eticheta subreteaua 2
lbl(ax,6.0,5.2,'Subrețeaua 2',fontsize=10)
lbl(ax,6.0,4.75,'N=3 unități în paralel',fontsize=8.5)
ax.annotate('',xy=(5.0,4.55),xytext=(7.3,4.55),
            arrowprops=dict(arrowstyle='<->',color=CLR_LBL,lw=1.0))

wire(ax,8.0,2.5,8.8,2.5)

# Eticheta M in serie — jos
ax.annotate('',xy=(1.2,-0.55),xytext=(8.0,-0.55),
            arrowprops=dict(arrowstyle='<->',color='#717D7E',lw=1.0))
lbl(ax,4.6,-0.32,'M=2 subretele în serie',fontsize=9,
    style='italic',color='#717D7E')

# ═══ PANOUL DREAPTA — PS ════════════════════════════════════
ax2=axes[1]
ax2.set_xlim(-1.5, 9.5); ax2.set_ylim(-0.8, 5.8)
ax2.axis('off')
ax2.set_title("Rețeaua Paralel-Serială (PS)\n"
              "$V = \\max(\\min(X_{1j}),\\ \\min(X_{2j}))$",
              fontsize=10.5, fontweight='bold', pad=12)

# IN / OUT
ax2.text(0.0,2.5,'IN',ha='center',va='center',fontsize=11,fontweight='bold',color=CLR_LINE,
         bbox=dict(boxstyle='round,pad=0.3',facecolor='#EBF5FB',edgecolor=CLR_EDGE,lw=1.5))
ax2.text(9.2,2.5,'OUT',ha='center',va='center',fontsize=11,fontweight='bold',color=CLR_LINE,
         bbox=dict(boxstyle='round,pad=0.3',facecolor='#EBF5FB',edgecolor=CLR_EDGE,lw=1.5))

wire(ax2,0.4,2.5,0.9,2.5); junction(ax2,0.9,2.5)

# Subreteaua 1 (sus) — 3 unitati in serie
y1=3.9
wire(ax2,0.9,2.5,0.9,y1); wire(ax2,0.9,y1,1.5,y1)
xpos=[2.1,4.1,6.1]
for i,xp in enumerate(xpos):
    draw_unit(ax2,xp,y1,f'$X_{{1{i+1}}}$')
    if i<2: wire(ax2,xp+0.5,y1,xpos[i+1]-0.5,y1)
wire(ax2,6.6,y1,7.3,y1); wire(ax2,7.3,y1,7.3,2.5); junction(ax2,7.3,2.5)

# Eticheta subreteaua 1 — deasupra
lbl(ax2,4.1,5.2,'Subrețeaua 1',fontsize=10)
lbl(ax2,4.1,4.75,'N=3 unități în serie',fontsize=8.5)
ax2.annotate('',xy=(1.5,4.55),xytext=(6.6,4.55),
             arrowprops=dict(arrowstyle='<->',color=CLR_LBL,lw=1.0))

# Subreteaua 2 (jos) — 3 unitati in serie
y2=1.1
wire(ax2,0.9,2.5,0.9,y2); wire(ax2,0.9,y2,1.5,y2)
for i,xp in enumerate(xpos):
    draw_unit(ax2,xp,y2,f'$X_{{2{i+1}}}$')
    if i<2: wire(ax2,xp+0.5,y2,xpos[i+1]-0.5,y2)
wire(ax2,6.6,y2,7.3,y2); wire(ax2,7.3,y2,7.3,2.5)
wire(ax2,7.3,2.5,8.0,2.5); wire(ax2,8.0,2.5,8.8,2.5)

# Eticheta subreteaua 2 — dedesubt, spatiu suficient
lbl(ax2,4.1,0.45,'Subrețeaua 2',fontsize=10)
lbl(ax2,4.1,0.05,'N=3 unități în serie',fontsize=8.5)
ax2.annotate('',xy=(1.5,-0.2),xytext=(6.6,-0.2),
             arrowprops=dict(arrowstyle='<->',color=CLR_LBL,lw=1.0))

# Eticheta M in paralel — stanga, cu sageata verticala
# ax2.annotate('',xy=(0.0004,y2+0.05),xytext=(0.0004,y1-0.05),
#              arrowprops=dict(arrowstyle='<->',color='#717D7E',lw=1.0))
ax2.annotate('', xy=(-0.5, y2+0.05), xytext=(-0.5, y1-0.05),
             arrowprops=dict(arrowstyle='<->', color='#717D7E', lw=1.0))
ax2.text(-1.3,3.5,'M=2\nsubretele\nîn paralel',ha='center',va='center',
         fontsize=8.5,color='#717D7E', fontweight='bold', style='italic')

plt.tight_layout()
plt.savefig('figura4_scheme.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("figura4_scheme.png salvata")
