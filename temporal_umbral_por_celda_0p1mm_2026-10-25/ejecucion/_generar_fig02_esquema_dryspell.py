"""Figura 2: esquema de la definicion operativa de dia humedo/seco y dry spell."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper_figuras' / 'fig_esquema_dryspell.png'

R = 0.1
PR = np.array([3.2, 0.3, 0.0, 0.2, 0.0, 0.1, 0.0, 2.4, 0.4, 0.0, 5.1, 1.6])
WET_COLOR, DRY_COLOR, SPELL_COLOR = '#2B6CB0', '#C05621', '#9B2C2C'


def spells(pr, thr):
    dry = np.r_[False, pr < thr, False].astype(np.int8)
    d = np.diff(dry)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def main():
    days = np.arange(1, PR.size + 1)
    wet = PR >= R
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(days[wet], PR[wet], width=0.8, color=WET_COLOR)
    ax.bar(days[~wet], PR[~wet], width=0.8, color=DRY_COLOR)
    zero = ~wet & (PR == 0)
    ax.plot(days[zero], np.zeros(zero.sum()), 'o', color=DRY_COLOR, ms=6, zorder=3)
    ax.axhline(R, color='k', ls='--', lw=1)
    ax.text(PR.size + 0.65, R, '$R$', va='center', fontsize=12)

    for s, e in spells(PR, R):
        t = e - s
        x0, x1 = days[s] - 0.4, days[e - 1] + 0.4
        ax.annotate('', xy=(x0, -0.75), xytext=(x1, -0.75),
                    arrowprops=dict(arrowstyle='<->', color=SPELL_COLOR, lw=1.6))
        label = f'dry spell, $t = {t}$ días' if t > 2 else f'$t = {t}$ días'
        ax.text(0.5 * (x0 + x1), -1.3, label, ha='center', va='center', color=SPELL_COLOR, fontsize=11)

    ax.legend(handles=[
        Patch(color=WET_COLOR, label='día húmedo ($pr \\geq R$)'),
        Patch(color=DRY_COLOR, label='día seco ($pr < R$)'),
        Line2D([0], [0], marker='o', ls='none', color=DRY_COLOR, label='día seco con 0 mm'),
    ], loc='upper left', frameon=False, fontsize=10)
    ax.set_xticks(days)
    ax.set_xlim(0.4, PR.size + 0.9)
    ax.set_ylim(-1.6, 6.2)
    ax.set_yticks(range(7))
    ax.spines['left'].set_bounds(0, 6.2)
    ax.set_xlabel('Día', fontsize=12)
    ax.set_ylabel('Precipitación (mm día$^{-1}$)', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
