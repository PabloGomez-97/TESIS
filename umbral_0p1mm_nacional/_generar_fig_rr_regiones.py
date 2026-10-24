"""Figura de los risk ratios RR(D) = N_fut(t >= D) / N_hist(t >= D) por region, version 2x2 con eje y logaritmico.

Lee outputs_pregunta5_aladin/results.json (lo genera _run_pregunta5_aladin.py: ALADIN SSP5-8.5 2040-2074 frente a 1980-2014,
R0 = 3,668 mm, rachas con inicio mar-nov, IC95 por bootstrap de anios). No recalcula nada.

Elementos: eje y logaritmico (un aumento y una disminucion de igual factor quedan a la misma distancia de 1); letras (a)-(d); banda
IC95 solo donde ninguna replica del bootstrap se descarto (N_hist* = 0); donde se descartaron replicas el punto es un circulo hueco y
no se dibuja banda; donde N_hist(t >= D) = 0 el RR no esta definido y la curva termina; linea vertical en D = 20 d (valor de
referencia del texto) con RR e IC95 anotados.
Salida: paper_figuras/fig_rr_regiones.png
"""
import warnings
warnings.filterwarnings('ignore')

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT / 'outputs_pregunta5_aladin' / 'results.json'
SALIDA = ROOT / 'paper_figuras' / 'fig_rr_regiones.png'
REGIONES = [("Coquimbo", '(a) Coquimbo', '#7B68B5'), ("O'Higgins", "(b) O'Higgins", '#2C7FB8'),
            ('La Araucanía', '(c) La Araucanía', '#2AA198'), ('Los Lagos', '(d) Los Lagos', '#253494')]
YLIM = (0.5, 40.0)
D_REF = 20


def coma(x, n=2):
    return f'{x:.{n}f}'.replace('.', ',')


def main():
    res = json.loads(ENTRADA.read_text(encoding='utf-8'))
    tabla4 = {r['Region']: r for r in res['rr_table']}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True)
    for ax, (nombre, titulo, color) in zip(axes.ravel(), REGIONES):
        c = res['curves'][nombre]
        D = np.array([x['Umbral (dias)'] for x in c], dtype=float)
        rr = np.array([x['RR'] if x['RR'] is not None else np.nan for x in c], dtype=float)
        lo = np.array([x['IC95 inferior'] if x['IC95 inferior'] is not None else np.nan for x in c], dtype=float)
        hi = np.array([x['IC95 superior'] if x['IC95 superior'] is not None else np.nan for x in c], dtype=float)
        desc = np.array([x['Replicas descartadas'] for x in c])
        def_ = np.isfinite(rr)
        firme = def_ & (desc == 0) & np.isfinite(lo) & np.isfinite(hi)
        ax.axhline(1.0, color='0.35', lw=1.0, ls=':', zorder=1)
        ax.axvline(D_REF, color='0.75', lw=1.0, ls='--', zorder=1)
        # banda IC95 en tramos continuos donde es firme
        i = 0
        while i < len(D):
            if firme[i]:
                j = i
                while j + 1 < len(D) and firme[j + 1]:
                    j += 1
                ax.fill_between(D[i:j + 1], np.clip(lo[i:j + 1], YLIM[0], None), np.clip(hi[i:j + 1], None, YLIM[1]),
                                color=color, alpha=0.22, lw=0, zorder=2)
                i = j + 1
            else:
                i += 1
        ax.plot(D[def_], np.clip(rr[def_], YLIM[0], YLIM[1]), color=color, lw=2.0, zorder=3)
        ax.plot(D[firme], rr[firme], 'o', color=color, ms=5.5, zorder=4)
        hueco = def_ & ~firme
        ax.plot(D[hueco], np.clip(rr[hueco], YLIM[0], YLIM[1]), 'o', mfc='white', mec=color, mew=1.4, ms=6, zorder=4)
        if (~def_).any():
            d_ndef = D[~def_].min()
            ax.annotate('RR no definido\n(N_hist = 0)', xy=(d_ndef, 1.0), xytext=(d_ndef + 1.5, 2.2),
                        fontsize=8.5, color='0.25', ha='left', va='bottom')
        t = tabla4[nombre]
        miles = lambda n: f'{int(n):,}'.replace(',', ' ')
        ax.text(0.02, 0.97,
                f"D = {D_REF} d: RR = {coma(t['RR observado'])} [{coma(t['IC95 inferior'])}; {coma(t['IC95 superior'])}]\n"
                f"N_hist = {miles(t['N_hist'])} → N_fut = {miles(t['N_fut'])}",
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.8', alpha=0.95))
        ax.set_yscale('log')
        ax.set_ylim(*YLIM)
        ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32])
        ax.set_yticklabels(['0,5', '1', '2', '4', '8', '16', '32'])
        ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
        ax.set_title(titulo, loc='left', fontweight='bold')
        ax.grid(True, which='major', axis='y', alpha=0.25)
        ax.set_ylabel('RR(D) = N_fut / N_hist')
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel('Duración mínima de la racha, D (días)')
    handles = [
        Line2D([0], [0], color='0.3', lw=2, marker='o', ms=5, label='RR (IC95 en la banda)'),
        Patch(facecolor='0.6', alpha=0.3, label='IC95 bootstrap de años'),
        Line2D([0], [0], marker='o', mfc='white', mec='0.3', ls='none', ms=6, mew=1.4,
               label='réplicas sin rachas históricas: sin IC'),
        Line2D([0], [0], color='0.35', ls=':', lw=1.2, label='RR = 1'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    SALIDA.parent.mkdir(exist_ok=True)
    fig.savefig(SALIDA, dpi=300, facecolor='white')
    plt.close(fig)
    print(f'-> {SALIDA}')


if __name__ == '__main__':
    main()
