"""Figura de las PDFs de duracion de dry spells por ventana (criterio i), version 2x2 con leyenda unica.

Lee las duraciones por ventana de _pregunta9_outputs/pdf_durations_i_global.pkl (las genera
_generar_pregunta9_pdf_loglog_cuencas.py; reglas §2.2, R0 = 5,285 mm en ALADIN, 1 mm en CR2MET) y usa los mismos bins
comunes por ventana que la version anterior. No recalcula las rachas.

Elementos: letras (a)-(d); leyenda unica; marcas triangulares del p99 de cada serie sobre el eje x; circulos huecos en los
bins con menos de MIN_N rachas (los extremos de la cola, con pocos eventos, no deben leerse como senal).
Salida: paper_figuras/fig_pdfs_ventanas.png
"""
import warnings
warnings.filterwarnings('ignore')

import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import _generar_pregunta9_pdf_loglog_cuencas as f6

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT / '_pregunta9_outputs' / 'pdf_durations_i_global.pkl'
SALIDA = ROOT / 'paper_figuras' / 'fig_pdfs_ventanas.png'
MIN_N = 10  # bins con menos rachas se dibujan como circulo hueco

ESTILO = {
    'CR2MET (hist)': dict(color='#111111', ls='-', lw=2.0, label='CR2MET 1980–2014'),
    'ALADIN (hist)': dict(color='#1f77b4', ls='-', lw=1.8, label='ALADIN 1980–2014'),
    'ALADIN (futuro)': dict(color='#d62728', ls='--', lw=1.8, label='ALADIN 2040–2074 (SSP5-8.5)'),
}
VENTANAS = [('Loa', '(a) Loa'), ('Maipo', '(b) Maipo'), ('Maule', '(c) Maule'), ('Biobio', '(d) Biobío')]


def main():
    with ENTRADA.open('rb') as f:
        durs = pickle.load(f)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    for ax, (clave, titulo) in zip(axes.ravel(), VENTANAS):
        d_v = durs[clave]
        bins = f6.common_log_bins(d_v)
        ymin = np.inf
        for serie, est in ESTILO.items():
            d = np.asarray(d_v[serie], dtype=float)
            counts, edges = np.histogram(d, bins=bins)
            centros = np.sqrt((edges[:-1] + 0.5) * (edges[1:] - 0.5))
            alturas = counts / np.diff(edges)
            ok = alturas > 0
            ax.plot(centros[ok], alturas[ok], color=est['color'], ls=est['ls'], lw=est['lw'], zorder=3)
            pocos = ok & (counts < MIN_N)
            ax.plot(centros[pocos], alturas[pocos], 'o', mfc='white', mec=est['color'], ms=4.5, mew=1.1, zorder=4)
            ymin = min(ymin, alturas[ok].min())
        y0 = ymin / 3.0
        ax.set_ylim(bottom=y0)
        for serie, est in ESTILO.items():
            p99 = np.percentile(np.asarray(d_v[serie], dtype=float), 99)
            ax.plot([p99], [y0 * 1.25], marker='^', color=est['color'], ms=8, clip_on=False, zorder=5, ls='none')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(titulo, loc='left', fontweight='bold')
        ax.grid(True, which='major', alpha=0.25)
        ax.set_xlabel('Duración del dry spell (días)')
        ax.set_ylabel('f(t)  (rachas por día de duración)')
    handles = [Line2D([0], [0], color=e['color'], ls=e['ls'], lw=e['lw'], label=e['label']) for e in ESTILO.values()]
    handles += [
        Line2D([0], [0], marker='o', mfc='white', mec='0.3', ls='none', ms=5, label=f'bin con < {MIN_N} rachas'),
        Line2D([0], [0], marker='^', color='0.3', ls='none', ms=8, label='percentil 99 de cada serie'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    SALIDA.parent.mkdir(exist_ok=True)
    fig.savefig(SALIDA, dpi=300, facecolor='white')
    plt.close(fig)
    print(f'-> {SALIDA}')


if __name__ == '__main__':
    main()
