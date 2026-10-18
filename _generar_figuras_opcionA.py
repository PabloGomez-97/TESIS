"""Figuras de la geografia unica (Opcion A): mapa de zonas, gradiente de RR(20) y curvas RR(D) por macrozona."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE, region_geometries

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'
FIG = ROOT / 'paper_figuras'
COLOR = {'Norte árido': '#1D9E75', 'Norte semiárido': '#D85A30', 'Centro mediterráneo': '#534AB7', 'Sur templado': '#D4537E'}


def fig_mapa():
    g = np.load(OUT / 'grilla.npz')
    lat, lon, mask, idx = g['lat'], g['lon'], g['mask'], g['region_idx']
    fig, ax = plt.subplots(figsize=(5.2, 9))
    ax.scatter(lon[mask & (idx < 0)], lat[mask & (idx < 0)], s=3, c='#B4B2A9', marker='s', label='Aysén (fuera del análisis)')
    for m in MACROZONAS:
        ks = [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == m]
        sel = np.isin(idx, ks)
        ax.scatter(lon[sel], lat[sel], s=3, c=COLOR[m], marker='s', label=f'{m} ({int(sel.sum())} celdas)')
    for n, geom in region_geometries().items():
        for poly in getattr(geom, 'geoms', [geom]):
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', lw=0.4)
    for k, n in enumerate(NOMBRES_CORTOS):
        sel = idx == k
        ax.text(lon[sel].mean(), lat[sel].mean(), n, fontsize=6.5, ha='center', va='center', color='black')
    ax.set_xlim(-76, -66); ax.set_ylim(-46.5, -17)
    ax.set_xlabel('Longitud'); ax.set_ylabel('Latitud')
    ax.set_aspect(1.1)
    ax.legend(fontsize=7, loc='lower left', markerscale=3)
    ax.set_title('14 regiones y 4 macrozonas sobre la grilla de ALADIN', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / 'fig01_regiones_macrozonas.png', dpi=170, facecolor='white')


def fig_gradiente():
    rr = pd.read_csv(OUT / 'rr_ssp585.csv')
    t3 = pd.read_csv(OUT / 'tabla3_regiones.csv')
    d = rr[(rr.nivel == 'region') & (rr.D == 20)].set_index('unidad').loc[NOMBRES_CORTOS]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})
    x = np.arange(len(NOMBRES_CORTOS))
    cols = [COLOR[MACROZONA_DE[n]] for n in NOMBRES_CORTOS]
    a1.axhline(1, color='gray', ls=':')
    for xi, (n, r), c in zip(x, d.iterrows(), cols):
        a1.plot(xi, r.RR, 'o', color=c, ms=7)
        if r.estimable:
            a1.plot([xi, xi], [r.IC_low, r.IC_high], color=c, lw=2)
    a1.set_yscale('log'); a1.set_yticks([0.8, 1, 1.5, 2, 3, 5, 8]); a1.set_yticklabels(['0,8', '1', '1,5', '2', '3', '5', '8'])
    a1.set_ylabel('RR(20) = N_fut / N_hist  (IC95 %)')
    a1.set_title('Cambio proyectado en rachas ≥ 20 días (ALADIN SSP5-8.5 2040–2074 vs 1980–2014)', fontsize=10)
    a1.grid(alpha=0.25)
    w = 0.27
    for j, (nombre, lab, c) in enumerate([('cr2met_hist', 'CR2MET hist.', 'black'), ('aladin_hist', 'ALADIN hist.', 'steelblue'),
                                          ('aladin_ssp585', 'ALADIN SSP5-8.5', 'crimson')]):
        v = t3[t3.conjunto == nombre].set_index('region').loc[NOMBRES_CORTOS, 'p99']
        a2.bar(x + (j - 1) * w, v.values, w, color=c, label=lab)
    a2.set_yscale('log'); a2.set_ylabel('Percentil 99 de la duración (días)'); a2.legend(fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels(NOMBRES_CORTOS, rotation=45, ha='right', fontsize=8)
    a2.grid(alpha=0.25, axis='y')
    a1.legend(handles=[Patch(color=COLOR[m], label=m) for m in MACROZONAS], fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIG / 'fig09_gradiente_rr_regiones.png', dpi=170, facecolor='white')


def fig_curvas():
    rr = pd.read_csv(OUT / 'rr_ssp585.csv')
    fig, ejes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, m in zip(ejes, MACROZONAS):
        d = rr[(rr.nivel == 'macrozona') & (rr.unidad == m)].sort_values('D')
        ok = d.estimable
        ax.fill_between(d.D[ok], d.IC_low[ok], d.IC_high[ok], color=COLOR[m], alpha=0.25)
        ax.plot(d.D, d.RR, '-o', color=COLOR[m], ms=3.5)
        ax.set_yscale('log'); ax.set_yticks([0.5, 1, 2, 4, 8, 16]); ax.set_yticklabels(['0,5', '1', '2', '4', '8', '16'])
        ax.axhline(1, color='gray', ls=':')
        ax.axvline(20, color='gray', ls='--', lw=0.6)
        ax.set_title(m, fontsize=10); ax.set_xlabel('Umbral de duración D (días)'); ax.grid(alpha=0.25)
    ejes[0].set_ylabel('RR(D) con IC95 %')
    fig.tight_layout()
    fig.savefig(FIG / 'fig09b_curvas_rr_macrozonas.png', dpi=170, facecolor='white')


if __name__ == '__main__':
    fig_mapa(); fig_gradiente(); fig_curvas()
