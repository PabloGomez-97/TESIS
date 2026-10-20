"""Sesgo historico de ALADIN respecto de CR2MET por macrozona (criterio i, 1980-2014).

Diferencia por celda (ALADIN - CR2MET) de la duracion media y del percentil 99 (t99), usando las celdas con al
menos 30 rachas en ambos productos (mismas celdas y mismas reglas del bootstrap del sesgo). Se informa el promedio
y la mediana de las diferencias por macrozona y para las 14 regiones (Aysen queda fuera del analisis).

Entrada: _pregunta7_outputs/metrics_integrado_cache.npz (generada por _run_pregunta7_bootstrap.py)
Salidas: resultados_tesis/sesgo_por_macrozona.csv y paper_figuras/fig_sesgo_macrozonas.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE

ROOT = Path(__file__).resolve().parent
COLOR = {'Norte árido': '#1D9E75', 'Norte semiárido': '#D85A30', 'Centro mediterráneo': '#534AB7',
         'Sur templado': '#D4537E', 'Las 14 regiones': '#888780'}


def calcular():
    z = np.load(ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz')
    reg = np.load(ROOT / 'resultados_tesis' / 'grilla.npz')['region_idx']
    filas = []
    for met, nombre in (('mean', 'Duración media'), ('t99', 'Percentil 99')):
        c, a = z[f'{met}_cr2'], z[f'{met}_ala']
        ok = np.isfinite(c) & np.isfinite(a) & z['chile_mask']
        d = a - c
        for m in MACROZONAS + ['Las 14 regiones']:
            if m == 'Las 14 regiones':
                sel = ok & (reg >= 0)
            else:
                ks = [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == m]
                sel = ok & np.isin(reg, ks)
            filas.append(dict(metrica=nombre, zona=m, celdas=int(sel.sum()),
                              cr2met_prom=float(c[sel].mean()), aladin_prom=float(a[sel].mean()),
                              delta_promedio=float(d[sel].mean()), delta_mediana=float(np.median(d[sel])),
                              delta_relativo_pct=float(d[sel].mean() / c[sel].mean() * 100)))
    return pd.DataFrame(filas)


def figura(df):
    zonas = MACROZONAS + ['Las 14 regiones']
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, met in zip(ejes, ('Duración media', 'Percentil 99')):
        d = df[df.metrica == met].set_index('zona').loc[zonas]
        y = np.arange(len(zonas))[::-1]
        cols = [COLOR[z] for z in zonas]
        ax.barh(y + 0.18, d.delta_promedio, 0.34, color=cols, label='Promedio de las diferencias')
        ax.barh(y - 0.18, d.delta_mediana, 0.34, fill=False, edgecolor=cols, linewidth=1.8, label='Mediana de las diferencias')
        for yi, a, b in zip(y, d.delta_promedio, d.delta_mediana):
            ax.text(a, yi + 0.18, f' {a:.1f}'.replace('.', ','), va='center', ha='right', fontsize=8, fontweight='bold')
            ax.text(b, yi - 0.18, f' {b:.1f}'.replace('.', ','), va='center', ha='right', fontsize=8)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_title(f'{met} (días): ALADIN − CR2MET', fontsize=10)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{z}\n({int(d.loc[z, "celdas"])} celdas)' for z in zonas], fontsize=9)
        lim = d[['delta_promedio', 'delta_mediana']].min().min()
        ax.set_xlim(lim * 1.25, 3)
        ax.grid(alpha=0.25, axis='x')
    ejes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Cuánto acorta ALADIN las rachas respecto de CR2MET (1980–2014, criterio i)', fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / 'paper_figuras' / 'fig_sesgo_macrozonas.png', dpi=170, facecolor='white')


if __name__ == '__main__':
    df = calcular()
    df.to_csv(ROOT / 'resultados_tesis' / 'sesgo_por_macrozona.csv', index=False)
    figura(df)
    print(df.round(1).to_string(index=False))
