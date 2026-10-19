"""Mapa de la grilla de ALADIN sobre Chile: celdas del analisis, celdas excluidas por la regla de 30 rachas y
acercamiento que compara la grilla de ALADIN (~12 km) con la nativa de CR2MET (0,05 grados, ~5 km).

Entradas: resultados_tesis/grilla.npz, _pregunta7_outputs/metrics_integrado_cache.npz, un archivo de CR2MET.
Salida: paper_figuras/fig_grilla_aladin.png y resultados_tesis/celdas_resumen.csv
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE

ROOT = Path(__file__).resolve().parent
COLOR = {'Norte árido': '#1D9E75', 'Norte semiárido': '#D85A30', 'Centro mediterráneo': '#534AB7', 'Sur templado': '#D4537E'}


def main():
    g = np.load(ROOT / 'resultados_tesis' / 'grilla.npz')
    z = np.load(ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz')
    lat, lon, mask, idx = g['lat'], g['lon'], g['mask'], g['region_idx']
    valida = np.isfinite(z['mean_cr2']) & np.isfinite(z['mean_ala']) & z['chile_mask']
    excl = mask & ~valida

    resumen = [('Celdas de la grilla recortada de ALADIN (255 x 68)', mask.size),
               ('Celdas cuyo centro cae dentro de Chile (máscara)', int(mask.sum())),
               ('  en las 14 regiones del análisis', int((idx >= 0).sum())),
               ('  en Aysén (fuera del análisis)', int((mask & (idx < 0)).sum())),
               ('Celdas con al menos 30 rachas en ambos productos', int(valida.sum())),
               ('Celdas excluidas por la regla de 30 rachas', int(excl.sum()))]
    pd.DataFrame(resumen, columns=['concepto', 'celdas']).to_csv(ROOT / 'resultados_tesis' / 'celdas_resumen.csv', index=False)

    fig = plt.figure(figsize=(11, 8.6))
    ax = fig.add_axes([0.04, 0.06, 0.34, 0.88])
    ax.scatter(lon[~mask], lat[~mask], s=0.8, c='#E1E0D9', marker='s', label='Celdas fuera de Chile (océano, países vecinos)')
    ax.scatter(lon[mask & (idx < 0)], lat[mask & (idx < 0)], s=2.5, c='#B4B2A9', marker='s', label='Aysén (fuera del análisis)')
    for m in MACROZONAS:
        ks = [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == m]
        sel = np.isin(idx, ks) & valida
        ax.scatter(lon[sel], lat[sel], s=2.5, c=COLOR[m], marker='s', label=f'{m}')
    ax.scatter(lon[excl], lat[excl], s=22, c='black', marker='x', linewidths=0.9, label=f'Excluidas por la regla de 30 rachas ({int(excl.sum())})')
    ax.set_xlim(lon[mask].min() - 1, lon[mask].max() + 1)
    ax.set_ylim(lat[mask].min() - 0.5, lat[mask].max() + 0.5)
    ax.set_aspect(1.1)
    ax.set_xlabel('Longitud'); ax.set_ylabel('Latitud')
    ax.set_title('Cada punto es una celda de ALADIN (~12 km)', fontsize=10)
    ax.legend(fontsize=7, loc='lower left', markerscale=3)

    # acercamiento: Maule, ~1 grado
    lo0, la0 = -71.5, -35.7
    caja = (abs(lon - lo0) < 0.6) & (abs(lat - la0) < 0.6)
    ax2 = fig.add_axes([0.45, 0.30, 0.52, 0.64])
    cr = xr.open_dataset(sorted((ROOT / 'pr').glob('CR2MET_pr_v2.5_day_1980_01*.nc'))[0])
    LO, LA = np.meshgrid(cr['lon'].values, cr['lat'].values)
    c2 = (abs(LO - lo0) < 0.6) & (abs(LA - la0) < 0.6)
    ax2.scatter(LO[c2], LA[c2], s=6, c='#378ADD', label='Puntos de grilla de CR2MET (0,05°, ~5 km)')
    ax2.scatter(lon[caja], lat[caja], s=140, facecolors='none', edgecolors='#D85A30', linewidths=1.6, marker='s',
                label='Centros de celda de ALADIN (~12 km)')
    ax2.set_xlim(lo0 - 0.6, lo0 + 0.6); ax2.set_ylim(la0 - 0.6, la0 + 0.6)
    ax2.set_aspect(1.2)
    ax2.set_title('Acercamiento al Maule: una celda de ALADIN equivale a unos 6 puntos de CR2MET', fontsize=10)
    ax2.set_xlabel('Longitud'); ax2.set_ylabel('Latitud')
    ax2.legend(fontsize=8, loc='upper right')

    ax3 = fig.add_axes([0.45, 0.04, 0.52, 0.18]); ax3.axis('off')
    texto = '\n'.join(f'{a}: {b:,}'.replace(',', ' ') for a, b in resumen)
    ax3.text(0, 1, texto, va='top', fontsize=9, family='monospace')
    fig.savefig(ROOT / 'paper_figuras' / 'fig_grilla_aladin.png', dpi=170, facecolor='white')
    print(pd.DataFrame(resumen, columns=['concepto', 'celdas']).to_string(index=False))


if __name__ == '__main__':
    main()
