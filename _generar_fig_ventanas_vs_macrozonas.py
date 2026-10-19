"""Compara cuanto territorio cubren las 4 ventanas antiguas (cajas de lat-lon) frente a las 4 macrozonas.

Las cajas son las de _generar_pregunta9_pdf_loglog_cuencas.py (BASIN_SPECS). Se cuentan las celdas de ALADIN dentro de
cada caja (con la mascara de Chile) y dentro de cada macrozona.
Salidas: paper_figuras/fig_ventanas_vs_macrozonas.png y resultados_tesis/cobertura_ventanas_macrozonas.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE

ROOT = Path(__file__).resolve().parent
COLOR = {'Norte árido': '#1D9E75', 'Norte semiárido': '#D85A30', 'Centro mediterráneo': '#534AB7', 'Sur templado': '#D4537E'}
CAJAS = {'Loa': (-69.8, -68.2, -24.0, -21.5), 'Maipo': (-71.8, -69.8, -34.5, -33.2),
         'Maule': (-72.8, -71.0, -36.5, -34.8), 'Biobío': (-73.8, -71.5, -38.5, -36.5)}


def main():
    g = np.load(ROOT / 'resultados_tesis' / 'grilla.npz')
    lat, lon, mask, idx = g['lat'], g['lon'], g['mask'], g['region_idx']
    filas = []
    en_caja = np.zeros(mask.shape, dtype=bool)
    fig, ax = plt.subplots(figsize=(6.4, 9.6))
    ax.scatter(lon[mask & (idx < 0)], lat[mask & (idx < 0)], s=3, c='#D3D1C7', marker='s')
    for m in MACROZONAS:
        ks = [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == m]
        sel = np.isin(idx, ks)
        ax.scatter(lon[sel], lat[sel], s=3, c=COLOR[m], marker='s', alpha=0.55, label=f'{m} ({int(sel.sum())} celdas)')
        filas.append(dict(tipo='macrozona', nombre=m, celdas=int(sel.sum())))
    for n, (x0, x1, y0, y1) in CAJAS.items():
        c = mask & (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
        en_caja |= c
        filas.append(dict(tipo='ventana', nombre=n, celdas=int(c.sum())))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='black', lw=1.8))
        ax.text(x1 + 0.15, (y0 + y1) / 2, f'{n}\n({int(c.sum())} celdas)', fontsize=8, va='center', fontweight='bold')
    total = int(mask.sum())
    filas.append(dict(tipo='resumen', nombre='celdas dentro de alguna ventana', celdas=int(en_caja.sum())))
    filas.append(dict(tipo='resumen', nombre='celdas en las 4 macrozonas', celdas=int((idx >= 0).sum())))
    filas.append(dict(tipo='resumen', nombre='celdas de Chile en el dominio', celdas=total))
    pd.DataFrame(filas).to_csv(ROOT / 'resultados_tesis' / 'cobertura_ventanas_macrozonas.csv', index=False)
    ax.set_xlim(-76.5, -64.5); ax.set_ylim(-46.5, -17)
    ax.set_aspect(1.1)
    ax.set_xlabel('Longitud'); ax.set_ylabel('Latitud')
    ax.set_title(f'Colores: 4 macrozonas ({int((idx >= 0).sum())} celdas, {100 * (idx >= 0).sum() / total:.0f} % del dominio)\n'
                 f'Cajas negras: 4 ventanas antiguas ({int(en_caja.sum())} celdas, {100 * en_caja.sum() / total:.0f} %)', fontsize=9)
    ax.legend(fontsize=7, loc='lower left', markerscale=3)
    fig.tight_layout()
    fig.savefig(ROOT / 'paper_figuras' / 'fig_ventanas_vs_macrozonas.png', dpi=170, facecolor='white')
    print(pd.DataFrame(filas).to_string(index=False))


if __name__ == '__main__':
    main()
