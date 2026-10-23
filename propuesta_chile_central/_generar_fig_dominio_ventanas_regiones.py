"""Figura 1 (propuesta «Chile central»): dominio de estudio con las 4 regiones administrativas, usadas tanto para las PDFs
como para los risk ratios (REGLA.md).

Mismo estilo que la Figura 1 anterior (encuadre, colores de tierra y mar, rotulos al costado con linea guia, leyenda), pero con
las regiones Valparaiso, Metropolitana, O'Higgins y Maule y sin cajas de latitud-longitud. Geometrias admin_1 de Natural Earth 1:10m.
Las celdas de Chile de la grilla ALADIN recibida cubren 17,7-45,9 S (ver PROCEDENCIA_DATOS.md): el titulo lo declara.
Salida: paper_figuras/fig01_dominio_regiones_chile_central.png
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import unidades as U

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper_figuras'
OUT.mkdir(exist_ok=True)

CHILE_EXTENT = [-76, -65, -55, -17]
REGION_LABEL_LON = -66.6
SEPARACION_ROTULOS = 1.6  # grados de latitud minimos entre rotulos (las regiones son contiguas)


def main():
    z = np.load(ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz')
    lat, lon, chile = z['lat'], z['lon'], z['chile_mask']
    masks = U.mascaras(lat, lon, chile)
    recs = U.registros_chile()

    fig, ax = plt.subplots(figsize=(7.5, 14), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(CHILE_EXTENT)
    ax.add_feature(cfeature.LAND, facecolor='#F5F5F0', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#E8F4FC', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=2)

    # punto de anclaje de cada rotulo: centroide de las celdas de la region (la geometria de Valparaiso incluye islas lejanas)
    anclas = {k: (float(lon[m].mean()), float(lat[m].mean())) for k, m in masks.items()}
    y_prev = None
    for k, spec in U.UNIDADES.items():  # de norte a sur
        rec = U.registro(spec['query'], recs)
        ax.add_geometries([rec.geometry], ccrs.PlateCarree(), facecolor=spec['color'], edgecolor='white',
                          linewidth=0.6, alpha=0.45, zorder=1)
        x0, y0 = anclas[k]
        y_lab = y0 if y_prev is None else min(y0, y_prev - SEPARACION_ROTULOS)
        y_prev = y_lab
        ax.annotate(
            spec['etiqueta'], xy=(x0, y0), xytext=(REGION_LABEL_LON, y_lab),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            fontsize=8, ha='center', va='center', color=spec['color'], fontweight='bold',
            arrowprops=dict(arrowstyle='-', color=spec['color'], lw=0.8),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=spec['color'], lw=0.6, alpha=0.9),
            zorder=7,
        )

    legend_items = [
        mpatches.Patch(facecolor=spec['color'], edgecolor='white', alpha=0.6, label=f'Región {spec["etiqueta"]}')
        for spec in U.UNIDADES.values()
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=7, framealpha=0.9, ncol=1)
    ax.set_title(
        'Dominio de estudio — Chile entre 17,7°S y 45,9°S (grilla ALADIN CHP12)\n'
        'Regiones administrativas para las PDFs y los risk ratios',
        fontweight='bold', fontsize=10,
    )
    plt.tight_layout()
    path = OUT / 'fig01_dominio_regiones_chile_central.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {path}')
    for k, m in masks.items():
        print(f'{k}: {int(m.sum())} celdas, lat {lat[m].min():.2f} a {lat[m].max():.2f}, ancla {anclas[k]}')


if __name__ == '__main__':
    main()
