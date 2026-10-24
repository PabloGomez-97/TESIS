"""Figura 1: dominio de estudio con las 4 ventanas de las PDFs y las 4 regiones de los risk ratios.

Ventanas: cajas de latitud-longitud (mismos limites que _generar_pregunta9_pdf_loglog_cuencas.py).
Regiones: geometrias admin_1 de Natural Earth 1:10m (mismas que _run_pregunta5_aladin.py).
Las celdas de Chile de la grilla ALADIN recibida cubren 17,7-45,9 S (ver PROCEDENCIA_DATOS.md): el titulo lo declara.
Salida: paper_figuras/fig02_dominio_cuencas_regiones.png
"""
import warnings
warnings.filterwarnings('ignore')

import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper_figuras'
OUT.mkdir(exist_ok=True)

CHILE_EXTENT = [-76, -65, -55, -17]

BASIN_SPECS = {
    'Loa': {'bounds': (-69.8, -68.2, -24.0, -21.5)},
    'Maipo': {'bounds': (-71.8, -69.8, -34.5, -33.2)},
    'Maule': {'bounds': (-72.8, -71.0, -36.5, -34.8)},
    'Biobio': {'bounds': (-73.8, -71.5, -38.5, -36.5)},
}

REGION_SPECS = {
    'Coquimbo': {'query': 'coquimbo'},
    "O'Higgins": {'query': 'higgins'},
    'La Araucania': {'query': 'araucan'},
    'Los Lagos': {'query': 'los lagos'},
}

# Paletas separadas (cajas calidas, regiones frias) para no sugerir pares caja-region.
BASIN_STYLE = {
    'Loa': ('Loa', '#D4A017'),
    'Maipo': ('Maipo', '#C0392B'),
    'Maule': ('Maule', '#E67E22'),
    'Biobio': ('Biobío', '#C2185B'),
}
REGION_STYLE = {
    'Coquimbo': ('Coquimbo', '#7B68B5', None),
    "O'Higgins": ("O'Higgins", '#2C7FB8', (-71.2, -34.65)),
    'La Araucania': ('La Araucanía', '#2AA198', None),
    'Los Lagos': ('Los Lagos', '#253494', None),
}
REGION_LABEL_LON = -66.6


def normalize_text(text):
    text = unicodedata.normalize('NFKD', str(text).lower())
    return ''.join(ch for ch in text if not unicodedata.combining(ch))


def get_chile_admin1_records():
    reader = shpreader.Reader(
        shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
    )
    return [r for r in reader.records() if r.attributes.get('admin') == 'Chile']


def find_region_record(query, records):
    q = normalize_text(query)
    for rec in records:
        if q in normalize_text(rec.attributes.get('name', '')):
            return rec
    raise ValueError(f'No se encontro region chilena para: {query}')


def main():
    admin_records = get_chile_admin1_records()
    fig, ax = plt.subplots(figsize=(7.5, 14), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(CHILE_EXTENT)
    ax.add_feature(cfeature.LAND, facecolor='#F5F5F0', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#E8F4FC', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=2)

    for name, spec in REGION_SPECS.items():
        rec = find_region_record(spec['query'], admin_records)
        label, color, target = REGION_STYLE[name]
        ax.add_geometries([rec.geometry], ccrs.PlateCarree(), facecolor=color, edgecolor='white',
                          linewidth=0.6, alpha=0.45, zorder=1)
        if target is None:
            rep = rec.geometry.representative_point()
            target = (rep.x, rep.y)
        ax.annotate(
            label, xy=target, xytext=(REGION_LABEL_LON, target[1]),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            fontsize=8, ha='center', va='center', color=color, fontweight='bold',
            arrowprops=dict(arrowstyle='-', color=color, lw=0.8),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, lw=0.6, alpha=0.9),
            zorder=7,
        )

    for name, spec in BASIN_SPECS.items():
        label, color = BASIN_STYLE[name]
        lon_min, lon_max, lat_min, lat_max = spec['bounds']
        ax.add_patch(Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=2.2, edgecolor=color, facecolor='none', transform=ccrs.PlateCarree(), zorder=5,
        ))
        ax.text(
            0.5 * (lon_min + lon_max), 0.5 * (lat_min + lat_max), label, transform=ccrs.PlateCarree(),
            fontsize=9, ha='center', va='center', color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75), zorder=6,
        )

    legend_items = [
        mpatches.Patch(facecolor=color, edgecolor='white', alpha=0.6, label=f'Región {label}')
        for label, color, _ in REGION_STYLE.values()
    ]
    legend_items += [
        Line2D([0], [0], color=color, lw=2.5, label=f'Ventana {label}') for label, color in BASIN_STYLE.values()
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=7, framealpha=0.9, ncol=1)
    ax.set_title(
        'Dominio de estudio — Chile entre 17,7°S y 45,9°S (grilla ALADIN CHP12)\n'
        'Ventanas para las PDFs y regiones administrativas para los risk ratios',
        fontweight='bold', fontsize=10,
    )
    plt.tight_layout()
    path = OUT / 'fig02_dominio_cuencas_regiones.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {path}')


if __name__ == '__main__':
    main()
