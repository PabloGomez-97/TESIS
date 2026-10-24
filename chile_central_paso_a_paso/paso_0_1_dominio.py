"""Paso 0.1 — Definir el dominio: que celdas de ALADIN pertenecen a Valparaiso, Metropolitana, O'Higgins y Maule.

Regla (REGLA.md): una celda pertenece a una region si su CENTRO cae dentro del poligono de esa region (Natural Earth admin_1,
1:10 m) o sobre su borde. El dominio es la union de las cuatro. Solo se lee la grilla (lat, lon, limites de celda) de un archivo de
ALADIN; no se usa ninguna precipitacion en este paso.

Salidas: salidas/paso_0_1_mascara_dominio.npz, salidas/paso_0_1_celdas_por_region.csv, salidas/paso_0_1_chequeos.txt,
         figuras/paso_0_1_dominio_celdas.png
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Point
from shapely.prepared import prep
import cartopy.io.shapereader as shpreader
import unicodedata

ROOT = Path(__file__).resolve().parent
REGIONES = {  # clave: (etiqueta, texto buscado en el nombre de Natural Earth, color)
    'Valparaiso': ('Valparaíso', 'valpara', '#7B68B5'),
    'Metropolitana': ('Metropolitana', 'metropolit', '#2C7FB8'),
    'OHiggins': ("O'Higgins", 'higgins', '#2AA198'),
    'Maule': ('Maule', 'maule', '#253494'),
}


def norm(t):
    t = unicodedata.normalize('NFKD', str(t).lower())
    return ''.join(c for c in t if not unicodedata.combining(c))


# 1) Grilla de ALADIN (solo coordenadas)
archivo = sorted(glob.glob(str(ROOT / 'pr1' / 'pr_CHP12_*_historical_*19790101-20141231*.nc')))[0]
ds = xr.open_dataset(archivo)
lat, lon = ds['lat'].values, ds['lon'].values
blat, blon = ds['bounds_lat'].values, ds['bounds_lon'].values  # (y, x, 4): esquinas de cada celda
ny, nx = lat.shape

# 2) Poligonos de las cuatro regiones
reader = shpreader.Reader(shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces'))
chile = [r for r in reader.records() if r.attributes.get('admin') == 'Chile']
geom = {}
for k, (etq, q, _) in REGIONES.items():
    hits = [r for r in chile if norm(q) in norm(r.attributes.get('name', ''))]
    assert len(hits) == 1, f'{q}: {[h.attributes["name"] for h in hits]}'
    geom[k] = hits[0].geometry
    print(k, '->', hits[0].attributes['name'])

# 3) Regla: el centro de la celda cae dentro del poligono (o sobre su borde)
region_id = np.full((ny, nx), -1, dtype=np.int8)
solapes = 0
for idx, (k, g) in enumerate(geom.items()):
    p = prep(g)
    dentro = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        for i in range(nx):
            pt = Point(float(lon[j, i]), float(lat[j, i]))
            dentro[j, i] = p.contains(pt) or g.touches(pt)
    solapes += int(((region_id >= 0) & dentro).sum())
    region_id[dentro & (region_id < 0)] = idx
dominio = region_id >= 0
nombres = list(REGIONES)

# 4) Resultados
filas = []
for idx, k in enumerate(nombres):
    m = region_id == idx
    filas.append(dict(region=REGIONES[k][0], celdas=int(m.sum()), lat_min=lat[m].min(), lat_max=lat[m].max(),
                      lon_min=lon[m].min(), lon_max=lon[m].max()))
filas.append(dict(region='Dominio (unión)', celdas=int(dominio.sum()), lat_min=lat[dominio].min(), lat_max=lat[dominio].max(),
                  lon_min=lon[dominio].min(), lon_max=lon[dominio].max()))
tabla = pd.DataFrame(filas)
(ROOT / 'salidas').mkdir(exist_ok=True)
tabla.to_csv(ROOT / 'salidas' / 'paso_0_1_celdas_por_region.csv', index=False)
np.savez(ROOT / 'salidas' / 'paso_0_1_mascara_dominio.npz', lat=lat, lon=lon, dominio=dominio, region_id=region_id,
         nombres=np.array([REGIONES[k][0] for k in nombres]))
print(tabla.round(2).to_string(index=False))

# 5) Chequeos
chk = []
chk.append(f'Grilla ALADIN: {ny} x {nx} = {ny * nx} celdas (archivo {Path(archivo).name})')
chk.append(f'Celdas cuyo centro cae en dos regiones a la vez (debe ser 0): {solapes}')
chk.append(f'Celdas del dominio: {int(dominio.sum())}; suma por region: {sum(f["celdas"] for f in filas[:-1])}')
# celdas cuyo centro cae sobre el borde (touches) y no estrictamente dentro
borde = 0
for k, g in geom.items():
    p = prep(g)
    for j, i in zip(*np.where(region_id == nombres.index(k))):
        pt = Point(float(lon[j, i]), float(lat[j, i]))
        if not p.contains(pt):
            borde += 1
chk.append(f'Celdas con el centro exactamente sobre el borde de un poligono: {borde}')
# tamano de celda: ancho y alto aproximados (km) a partir de las esquinas
km_lat = 111.19
alto = (blat.max(axis=2) - blat.min(axis=2)) * km_lat
ancho = (blon.max(axis=2) - blon.min(axis=2)) * km_lat * np.cos(np.deg2rad(lat))
chk.append(f'Tamano de celda en el dominio (aprox.): alto {alto[dominio].mean():.1f} km, ancho {ancho[dominio].mean():.1f} km '
           f'-> area {np.mean(alto[dominio] * ancho[dominio]):.0f} km2 por celda; area total ≈ {np.sum(alto[dominio] * ancho[dominio]):.0f} km2')
# ciudades de referencia: que region contiene la celda cuyo centro esta mas cerca
for ciudad, (la, lo, esperada) in {'Santiago': (-33.45, -70.67, 'Metropolitana'), 'Rancagua': (-34.17, -70.74, 'OHiggins'),
                                   'Talca': (-35.43, -71.66, 'Maule'), 'Valparaíso (ciudad)': (-33.05, -71.62, 'Valparaiso')}.items():
    d2 = (lat - la) ** 2 + ((lon - lo) * np.cos(np.deg2rad(la))) ** 2
    j, i = np.unravel_index(np.argmin(d2), d2.shape)
    r = region_id[j, i]
    chk.append(f'{ciudad}: celda más cercana (lat {lat[j, i]:.2f}, lon {lon[j, i]:.2f}) -> '
               f'{REGIONES[nombres[r]][0] if r >= 0 else "fuera del dominio"} (esperada: {REGIONES[esperada][0]})')
(ROOT / 'salidas' / 'paso_0_1_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))

# 6) Figura: panel izquierdo = grilla completa y dominio; derecho = zoom con celdas reales y contornos
fig, axs = plt.subplots(1, 2, figsize=(12, 7.5), gridspec_kw={'width_ratios': [1, 1.5]})
ax = axs[0]
ax.scatter(lon.ravel(), lat.ravel(), s=0.6, c='#cfcfcf', marker='s', linewidths=0)
for idx, k in enumerate(nombres):
    m = region_id == idx
    ax.scatter(lon[m], lat[m], s=3, c=REGIONES[k][2], marker='s', linewidths=0)
ax.set_title('Todas las celdas del archivo de ALADIN\n(gris) y celdas del dominio (color)', fontsize=10)
ax.set_xlabel('longitud'); ax.set_ylabel('latitud'); ax.set_aspect(1 / np.cos(np.deg2rad(34)))
ax = axs[1]
pad = 0.4
ext = [lon[dominio].min() - pad, lon[dominio].max() + pad, lat[dominio].min() - pad, lat[dominio].max() + pad]
cerca = (lon > ext[0] - 0.3) & (lon < ext[1] + 0.3) & (lat > ext[2] - 0.3) & (lat < ext[3] + 0.3)
polys, cols = [], []
for j, i in zip(*np.where(cerca)):
    polys.append(np.column_stack([blon[j, i], blat[j, i]]))
    r = region_id[j, i]
    cols.append(REGIONES[nombres[r]][2] if r >= 0 else '#ececec')
ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors='white', linewidths=0.3, alpha=0.9))
for k, g in geom.items():
    for poly in (g.geoms if hasattr(g, 'geoms') else [g]):
        x, y = poly.exterior.xy
        ax.plot(x, y, color='k', lw=0.8)
ax.scatter(lon[dominio], lat[dominio], s=1.2, c='k', zorder=3)
for ciudad, (la, lo) in {'Valparaíso': (-33.05, -71.62), 'Santiago': (-33.45, -70.67), 'Rancagua': (-34.17, -70.74), 'Talca': (-35.43, -71.66)}.items():
    ax.plot(lo, la, marker='*', color='#c0392b', ms=9, zorder=4)
    ax.text(lo + 0.05, la + 0.05, ciudad, fontsize=8, color='#c0392b', zorder=4)
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect(1 / np.cos(np.deg2rad(34)))
handles = [plt.Rectangle((0, 0), 1, 1, color=REGIONES[k][2], label=f'{REGIONES[k][0]} ({int((region_id == i).sum())} celdas)') for i, k in enumerate(nombres)]
ax.legend(handles=handles, loc='lower left', fontsize=8)
ax.set_title(f'Zoom: cada cuadrado es una celda de ALADIN (≈ 12 km); punto negro = centro\nDominio = unión de las 4 regiones: {int(dominio.sum())} celdas', fontsize=10)
ax.set_xlabel('longitud')
fig.tight_layout()
(ROOT / 'figuras').mkdir(exist_ok=True)
fig.savefig(ROOT / 'figuras' / 'paso_0_1_dominio_celdas.png', dpi=170)
print('figura lista')
