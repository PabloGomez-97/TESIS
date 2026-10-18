"""Geografia comun de la tesis (Opcion A): 14 regiones administrativas y 4 macrozonas.

Una sola particion para todo el analisis:
- nivel fino: las 14 regiones administrativas (Natural Earth admin-1) cuyas celdas caen en el dominio de ALADIN;
- nivel grueso: 4 macrozonas, que son uniones de esas mismas regiones.

Aysen queda fuera: el dominio de ALADIN solo llega a ~45,9 S y cubre una fraccion de la region.
Regla de pertenencia: el centro de la celda de ALADIN cae dentro del poligono de la region.
"""
import numpy as np
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

# (nombre en Natural Earth, nombre corto, macrozona), ordenadas de norte a sur
REGIONES = [
    ('Arica y Parinacota', 'Arica y Parinacota', 'Norte árido'),
    ('Tarapacá', 'Tarapacá', 'Norte árido'),
    ('Antofagasta', 'Antofagasta', 'Norte árido'),
    ('Atacama', 'Atacama', 'Norte semiárido'),
    ('Coquimbo', 'Coquimbo', 'Norte semiárido'),
    ('Valparaíso', 'Valparaíso', 'Centro mediterráneo'),
    ('Región Metropolitana de Santiago', 'Metropolitana', 'Centro mediterráneo'),
    ("Libertador General Bernardo O'Higgins", "O'Higgins", 'Centro mediterráneo'),
    ('Maule', 'Maule', 'Centro mediterráneo'),
    ('Ñuble', 'Ñuble', 'Centro mediterráneo'),
    ('Bío-Bío', 'Biobío', 'Sur templado'),
    ('La Araucanía', 'La Araucanía', 'Sur templado'),
    ('Los Ríos', 'Los Ríos', 'Sur templado'),
    ('Los Lagos', 'Los Lagos', 'Sur templado'),
]
MACROZONAS = ['Norte árido', 'Norte semiárido', 'Centro mediterráneo', 'Sur templado']
NOMBRES_CORTOS = [r[1] for r in REGIONES]
MACROZONA_DE = {r[1]: r[2] for r in REGIONES}


def chile_geometry():
    reader = shpreader.Reader(
        shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
    )
    return unary_union([
        r.geometry for r in reader.records()
        if r.attributes.get('NAME') == 'Chile' or r.attributes.get('ADMIN') == 'Chile'
    ])


def region_geometries():
    """dict nombre_corto -> geometria (Natural Earth admin-1, 10m)."""
    reader = shpreader.Reader(
        shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
    )
    por_ne = {r.attributes['name']: r.geometry for r in reader.records() if r.attributes.get('admin') == 'Chile'}
    return {corto: por_ne[ne] for ne, corto, _ in REGIONES}


def chile_mask(lat2d, lon2d):
    """Mascara de Chile sobre la grilla de ALADIN (centro dentro del poligono o tocandolo)."""
    geom = chile_geometry()
    prepared = prep(geom)
    return np.fromiter(
        (prepared.contains(Point(float(x), float(y))) or geom.touches(Point(float(x), float(y)))
         for y, x in zip(lat2d.ravel(), lon2d.ravel())),
        dtype=bool, count=lat2d.size,
    ).reshape(lat2d.shape)


def region_index(lat2d, lon2d):
    """Devuelve (mascara_chile, idx_region) sobre la grilla 2D.

    idx_region[j, i] = posicion (0..13) en REGIONES, o -1 si la celda no esta en Chile o cae en Aysen.
    """
    mask = chile_mask(lat2d, lon2d)
    geoms = region_geometries()
    idx = np.full(lat2d.shape, -1, dtype=np.int16)
    jj, ii = np.where(mask)
    prepared = [prep(geoms[n]) for n in NOMBRES_CORTOS]
    for j, i in zip(jj, ii):
        p = Point(float(lon2d[j, i]), float(lat2d[j, i]))
        for k, pg in enumerate(prepared):
            if pg.contains(p):
                idx[j, i] = k
                break
    return mask, idx
