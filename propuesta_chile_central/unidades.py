"""Unidades de la propuesta «Chile central» (ver REGLA.md).

Cuatro regiones administrativas (Natural Earth admin_1, 1:10 m) usadas tanto para las PDFs/Tabla 3 como para los risk ratios.
Clave = nombre sin tildes ni apostrofes (archivos y pickles); etiqueta = nombre en las figuras; query = texto buscado en el
nombre de Natural Earth (sin tildes, en minusculas).
"""
import numpy as np
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.prepared import prep

UNIDADES = {
    'Valparaiso': {'etiqueta': 'Valparaíso', 'query': 'valpara', 'color': '#7B68B5'},
    'Metropolitana': {'etiqueta': 'Metropolitana', 'query': 'metropolit', 'color': '#2C7FB8'},
    'OHiggins': {'etiqueta': "O'Higgins", 'query': 'higgins', 'color': '#2AA198'},
    'Maule': {'etiqueta': 'Maule', 'query': 'maule', 'color': '#253494'},
}


def registros_chile():
    reader = shpreader.Reader(shpreader.natural_earth(resolution='10m', category='cultural',
                                                      name='admin_1_states_provinces'))
    return [r for r in reader.records() if r.attributes.get('admin') == 'Chile']


def _norm(texto):
    import unicodedata
    t = unicodedata.normalize('NFKD', str(texto).lower())
    return ''.join(c for c in t if not unicodedata.combining(c))


def registro(query, records=None):
    records = records or registros_chile()
    hits = [r for r in records if _norm(query) in _norm(r.attributes.get('name', ''))]
    if len(hits) != 1:
        raise ValueError(f'«{query}» coincide con {len(hits)} regiones: {[h.attributes["name"] for h in hits]}')
    return hits[0]


def mascara_poligono(lat2d, lon2d, geom):
    """Celdas cuyo centro cae dentro del poligono o sobre su borde (misma regla que _run_pregunta5_aladin.py)."""
    p = prep(geom)
    return np.fromiter(
        (p.contains(Point(float(lon2d[j, i]), float(lat2d[j, i])))
         or geom.touches(Point(float(lon2d[j, i]), float(lat2d[j, i])))
         for j in range(lat2d.shape[0]) for i in range(lat2d.shape[1])),
        dtype=bool, count=lat2d.size).reshape(lat2d.shape)


def mascaras(lat2d, lon2d, chile_bool=None):
    """Diccionario clave -> mascara booleana (opcionalmente interseccion con la mascara de Chile). Exige que no se solapen."""
    recs = registros_chile()
    out, ocupado = {}, np.zeros(lat2d.shape, dtype=bool)
    for k, spec in UNIDADES.items():
        m = mascara_poligono(lat2d, lon2d, registro(spec['query'], recs).geometry)
        if chile_bool is not None:
            m = m & chile_bool
        assert not np.any(m & ocupado), f'{k} se solapa con otra unidad'
        ocupado |= m
        out[k] = m
    return out
