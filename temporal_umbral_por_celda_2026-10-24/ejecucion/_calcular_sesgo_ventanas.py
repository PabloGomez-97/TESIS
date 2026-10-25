"""Sesgo historico de ALADIN respecto de CR2MET por ventana (Loa, Maipo, Maule, Biobio), umbral local principal, 1980-2014.

Diferencia por celda (ALADIN - CR2MET) de la duracion media y del percentil 99 (t99) dentro de cada ventana de
latitud-longitud del paper, usando las celdas con al menos 30 rachas en ambos productos (mismas celdas y reglas
del bootstrap del sesgo nacional). Se informa el promedio y la mediana de las diferencias por ventana, el numero de
celdas de la ventana dentro de Chile y cuantas entran (con >= 30 rachas), y lo mismo para todo el dominio.

Entrada: _pregunta7_outputs/metrics_integrado_cache.npz (generada por _run_pregunta7_bootstrap.py)
Salida: resultados_tesis/sesgo_por_ventana.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# (lon_min, lon_max, lat_min, lat_max); mismas ventanas que _generar_pregunta9_pdf_loglog_cuencas.py
VENTANAS = {
    'Loa': (-69.8, -68.2, -24.0, -21.5),
    'Maipo': (-71.8, -69.8, -34.5, -33.2),
    'Maule': (-72.8, -71.0, -36.5, -34.8),
    'Biobio': (-73.8, -71.5, -38.5, -36.5),
}


def main():
    z = np.load(ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz')
    lat, lon, chile = z['lat'], z['lon'], z['chile_mask']
    filas = []
    for met, nombre in (('mean', 'Duración media'), ('t99', 'Percentil 99')):
        c, a = z[f'{met}_cr2'], z[f'{met}_ala']
        ok = np.isfinite(c) & np.isfinite(a) & chile
        d = a - c
        zonas = dict(VENTANAS)
        zonas['Todo el dominio'] = None
        for nom, b in zonas.items():
            if b is None:
                en_ventana = chile
            else:
                lo, hi, la, lb = b
                en_ventana = chile & (lon >= lo) & (lon <= hi) & (lat >= la) & (lat <= lb)
            sel = ok & en_ventana
            filas.append(dict(
                metrica=nombre, zona=nom, celdas_ventana=int(en_ventana.sum()), celdas_validas=int(sel.sum()),
                cr2met_prom=float(c[sel].mean()), aladin_prom=float(a[sel].mean()),
                delta_promedio=float(d[sel].mean()), delta_mediana=float(np.median(d[sel])),
            ))
    df = pd.DataFrame(filas)
    (ROOT / 'resultados_tesis').mkdir(exist_ok=True)
    df.to_csv(ROOT / 'resultados_tesis' / 'sesgo_por_ventana.csv', index=False)
    print(df.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
