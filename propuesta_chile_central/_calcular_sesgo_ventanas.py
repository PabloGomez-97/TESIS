"""Sesgo historico de ALADIN respecto de CR2MET por unidad (4 regiones de la propuesta «Chile central»), criterio i, 1980-2014.

Diferencia por celda (ALADIN - CR2MET) de la duracion media y del percentil 99 (t99) dentro de cada region, usando las celdas
con al menos 30 rachas en ambos productos (mismas celdas y reglas del bootstrap del sesgo nacional). Se informa el promedio
y la mediana de las diferencias por region, el numero de celdas de la region y cuantas entran, y lo mismo para todo el dominio.

Entrada: _pregunta7_outputs/metrics_integrado_cache.npz (enlace a la cache del analisis de dominio completo, que no cambia)
Salida: resultados_tesis/sesgo_por_ventana.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import unidades as U

ROOT = Path(__file__).resolve().parent


def main():
    z = np.load(ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz')
    lat, lon, chile = z['lat'], z['lon'], z['chile_mask']
    zonas = dict(U.mascaras(lat, lon, chile))
    zonas['Todo el dominio'] = chile
    filas = []
    for met, nombre in (('mean', 'Duración media'), ('t99', 'Percentil 99')):
        c, a = z[f'{met}_cr2'], z[f'{met}_ala']
        ok = np.isfinite(c) & np.isfinite(a) & chile
        d = a - c
        for nom, en_ventana in zonas.items():
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
