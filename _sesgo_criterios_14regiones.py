"""Sesgo historico ALADIN - CR2MET en las 14 regiones (sin Aysen) para los tres criterios de umbral.

Criterio i: R0 = 5,285 mm en ALADIN; ii: 1 mm en ambos; iii: umbral local por celda.
Usa los resultados por celda de _run_pregunta7_bootstrap.py (resultados_tesis/bootstrap_por_celda_*.npz):
celdas validas (al menos 30 rachas en ambos productos) dentro de las 14 regiones.
Salida: resultados_tesis/sesgo_criterios_14regiones.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'
CRITERIOS = {'i (R0 global)': 'A_opcion_B_integrado', 'ii (1 mm en ambos)': 'B_mismo_umbral', 'iii (R0 local)': 'C_umbral_local'}


def main():
    reg = np.load(OUT / 'grilla.npz')['region_idx']
    filas = []
    for nombre, clave in CRITERIOS.items():
        for met, etiqueta in (('mean', 'Duración media'), ('t99', 'Percentil 99')):
            z = np.load(OUT / f'bootstrap_por_celda_{clave}_{met}.npz')
            sel = z['valida'] & (reg >= 0)
            d = z['delta'][sel]
            filas.append(dict(criterio=nombre, metrica=etiqueta, celdas=int(sel.sum()),
                              delta_promedio=float(d.mean()), delta_mediana=float(np.median(d)),
                              pct_mas_corto=float(100 * z['mas_corto'][sel].mean()),
                              pct_mas_largo=float(100 * z['mas_largo'][sel].mean())))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / 'sesgo_criterios_14regiones.csv', index=False)
    print(df.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
