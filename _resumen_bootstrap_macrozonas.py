"""Porcentaje de celdas con diferencia ALADIN - CR2MET significativa, por macrozona y region.

Usa los resultados por celda que guarda _run_pregunta7_bootstrap.py (resultados_tesis/bootstrap_por_celda_*.npz).
Criterio i (A_opcion_B_integrado). 'mas_corto' = IC95 del Delta completamente bajo cero (ALADIN mas corto que CR2MET);
'mas_largo' = IC95 completamente sobre cero. Sin diferencia clara = IC95 incluye el cero.
Salida: resultados_tesis/bootstrap_significancia_por_zona.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'


def main():
    idx = np.load(OUT / 'grilla.npz')['region_idx']
    filas = []
    for met, nombre in (('mean', 'Duración media'), ('t99', 'Percentil 99')):
        z = np.load(OUT / f'bootstrap_por_celda_A_opcion_B_integrado_{met}.npz')
        val, corto, largo = z['valida'], z['mas_corto'], z['mas_largo']
        unidades = [('pais', 'Todo el país', val)]
        for m in MACROZONAS:
            unidades.append(('macrozona', m, val & np.isin(idx, [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == m])))
        for k, n in enumerate(NOMBRES_CORTOS):
            unidades.append(('region', n, val & (idx == k)))
        unidades.append(('Aysén', 'Aysén (fuera del análisis)', val & (idx < 0)))
        for nivel, u, sel in unidades:
            n = int(sel.sum())
            if n == 0:
                continue
            filas.append(dict(metrica=nombre, nivel=nivel, unidad=u, celdas=n,
                              pct_mas_corto=100 * corto[sel].mean(), pct_mas_largo=100 * largo[sel].mean(),
                              pct_sin_diferencia=100 * (1 - corto[sel].mean() - largo[sel].mean())))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / 'bootstrap_significancia_por_zona.csv', index=False)
    print(df[df.nivel.isin(['pais', 'macrozona'])].round(1).to_string(index=False))


if __name__ == '__main__':
    main()
