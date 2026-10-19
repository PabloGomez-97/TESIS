"""Fraccion de la precipitacion anual que cae entre marzo y noviembre, por region y macrozona (CR2MET y ALADIN, 1980-2014).

Justifica (o cuestiona) restringir los risk ratios a rachas que empiezan entre marzo y noviembre. Se calcula la
precipitacion mensual media por celda, se suma por region y se obtiene la fraccion mar-nov de la precipitacion anual.
Salida: resultados_tesis/estacionalidad_marnov.csv
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE
from _extraer_rachas_dominio import open_aladin, open_cr2met, a_celdas, HIST

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'


def mensual(da, jj, ii):
    pr = a_celdas(da, jj, ii)
    mes = da['time'].dt.month.values
    return np.stack([pr[mes == m].mean(axis=0) * 30.4375 for m in range(1, 13)])  # (12, celdas) mm/mes


def main():
    g = np.load(OUT / 'grilla.npz')
    jj, ii = g['jj'], g['ii']
    reg = g['region_idx'][jj, ii]
    ref = open_aladin(HIST, None)
    prod = {'CR2MET': mensual(open_cr2met(HIST, ref['lat'], ref['lon']), jj, ii), 'ALADIN': mensual(ref, jj, ii)}
    filas = []
    unidades = [('region', n, reg == k) for k, n in enumerate(NOMBRES_CORTOS)]
    unidades += [('macrozona', z, np.isin(reg, [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == z])) for z in MACROZONAS]
    unidades += [('dominio', '14 regiones', reg >= 0)]
    for nivel, u, sel in unidades:
        fila = dict(nivel=nivel, unidad=u)
        for nom, M in prod.items():
            tot = M[:, sel].mean(axis=1)
            fila[f'{nom}_anual_mm'] = tot.sum()
            fila[f'{nom}_marnov_pct'] = 100 * tot[2:11].sum() / tot.sum()
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / 'estacionalidad_marnov.csv', index=False)
    print(df.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
