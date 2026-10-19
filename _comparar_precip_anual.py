"""Precipitacion anual media de ALADIN frente a CR2MET (1980-2014) por region y macrozona.

Mismo calculo que el resto: CR2MET interpolado linealmente a las celdas de ALADIN, media de la precipitacion diaria
por 365,25. Esto NO es la frecuencia de dias humedos (que se iguala con R0): es la cantidad total de lluvia.
Salida: resultados_tesis/precip_anual_aladin_vs_cr2met.csv
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


def main():
    g = np.load(OUT / 'grilla.npz')
    jj, ii = g['jj'], g['ii']
    ref = open_aladin(HIST, None)
    a = a_celdas(ref, jj, ii).mean(axis=0) * 365.25
    c = a_celdas(open_cr2met(HIST, ref['lat'], ref['lon']), jj, ii).mean(axis=0) * 365.25
    reg = g['region_idx'][jj, ii]
    filas = []
    for k, n in enumerate(NOMBRES_CORTOS):
        s = reg == k
        filas.append(dict(nivel='region', unidad=n, celdas=int(s.sum()), cr2met_mm=c[s].mean(), aladin_mm=a[s].mean()))
    for z in MACROZONAS:
        s = np.isin(reg, [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == z])
        filas.append(dict(nivel='macrozona', unidad=z, celdas=int(s.sum()), cr2met_mm=c[s].mean(), aladin_mm=a[s].mean()))
    s = reg >= 0
    filas.append(dict(nivel='dominio', unidad='14 regiones', celdas=int(s.sum()), cr2met_mm=c[s].mean(), aladin_mm=a[s].mean()))
    df = pd.DataFrame(filas)
    df['razon_aladin_cr2met'] = df.aladin_mm / df.cr2met_mm
    df.to_csv(OUT / 'precip_anual_aladin_vs_cr2met.csv', index=False)
    print(df.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
