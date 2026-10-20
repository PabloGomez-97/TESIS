"""Fraccion espacial media de dias humedos en 1980-2014 para respaldar la calibracion del umbral (Seccion 3.4).

Calcula, para el conjunto de celdas de Chile de la grilla de ALADIN (mascara de grilla.npz) y para las celdas de las
14 regiones, la fraccion de dias con pr >= umbral promediada sobre celdas:
  CR2MET (interpolado linealmente a las celdas) con 1 mm; ALADIN con 1 mm; ALADIN con R0 = 5,285 mm.
Reemplaza la constante escrita a mano (F_ALA_1 = 30.68) de _generar_fig_umbral_1mm_vs_R0.py por una salida trazable.

Salida: resultados_tesis/frecuencia_dias_humedos_hist.csv
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd

from _extraer_rachas_dominio import open_aladin, open_cr2met, a_celdas, HIST, R0, R_CR2MET

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'


def frac(pr, umbral, sel):
    return float((pr[:, sel] >= umbral).mean(axis=0).mean() * 100)


def main():
    g = np.load(OUT / 'grilla.npz')
    jj, ii = g['jj'], g['ii']
    en_14 = g['region_idx'][jj, ii] >= 0
    todas = np.ones(jj.size, dtype=bool)
    da_a = open_aladin(HIST, None)
    pr_a = a_celdas(da_a, jj, ii)
    pr_c = a_celdas(open_cr2met(HIST, da_a['lat'], da_a['lon']), jj, ii)
    assert not np.isnan(pr_c).any() and not np.isnan(pr_a).any()
    filas = []
    for dominio, sel in (('celdas de Chile', todas), ('14 regiones', en_14)):
        for producto, pr, umbral in (('CR2MET', pr_c, R_CR2MET), ('ALADIN', pr_a, R_CR2MET), ('ALADIN', pr_a, R0)):
            filas.append(dict(dominio=dominio, celdas=int(sel.sum()), producto=producto, umbral_mm=umbral,
                              F_dias_humedos_pct=frac(pr, umbral, sel)))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / 'frecuencia_dias_humedos_hist.csv', index=False)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
