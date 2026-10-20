"""Sensibilidad al regrillado de CR2MET (lineal vs promedio por area) con la geografia de la tesis.

Usa la cache de _test_regrillado_conservativo.py (_test_regrillado/cr2met_lineal_vs_area.npz).
Salidas:
- resultados_tesis/regrillado_sesgo_14regiones.csv: sesgo de ALADIN (criterios i y ii) en las celdas de las 14 regiones
  con al menos 30 rachas en ambos productos.
- resultados_tesis/regrillado_cr2met_macrozonas.csv: media, p90, p99 y maximo de CR2MET (1 mm) por macrozona.
- resultados_tesis/regrillado_celdas_por_punto.csv: numero de puntos de grilla de CR2MET dentro de cada celda de ALADIN.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from _test_regrillado_conservativo import spells_per_cell, metrics
from _extraer_rachas_dominio import open_aladin, a_celdas, HIST, R0
from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE

OUT = 'resultados_tesis/'


def main():
    z = np.load('_test_regrillado/cr2met_lineal_vs_area.npz')
    lin, area, n_in = z['lin'], z['area'], z['n_inside']
    g = np.load(OUT + 'grilla.npz')
    jj, ii = g['jj'], g['ii']
    reg = g['region_idx'][jj, ii]
    use = reg >= 0
    pd.DataFrame([dict(mediana=np.median(n_in), minimo=n_in.min(), maximo=n_in.max())]).to_csv(
        OUT + 'regrillado_celdas_por_punto.csv', index=False)

    ala = a_celdas(open_aladin(HIST, None), jj, ii).astype(np.float32)
    mala = {'i': metrics(spells_per_cell(ala, R0)), 'ii': metrics(spells_per_cell(ala, 1.0))}
    filas, filas_m = [], []
    for nombre, cr in (('lineal', lin), ('area', area)):
        durs = spells_per_cell(cr, 1.0)
        mc, tc = metrics(durs)
        for crit in ('i', 'ii'):
            ma, ta = mala[crit]
            for met, a, c in (('media', ma, mc), ('p99', ta, tc)):
                ok = use & np.isfinite(a) & np.isfinite(c)
                d = a[ok] - c[ok]
                filas.append(dict(cr2met=nombre, criterio=crit, metrica=met, CR2MET=c[ok].mean(), ALADIN=a[ok].mean(),
                                  Delta_medio=d.mean(), Delta_mediano=np.median(d), celdas=int(ok.sum())))
        for m in MACROZONAS:
            ks = [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == m]
            celdas = np.where(np.isin(reg, ks))[0]
            d = np.concatenate([durs[c] for c in celdas])
            filas_m.append(dict(macrozona=m, metodo=nombre, n=d.size, media=d.mean(), p90=np.percentile(d, 90),
                                p99=np.percentile(d, 99), maximo=d.max()))
    df, dfm = pd.DataFrame(filas), pd.DataFrame(filas_m)
    df.to_csv(OUT + 'regrillado_sesgo_14regiones.csv', index=False)
    dfm.to_csv(OUT + 'regrillado_cr2met_macrozonas.csv', index=False)
    w = df.pivot_table(index=['criterio', 'metrica'], columns='cr2met', values='Delta_medio')
    w['area_menos_lineal'] = w['area'] - w['lineal']
    print(w.round(2))
    wm = dfm.pivot_table(index='macrozona', columns='metodo', values=['media', 'p90', 'p99'])
    print((wm.xs('area', axis=1, level=1) - wm.xs('lineal', axis=1, level=1)).round(2))


if __name__ == '__main__':
    main()
