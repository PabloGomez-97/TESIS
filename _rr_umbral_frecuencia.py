"""Sensibilidad: umbral futuro R0_fut que CONSERVA la frecuencia de dias humedos del historico.

En el analisis principal el mismo R0 = 5,285 mm se usa en el historico y el futuro, y en el futuro la fraccion de
dias humedos baja (20,94 % -> 19,61 % en SSP5-8.5). Aqui se busca, por biseccion, el R0 del futuro tal que la
fraccion espacial media de dias con pr >= R0 sea igual a la del historico (mismo criterio i del paper), y se
repiten los risk ratios. Si el RR cae a ~1, el aumento de rachas largas se debe a la menor frecuencia de dias humedos.

Salidas: resultados_tesis/rachas_aladin_<escenario>_Rfrec.npz y resultados_tesis/umbral_frecuencia.csv
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from _extraer_rachas_dominio import open_aladin, a_celdas, extraer, HIST, FUT, R0

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'


def frac_humedos(pr, umbral):
    return float((pr >= umbral).mean(axis=0).mean() * 100)


def biseccion(pr, objetivo):
    lo, hi = 0.0, 40.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if frac_humedos(pr, mid) > objetivo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def por_region(pr, anios, meses, reg_celda, objetivos):
    """Recalibra el umbral futuro region por region: la frecuencia de dias humedos de cada region iguala la historica."""
    from geografia_tesis import NOMBRES_CORTOS
    cells, sy, sm, dur, umbrales = [], [], [], [], {}
    for k, nombre in enumerate(NOMBRES_CORTOS):
        cols = np.where(reg_celda == k)[0]
        r = biseccion(pr[:, cols], objetivos[k])
        umbrales[nombre] = r
        c, a, m, d = extraer(pr[:, cols], r, anios, meses)
        cells.append(cols[c].astype(np.int32)); sy.append(a); sm.append(m); dur.append(d)
    return np.concatenate(cells), np.concatenate(sy), np.concatenate(sm), np.concatenate(dur), umbrales


def main():
    g = np.load(OUT / 'grilla.npz')
    jj, ii = g['jj'], g['ii']
    da_h = open_aladin(HIST, None)
    objetivo = frac_humedos(a_celdas(da_h, jj, ii), R0)
    print(f'Frecuencia historica de dias humedos con R0={R0}: {objetivo:.3f} %', flush=True)
    filas, filas_reg = [], []
    reg_celda = g['region_idx'][jj, ii].astype(np.int64)
    pr_h = a_celdas(da_h, jj, ii)
    from geografia_tesis import NOMBRES_CORTOS
    obj_reg = [float((pr_h[:, reg_celda == k] >= R0).mean(axis=0).mean() * 100) for k in range(len(NOMBRES_CORTOS))]
    for esc in ('ssp585', 'ssp370'):
        da = open_aladin(FUT, esc)
        pr = a_celdas(da, jj, ii)
        f_fijo = frac_humedos(pr, R0)
        r_nuevo = biseccion(pr, objetivo)
        cell, sy, sm, dur = extraer(pr, r_nuevo, da['time'].dt.year.values, da['time'].dt.month.values)
        np.savez(OUT / f'rachas_aladin_{esc}_Rfrec.npz', cell=cell, start_year=sy, start_month=sm, duration=dur,
                 n_dias=pr.shape[0], umbral=r_nuevo)
        c2, a2, m2, d2, umb = por_region(pr, da['time'].dt.year.values, da['time'].dt.month.values, reg_celda, obj_reg)
        np.savez(OUT / f'rachas_aladin_{esc}_RfrecReg.npz', cell=c2, start_year=a2, start_month=m2, duration=d2,
                 n_dias=pr.shape[0], umbral=np.nan)
        for k, nombre in enumerate(NOMBRES_CORTOS):
            cols = reg_celda == k
            filas_reg.append(dict(escenario=esc, region=nombre, F_hist_pct=obj_reg[k],
                                  F_fut_con_R0_pct=float((pr[:, cols] >= R0).mean(axis=0).mean() * 100),
                                  R0_fut_que_conserva_F=umb[nombre]))
        filas.append(dict(escenario=esc, F_hist_pct=objetivo, F_fut_con_R0_pct=f_fijo, R0_hist=R0, R0_fut_que_conserva_F=r_nuevo))
        print(f'{esc}: F futura con R0 fijo = {f_fijo:.3f} % | R0 que conserva F = {r_nuevo:.3f} mm', flush=True)
    pd.DataFrame(filas).to_csv(OUT / 'umbral_frecuencia.csv', index=False)
    pd.DataFrame(filas_reg).to_csv(OUT / 'umbral_frecuencia_regiones.csv', index=False)
    print(pd.DataFrame(filas_reg).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
