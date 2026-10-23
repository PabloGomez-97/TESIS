"""Contraste directo de si cambia la persistencia de 1.er orden de los dias secos entre 1980-2014 y 2040-2074 (SSP5-8.5).

Complementa _referencia_markov_rr_regiones.py. Aquel compara conteos de rachas largas con lo esperado solo por frecuencia;
este mide el parametro que describe el agrupamiento dia a dia, rho = P(seco hoy | seco ayer) - P(seco hoy | humedo ayer),
con su incertidumbre. Si rho no cambia, un aumento de rachas largas con menos dias humedos no requiere un cambio en el agrupamiento.

Por region (mismas mascaras y R0 = 5,285 mm que _run_pregunta5_aladin.py) y por celda y mes (marzo-noviembre, los meses de inicio
de racha de los risk ratios) se estiman p_dd, p_wd (con el suavizado +0,5/+1 de la referencia Markov), rho y la frecuencia seca.
El estadistico regional es el promedio simple sobre celdas y meses. IC95 por bootstrap de anios: 1 000 replicas, 35 anios con
reemplazo sorteados de forma independiente en cada periodo (igual que los risk ratios). Una transicion se asigna al anio y mes del
dia de llegada.

Salida: resultados_tesis/contraste_persistencia_regiones.csv
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd

import _run_pregunta5_aladin as rr
import _referencia_markov_rr_regiones as mk

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis' / 'contraste_persistencia_regiones.csv'
N_BOOT = 1000
SEED = 11
MESES = np.arange(3, 12)  # marzo-noviembre


def conteos_por_anio_mes(dry, tiempo):
    """Devuelve arreglo (anio, mes, celda, 5): [llegada seca, ayer seco, ayer seco y hoy seco, ayer humedo, ayer humedo y hoy seco]."""
    anios = tiempo.dt.year.values[1:]
    meses = tiempo.dt.month.values[1:]
    ya, hoy = dry[:-1], dry[1:]
    y0 = anios.min()
    out = np.zeros((anios.max() - y0 + 1, 12, dry.shape[1], 5))
    for y in np.unique(anios):
        for m in range(1, 13):
            s = (anios == y) & (meses == m)
            if not s.any():
                continue
            a, h = ya[s], hoy[s]
            out[y - y0, m - 1, :, 0] = h.sum(0)
            out[y - y0, m - 1, :, 1] = a.sum(0)
            out[y - y0, m - 1, :, 2] = (a & h).sum(0)
            out[y - y0, m - 1, :, 3] = (~a).sum(0)
            out[y - y0, m - 1, :, 4] = ((~a) & h).sum(0)
    return out


def estadisticos(c):
    """c: (meses, celdas, 5) acumulado sobre los anios elegidos. Devuelve frecuencia seca, p_dd, rho promediados sobre celdas y meses."""
    ndias = c[..., 1] + c[..., 3]
    frec = c[..., 0] / ndias
    pdd = (c[..., 2] + 0.5) / (c[..., 1] + 1.0)
    pwd = (c[..., 4] + 0.5) / (c[..., 3] + 1.0)
    return frec.mean(), pdd.mean(), (pdd - pwd).mean()


def main():
    ph = rr.open_aladin_period(rr.HIST_START, rr.HIST_END)
    pf = rr.open_aladin_period(rr.FUT_START, rr.FUT_END, scenario='ssp585')
    jj, ii, reg = mk.celdas_de_regiones(ph['lat'].values, ph['lon'].values)
    dry_h = mk.a_celdas(ph, jj, ii) < rr.DRY_THRESHOLD
    dry_f = mk.a_celdas(pf, jj, ii) < rr.DRY_THRESHOLD
    cnt_h = conteos_por_anio_mes(dry_h, ph['time'])[:, MESES - 1]
    cnt_f = conteos_por_anio_mes(dry_f, pf['time'])[:, MESES - 1]
    rng = np.random.default_rng(SEED)
    filas = []
    for k, nombre in enumerate(rr.REGION_SPECS):
        sel = reg == k
        ch, cf = cnt_h[:, :, sel, :], cnt_f[:, :, sel, :]
        h = estadisticos(ch.sum(0))
        f = estadisticos(cf.sum(0))
        boot = np.empty((N_BOOT, 3))
        for b in range(N_BOOT):
            ih = rng.integers(0, ch.shape[0], ch.shape[0])
            i_f = rng.integers(0, cf.shape[0], cf.shape[0])
            boot[b] = np.array(estadisticos(cf[i_f].sum(0))) - np.array(estadisticos(ch[ih].sum(0)))
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        fila = {'region': nombre, 'celdas': int(sel.sum()), 'meses': 'mar-nov'}
        for j, nom in enumerate(('frec_seca', 'p_dd', 'rho')):
            fila.update({f'{nom}_hist': h[j], f'{nom}_fut': f[j], f'{nom}_delta': f[j] - h[j],
                         f'{nom}_delta_IC95_inf': lo[j], f'{nom}_delta_IC95_sup': hi[j],
                         f'{nom}_IC_incluye_0': bool(lo[j] <= 0 <= hi[j])})
        filas.append(fila)
    df = pd.DataFrame(filas)
    OUT.parent.mkdir(exist_ok=True)
    df.round(5).to_csv(OUT, index=False)
    print(df.filter(regex='region|_hist|_fut|delta$|IC95|incluye').round(4).T.to_string())


if __name__ == '__main__':
    main()
