"""Robustez de la referencia sin memoria larga: cadenas de Markov de orden 1, 2 y 3.

El estado es el conjunto de los ultimos k dias (secos/humedos); para cada celda y mes se estima
P(seco hoy | estado, mes) con suavizado jerarquico (se encoge hacia la probabilidad de orden k-1).
Tres mundos, con numeros aleatorios comunes:
  H  : parametros del historico
  F  : parametros del futuro (cambian frecuencia y dependencia)
  F0 : dependencia del HISTORICO (razones de momios entre estados) pero con la frecuencia seca mensual del FUTURO.
       Se logra desplazando el logit de todas las probabilidades del mes en una constante delta, elegida para que la
       frecuencia estacionaria seca de la cadena iguale la frecuencia futura de ese mes.
Mismas reglas de rachas que el resto (censuradas excluidas, inicio mar-nov). Salida: RR esperado en cada mundo
frente al RR observado y la cobertura del historico (N_markov / N_observado).
Uso: python _referencia_markov_orden_k.py 1 2 3
"""
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE
from _extraer_rachas_dominio import open_aladin, a_celdas, HIST, FUT, R0
from _referencia_markov_rr import conteos, DS, NREG

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'
R = 30
CHUNK = 300
SEED = 7
A_SUAVIZADO = 4.0


def parametros_k(dry, meses, k):
    """P(seco | estado, mes): arreglo (12, 2^k, c). Suavizado hacia el orden k-1."""
    n, c = dry.shape
    P_prev = None
    for kk in range(k + 1):
        S = 1 << kk
        s = np.zeros((n - kk, c), dtype=np.int32)
        for j in range(1, kk + 1):
            s += dry[kk - j:n - j].astype(np.int32) << (j - 1)
        mes = (meses[kk:] - 1).astype(np.int32)[:, None]
        celda = np.arange(c, dtype=np.int32)[None, :]
        key = ((celda * 12 + mes) * S + s).ravel()
        tot = np.bincount(key, minlength=c * 12 * S).reshape(c, 12, S).transpose(1, 2, 0)
        seco = np.bincount(key, weights=dry[kk:].ravel().astype(np.float64), minlength=c * 12 * S).reshape(c, 12, S).transpose(1, 2, 0)
        if kk == 0:
            P = (seco + 0.5) / (tot + 1.0)
        else:
            padre = np.arange(S) & ((1 << (kk - 1)) - 1)
            prior = P_prev[:, padre, :]
            P = (seco + A_SUAVIZADO * prior) / (tot + A_SUAVIZADO)
        P_prev = P
    return np.clip(P_prev, 1e-6, 1 - 1e-6)


def frecuencia_estacionaria(P, k):
    """Fraccion estacionaria de dias secos por (mes, celda) de la cadena de orden k con probabilidades P (12, S, c)."""
    S = 1 << k
    m, _, c = P.shape
    T = np.zeros((m, c, S, S))
    for s in range(S):
        for d in (0, 1):
            s2 = ((s << 1) | d) & (S - 1)
            T[:, :, s, s2] += (P[:, s, :] if d == 1 else 1 - P[:, s, :])
    A = np.swapaxes(T, -1, -2) - np.eye(S)
    A[..., -1, :] = 1.0
    b = np.zeros((m, c, S)); b[..., -1] = 1.0
    pi = np.linalg.solve(A, b[..., None])[..., 0]
    bit0 = np.array([(s & 1) == 1 for s in range(S)])
    return pi[..., bit0].sum(axis=-1)


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def parametros_solo_frecuencia_k(P_h, fr_f, k):
    lo = np.full(fr_f.shape, -9.0); hi = np.full(fr_f.shape, 9.0)
    L = logit(P_h)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        f = frecuencia_estacionaria(expit(L + mid[:, None, :]), k)
        sube = f < fr_f
        lo = np.where(sube, mid, lo); hi = np.where(sube, hi, mid)
    return np.clip(expit(L + (0.5 * (lo + hi))[:, None, :]), 1e-6, 1 - 1e-6)


def correr(k):
    g = np.load(OUT / 'grilla.npz')
    reg_celda = g['region_idx'][g['jj'], g['ii']].astype(np.int64)
    use = reg_celda >= 0
    jj, ii, reg = g['jj'][use], g['ii'][use], reg_celda[use]
    da_h, da_f = open_aladin(HIST, None), open_aladin(FUT, 'ssp585')
    dry_h, dry_f = a_celdas(da_h, jj, ii) < R0, a_celdas(da_f, jj, ii) < R0
    mh, mf = da_h['time'].dt.month.values, da_f['time'].dt.month.values
    n, nc = dry_h.shape
    obs_h, obs_f = conteos(dry_h, mh, reg), conteos(dry_f, mf, reg)

    P_h, P_f = parametros_k(dry_h, mh, k), parametros_k(dry_f, mf, k)
    fr_f = np.stack([dry_f[mf == m].mean(0) for m in range(1, 13)])
    P_0 = parametros_solo_frecuencia_k(P_h, fr_f, k)
    print(f'orden {k}: parametros listos', flush=True)

    S = 1 << k
    mask = S - 1
    rng = np.random.default_rng(SEED)
    tot = {q: np.zeros((len(DS), NREG)) for q in ('H', 'F', 'F0')}
    for a in range(0, nc, CHUNK):
        b = min(nc, a + CHUNK)
        cols = (b - a) * R
        ar = np.arange(cols)
        P = {q: np.tile(M[:, :, a:b], (1, 1, R)) for q, M in (('H', P_h), ('F', P_f), ('F0', P_0))}
        meses = {'H': mh, 'F': mf, 'F0': mf}
        fr0 = np.tile(np.stack([dry_h[mh == 1, a:b].mean(0)]), (1, R))[0]
        st = {q: np.zeros(cols, dtype=np.int64) for q in P}
        for q in P:
            for j in range(k):
                st[q] |= ((rng.random(cols) < fr0).astype(np.int64) << j)
        sim = {q: np.empty((n, cols), dtype=bool) for q in P}
        for t in range(n):
            u = rng.random(cols)
            for q in P:
                p = P[q][meses[q][t] - 1][st[q], ar]
                d = u < p
                st[q] = ((st[q] << 1) | d.astype(np.int64)) & mask
                sim[q][t] = d
        reg_col = np.tile(reg[a:b], R)
        for q in tot:
            tot[q] += conteos(sim[q], meses[q], reg_col) / R
        print(f'  orden {k}: celdas {b}/{nc}', flush=True)

    filas = []
    def agrega(nivel, unidad, regs):
        i = DS.index(20)
        f = lambda M: float(M[i][regs].sum())
        Nh, Nf, Mh, Mf, M0 = f(obs_h), f(obs_f), f(tot['H']), f(tot['F']), f(tot['F0'])
        filas.append(dict(orden=k, nivel=nivel, unidad=unidad, N_hist_obs=Nh, N_fut_obs=Nf,
                          cobertura_markov_hist=Mh / Nh, RR_obs=Nf / Nh, RR_markov=Mf / Mh, RR_solo_frec=M0 / Mh))
    for j, nom in enumerate(NOMBRES_CORTOS):
        agrega('region', nom, [j])
    for z in MACROZONAS:
        agrega('macrozona', z, [j for j, nn in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[nn] == z])
    agrega('dominio', '14 regiones', list(range(NREG)))
    return pd.DataFrame(filas)


if __name__ == '__main__':
    ordenes = [int(x) for x in sys.argv[1:]] or [1, 2, 3]
    dfs = []
    for k in ordenes:
        df = correr(k)
        dfs.append(df)
        print(df.round(3).to_string(index=False), flush=True)
        pd.concat(dfs).to_csv(OUT / 'descomposicion_markov_orden_k.csv', index=False)
