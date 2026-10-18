"""Separa 'menos dias humedos' de 'mas agrupacion' en el aumento de rachas largas (referencia sin memoria larga).

Para cada celda y cada mes del anio se estima una cadena de Markov de 1er orden (dia seco/humedo) con
  p_dd = P(seco hoy | seco ayer)   y   p_wd = P(seco hoy | humedo ayer),
y se simulan series de 35 anios (R replicas por celda, con numeros aleatorios comunes entre escenarios):
  H   : parametros del historico (1980-2014)
  F   : parametros del futuro (2040-2074) -> cambian frecuencia y persistencia de 1er orden
  F0  : frecuencia seca del FUTURO por mes, pero persistencia (rho = p_dd - p_wd) del HISTORICO -> solo frecuencia
Con las mismas reglas de rachas (censuradas excluidas, inicio mar-nov), se obtienen N_H, N_F, N_F0 y
  RR_frec  = N_F0 / N_H      (cambio esperado si solo cambia la frecuencia de dias secos)
  RR_markov = N_F  / N_H      (frecuencia + persistencia de 1er orden)
y se comparan con el RR observado. Descomposicion multiplicativa:
  RR_obs = RR_frec x (RR_markov / RR_frec) x (RR_obs / RR_markov)
           frecuencia      persistencia 1er orden        residuo (memoria larga, variabilidad entre anios)
Es un diagnostico con estimaciones puntuales (no incluye la incertidumbre de estimar los parametros).
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE
from _extraer_rachas_dominio import open_aladin, a_celdas, HIST, FUT, R0

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'
DS = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
R = 30
CHUNK = 300
SEED = 7
NREG = len(NOMBRES_CORTOS)


def parametros(dry, meses):
    """dry (n, c) bool. Devuelve p_dd, p_wd y frecuencia seca por mes: arreglos (12, c)."""
    prev, cur, m = dry[:-1], dry[1:], meses[1:]
    c = dry.shape[1]
    p_dd, p_wd, frec = np.zeros((12, c)), np.zeros((12, c)), np.zeros((12, c))
    for k in range(1, 13):
        s = m == k
        p_, c_ = prev[s], cur[s]
        p_dd[k - 1] = ((p_ & c_).sum(0) + 0.5) / (p_.sum(0) + 1.0)
        p_wd[k - 1] = (((~p_) & c_).sum(0) + 0.5) / ((~p_).sum(0) + 1.0)
        frec[k - 1] = c_.mean(0)
    return p_dd, p_wd, frec


def parametros_solo_frecuencia(p_dd_h, p_wd_h, frec_f):
    """Mantiene la persistencia historica rho = p_dd - p_wd y fija la frecuencia seca futura."""
    rho = p_dd_h - p_wd_h
    p_wd = np.clip(frec_f * (1 - rho), 1e-6, 1 - 1e-6)
    p_dd = np.clip(p_wd + rho, 1e-6, 1 - 1e-6)
    return p_dd, p_wd


def conteos(dry, meses, reg_col):
    """Rachas no censuradas con inicio mar-nov y duracion >= D, por region. dry: (n, cols) bool."""
    n, c = dry.shape
    pad = np.zeros((c, n + 2), dtype=np.int8)
    pad[:, 1:-1] = dry.T
    dx = np.diff(pad.ravel())
    s = np.flatnonzero(dx == 1) + 1
    e = np.flatnonzero(dx == -1) + 1
    col, fila_s, fila_e = s // (n + 2), s % (n + 2) - 1, e % (n + 2) - 1
    dur = e - s
    ok = (fila_s > 0) & (fila_e < n)
    mes = meses[np.clip(fila_s, 0, n - 1)]
    ok &= (mes >= 3) & (mes <= 11)
    out = np.zeros((len(DS), NREG))
    for i, D in enumerate(DS):
        sel = ok & (dur >= D)
        out[i] = np.bincount(reg_col[col[sel]], minlength=NREG)
    return out


def main():
    g = np.load(OUT / 'grilla.npz')
    reg_celda = g['region_idx'][g['jj'], g['ii']].astype(np.int64)
    ref = open_aladin(HIST, None)
    da_h, da_f = ref, open_aladin(FUT, 'ssp585')
    jj, ii = g['jj'], g['ii']
    use = reg_celda >= 0
    jj, ii, reg = jj[use], ii[use], reg_celda[use]
    dry_h = a_celdas(da_h, jj, ii) < R0
    dry_f = a_celdas(da_f, jj, ii) < R0
    mh, mf = da_h['time'].dt.month.values, da_f['time'].dt.month.values
    n = dry_h.shape[0]
    assert dry_f.shape[0] == n

    obs_h = conteos(dry_h, mh, reg)
    obs_f = conteos(dry_f, mf, reg)
    print('Observado (D=20):', obs_h[DS.index(20)].astype(int), flush=True)

    pdd_h, pwd_h, fr_h = parametros(dry_h, mh)
    pdd_f, pwd_f, fr_f = parametros(dry_f, mf)
    pdd_0, pwd_0 = parametros_solo_frecuencia(pdd_h, pwd_h, fr_f)

    rng = np.random.default_rng(SEED)
    tot = {k: np.zeros((len(DS), NREG)) for k in ('H', 'F', 'F0')}
    nc = dry_h.shape[1]
    for a in range(0, nc, CHUNK):
        b = min(nc, a + CHUNK)
        w = b - a
        cols = w * R
        rep = lambda M: np.tile(M[:, a:b], (1, R))
        P = {'H': (rep(pdd_h), rep(pwd_h)), 'F': (rep(pdd_f), rep(pwd_f)), 'F0': (rep(pdd_0), rep(pwd_0))}
        fr0 = np.tile(fr_h[0, a:b], R)
        u0 = rng.random(cols)
        st = {k: u0 < fr0 for k in P}
        sim = {k: np.empty((n, cols), dtype=bool) for k in P}
        meses = {'H': mh, 'F': mf, 'F0': mf}   # F0 usa el calendario del futuro (mismo largo)
        for t in range(n):
            u = rng.random(cols)
            for k, (pd_, pw_) in P.items():
                m = meses[k][t] - 1
                p = np.where(st[k], pd_[m], pw_[m])
                st[k] = u < p
                sim[k][t] = st[k]
        reg_col = np.tile(reg[a:b], R)
        for k in tot:
            tot[k] += conteos(sim[k], meses[k], reg_col) / R
        print(f'  celdas {b}/{nc}', flush=True)

    filas = []
    def agrega(nivel, unidad, filas_reg):
        for i, D in enumerate(DS):
            f = lambda M: float(M[i][filas_reg].sum())
            Nh, Nf = f(obs_h), f(obs_f)
            mh_, mf_, m0_ = f(tot['H']), f(tot['F']), f(tot['F0'])
            dv = lambda a, b: a / b if b > 0 else np.nan
            rr_obs, rr_m, rr_fr = dv(Nf, Nh), dv(mf_, mh_), dv(m0_, mh_)
            filas.append(dict(nivel=nivel, unidad=unidad, D=D, N_hist_obs=Nh, N_fut_obs=Nf,
                              N_hist_markov=mh_, N_fut_markov=mf_, N_fut_solo_frec=m0_,
                              cobertura_markov_hist=dv(mh_, Nh), RR_obs=rr_obs, RR_markov=rr_m, RR_solo_frec=rr_fr,
                              f_persist=dv(rr_m, rr_fr), f_residuo=dv(rr_obs, rr_m)))
    for k, nom in enumerate(NOMBRES_CORTOS):
        agrega('region', nom, [k])
    for m in MACROZONAS:
        agrega('macrozona', m, [k for k, nn in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[nn] == m])
    agrega('dominio', '14 regiones', list(range(NREG)))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / 'descomposicion_markov_ssp585.csv', index=False)
    print(df[df.D == 20].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
