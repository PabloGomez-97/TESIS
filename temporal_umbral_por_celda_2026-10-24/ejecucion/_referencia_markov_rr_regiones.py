"""Separa 'menos dias humedos' de 'mas agrupacion' en el aumento de rachas largas, en las 4 regiones de los RR.

Referencia sin memoria larga. Para cada celda de la region y cada mes del anio se estima una cadena de Markov de
1.er orden (dia seco/humedo) con p_dd = P(seco hoy | seco ayer) y p_wd = P(seco hoy | humedo ayer), y se simulan series
de 35 anios (R replicas por celda, con numeros aleatorios comunes entre escenarios):
  H  : parametros del historico (1980-2014)
  F  : parametros del futuro (2040-2074, SSP5-8.5): cambian frecuencia y persistencia de 1.er orden
  F0 : frecuencia seca del FUTURO por mes, con la persistencia (rho = p_dd - p_wd) del HISTORICO: solo frecuencia
Con las mismas reglas de rachas que _run_pregunta5_aladin.py (umbral histórico local por celda; censuradas excluidas; inicio mar-nov)
se obtienen N_H, N_F, N_F0 y
  RR_frec   = N_F0 / N_H   (cambio esperado si solo cambia la frecuencia de dias secos)
  RR_markov = N_F  / N_H   (frecuencia + persistencia de 1.er orden)
  RR_obs    = N_fut_obs / N_hist_obs (mismos conteos que results.json)
Descomposicion multiplicativa: RR_obs = RR_frec x (RR_markov / RR_frec) x (RR_obs / RR_markov).
Es un diagnostico con estimaciones puntuales; el sd entre replicas mide solo el ruido de la simulacion, no la
incertidumbre de estimar los parametros ni la variabilidad interanual observada.

Entrada: pr1/ (ALADIN CHP12) y outputs_pregunta5_aladin/results.json (IC95 del RR observado). Salida: resultados_tesis/descomposicion_markov_rr_regiones.csv
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import json

import numpy as np
import pandas as pd
import cartopy.io.shapereader as shpreader

import _run_pregunta5_aladin as rr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis' / 'descomposicion_markov_rr_regiones.csv'
R0 = rr.DRY_THRESHOLD
DS = rr.THRESHOLD_SWEEP
REGIONES = list(rr.REGION_SPECS)
NREG = len(REGIONES)
R = 40
CHUNK = 200
SEED = 7
MES_MIN, MES_MAX = rr.SEASON_MONTH_MIN, rr.SEASON_MONTH_MAX


def celdas_de_regiones(lat2d, lon2d):
    reader = shpreader.Reader(shpreader.natural_earth(resolution='10m', category='cultural',
                                                      name='admin_1_states_provinces'))
    chile = [r for r in reader.records() if r.attributes.get('admin') == 'Chile']
    idx = np.full(lat2d.shape, -1, dtype=np.int16)
    for k, nombre in enumerate(REGIONES):
        rec = rr.get_region_record(rr.REGION_SPECS[nombre]['query'], chile)
        m = rr.build_region_mask(lat2d, lon2d, rec.geometry)
        assert not np.any(m & (idx >= 0)), f'{nombre} se solapa con otra region'
        idx[m] = k
    jj, ii = np.where(idx >= 0)
    return jj, ii, idx[jj, ii].astype(np.int64)


def a_celdas(da, jj, ii):
    import xarray as xr
    return da.isel(y=xr.DataArray(jj, dims='p'), x=xr.DataArray(ii, dims='p')).compute().values


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


def conteos(dry, meses, grupo, ngrupo):
    """Rachas no censuradas con inicio mar-nov y duracion >= D, por grupo. dry: (n, cols) bool."""
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
    ok &= (mes >= MES_MIN) & (mes <= MES_MAX)
    out = np.zeros((len(DS), ngrupo))
    for i, D in enumerate(DS):
        sel = ok & (dur >= D)
        out[i] = np.bincount(grupo[col[sel]], minlength=ngrupo)
    return out


def main():
    ph = rr.open_aladin_period(rr.HIST_START, rr.HIST_END)
    pf = rr.open_aladin_period(rr.FUT_START, rr.FUT_END, scenario='ssp585')
    jj, ii, reg = celdas_de_regiones(ph['lat'].values, ph['lon'].values)
    print('Celdas por region:', {n: int((reg == k).sum()) for k, n in enumerate(REGIONES)}, flush=True)
    dry_h = a_celdas(ph, jj, ii) < R0.values[jj, ii]
    dry_f = a_celdas(pf, jj, ii) < R0.values[jj, ii]
    mh, mf = ph['time'].dt.month.values, pf['time'].dt.month.values
    n = dry_h.shape[0]
    assert dry_f.shape[0] == n

    obs_h = conteos(dry_h, mh, reg, NREG)
    obs_f = conteos(dry_f, mf, reg, NREG)
    pdd_h, pwd_h, fr_h = parametros(dry_h, mh)
    pdd_f, pwd_f, fr_f = parametros(dry_f, mf)
    pdd_0, pwd_0 = parametros_solo_frecuencia(pdd_h, pwd_h, fr_f)

    rng = np.random.default_rng(SEED)
    nc = dry_h.shape[1]
    # tot[k][replica, D, region]
    tot = {k: np.zeros((R, len(DS), NREG)) for k in ('H', 'F', 'F0')}
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
        meses = {'H': mh, 'F': mf, 'F0': mf}
        for t in range(n):
            u = rng.random(cols)
            for k, (pd_, pw_) in P.items():
                m = meses[k][t] - 1
                p = np.where(st[k], pd_[m], pw_[m])
                st[k] = u < p
                sim[k][t] = st[k]
        reg_col = np.tile(reg[a:b], R)
        rep_col = np.repeat(np.arange(R), w)
        grupo = rep_col * NREG + reg_col
        for k in tot:
            tot[k] += conteos(sim[k], meses[k], grupo, R * NREG).reshape(len(DS), R, NREG).transpose(1, 0, 2)
        print(f'  celdas {b}/{nc}', flush=True)

    dv = lambda x, y: np.where(y > 0, x / np.where(y > 0, y, 1), np.nan)
    filas = []
    for k, nom in enumerate(REGIONES):
        for i, D in enumerate(DS):
            Nh, Nf = obs_h[i, k], obs_f[i, k]
            H, F, F0 = tot['H'][:, i, k], tot['F'][:, i, k], tot['F0'][:, i, k]
            rr_m, rr_fr = dv(F, H), dv(F0, H)
            rr_obs = Nf / Nh if Nh > 0 else np.nan
            filas.append(dict(
                region=nom, D=D, N_hist_obs=int(Nh), N_fut_obs=int(Nf),
                N_hist_markov=H.mean(), N_fut_markov=F.mean(), N_fut_solo_frec=F0.mean(),
                cobertura_markov_hist=H.mean() / Nh if Nh > 0 else np.nan,
                RR_obs=rr_obs, RR_markov=np.nanmean(rr_m), RR_markov_sd=np.nanstd(rr_m),
                RR_solo_frec=np.nanmean(rr_fr), RR_solo_frec_sd=np.nanstd(rr_fr),
                f_persist=np.nanmean(rr_m) / np.nanmean(rr_fr), f_residuo=rr_obs / np.nanmean(rr_m),
            ))
    df = pd.DataFrame(filas)
    # IC95 bootstrap (por anios) del RR observado, del mismo analisis de risk ratios (_run_pregunta5_aladin.py)
    curvas = json.load(open(ROOT / 'outputs_pregunta5_aladin' / 'results.json', encoding='utf-8'))['curves']
    ic = {(reg_, c['Umbral (dias)']): (c['IC95 inferior'], c['IC95 superior']) for reg_, cs in curvas.items() for c in cs}
    df['IC95_obs_inf'] = [ic.get((r_, d_), (np.nan, np.nan))[0] for r_, d_ in zip(df.region, df.D)]
    df['IC95_obs_sup'] = [ic.get((r_, d_), (np.nan, np.nan))[1] for r_, d_ in zip(df.region, df.D)]
    df['solo_frec_dentro_IC95_obs'] = (df.RR_solo_frec >= df.IC95_obs_inf) & (df.RR_solo_frec <= df.IC95_obs_sup)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df[df.D == 20].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
