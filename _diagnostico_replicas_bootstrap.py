"""Diagnostico: cuantas replicas del bootstrap por celda se descartan por tener menos de 30 rachas.

En _run_pregunta7_bootstrap.py cada replica remuestrea 35 anios y recalcula la media o el p99; si CR2MET o ALADIN
tienen menos de MIN_SPELLS_FOR_STATS rachas en esa replica, la metrica es NaN y la replica se elimina antes de
calcular el intervalo. Aqui se repite el bootstrap del criterio i (R0 = 5,285 mm en ALADIN, 1 mm en CR2MET) con las
mismas semillas (42 + indice de celda) y los mismos sorteos, contando las replicas descartadas por celda.

Validacion: delta, ic_low e ic_high recalculados deben coincidir con resultados_tesis/bootstrap_por_celda_A_*.npz.
Salidas: resultados_tesis/bootstrap_replicas_descartadas_celdas.csv (por celda) y
         resultados_tesis/bootstrap_replicas_descartadas_zonas.csv (resumen por macrozona).
No modifica ningun archivo existente.
"""
import warnings
warnings.filterwarnings('ignore')

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import _run_pregunta7_bootstrap as b
from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'


def replicas_una_celda(args):
    idx, g_cr2, g_ala, years_arr, metrica, n_iter, seed = args
    rng = np.random.default_rng(seed)
    nan_ = (np.nan, np.nan, np.nan, 0, 0, 0, 0)
    if not g_cr2 or not g_ala:
        return idx, nan_
    todos = [y for y in years_arr if y in g_cr2 or y in g_ala]
    m_c = b.spell_metric_from_durations(b.pool_spells_by_years(g_cr2, todos), metrica)
    m_a = b.spell_metric_from_durations(b.pool_spells_by_years(g_ala, todos), metrica)
    if not (np.isfinite(m_c) and np.isfinite(m_a)):
        return idx, nan_
    obs = m_a - m_c
    n_years = years_arr.size
    boots = np.empty(n_iter)
    nan_c = nan_a = 0
    for i in range(n_iter):
        dc = rng.choice(years_arr, size=n_years, replace=True)
        da = rng.choice(years_arr, size=n_years, replace=True)
        bc = b.spell_metric_from_durations(b.pool_spells_by_years(g_cr2, dc), metrica)
        ba = b.spell_metric_from_durations(b.pool_spells_by_years(g_ala, da), metrica)
        nan_c += not np.isfinite(bc)
        nan_a += not np.isfinite(ba)
        boots[i] = ba - bc if np.isfinite(bc) and np.isfinite(ba) else np.nan
    ok = np.isfinite(boots)
    if ok.sum() < 50:
        return idx, (obs, np.nan, np.nan, int(n_iter - ok.sum()), nan_c, nan_a, int(ok.sum()))
    lo, hi = np.percentile(boots[ok], [100 * b.ALPHA / 2, 100 * (1 - b.ALPHA / 2)])
    return idx, (obs, lo, hi, int(n_iter - ok.sum()), nan_c, nan_a, int(ok.sum()))


def main():
    print('Cargando datos...', flush=True)
    pr_ala = b.open_aladin_historical()
    pr_cr2_nat = b.open_cr2met_historical()
    geom = b.load_chile_geometry()
    m_bool = b.build_chile_mask_on_aladin_grid(pr_ala['lat'].values, pr_ala['lon'].values, geom)
    chile_mask = xr.DataArray(m_bool, coords={'y': pr_ala['y'], 'x': pr_ala['x'], 'lat': pr_ala['lat'],
                                              'lon': pr_ala['lon']}, dims=['y', 'x'])
    pr_cr2 = b.regrid_cr2met_to_aladin(pr_cr2_nat, pr_ala)
    pr_cr2, pr_ala = xr.align(b.normalize_daily_time(pr_cr2), b.normalize_daily_time(pr_ala), join='inner')
    pr_cr2, pr_ala = pr_cr2.where(chile_mask), pr_ala.where(chile_mask)
    mask_stacked = chile_mask.stack(cell=('y', 'x')).values
    years = np.unique(b.years_from_time(pr_ala['time']))
    ny, nx = m_bool.shape

    print('Extrayendo rachas...', flush=True)
    g_cr2, _ = b.extract_spell_groups_by_year(pr_cr2, b.TAU_CR2MET_REF, chile_mask)
    g_ala, _ = b.extract_spell_groups_by_year(pr_ala, b.TAU_ALADIN_DOMINIO, chile_mask)

    reg = np.load(OUT / 'grilla.npz')['region_idx'].ravel()
    filas = []
    for metrica in ('mean', 't99'):
        print(f'Bootstrap con conteo: {metrica}...', flush=True)
        tareas = [(i, g_cr2[i], g_ala[i], years, metrica, b.BOOTSTRAP_ITER, b.RANDOM_SEED + i)
                  for i in range(len(g_cr2)) if mask_stacked[i]]
        res = {}
        with ProcessPoolExecutor(max_workers=b.N_WORKERS) as pool:
            for fut in as_completed([pool.submit(replicas_una_celda, t) for t in tareas]):
                i, r = fut.result()
                res[i] = r
        ref = np.load(OUT / f'bootstrap_por_celda_A_opcion_B_integrado_{metrica}.npz')
        d_ref, lo_ref, hi_ref = (ref[k].ravel() for k in ('delta', 'ic_low', 'ic_high'))
        dif = []
        for i, (obs, lo, hi, desc, nc, na, validas) in res.items():
            if np.isfinite(obs):
                dif.append(max(abs(obs - d_ref[i]),
                               abs(lo - lo_ref[i]) if np.isfinite(lo) and np.isfinite(lo_ref[i]) else 0,
                               abs(hi - hi_ref[i]) if np.isfinite(hi) and np.isfinite(hi_ref[i]) else 0))
            filas.append(dict(metrica=metrica, celda=i, region_idx=int(reg[i]), obs=obs, descartadas=desc,
                              descartadas_cr2met=nc, descartadas_aladin=na, validas=validas))
        print(f'   max |diferencia| contra el bootstrap guardado ({metrica}): {max(dif):.2e}', flush=True)

    df = pd.DataFrame(filas)
    df['n_replicas'] = b.BOOTSTRAP_ITER
    df['pct_descartadas'] = 100 * df.descartadas / df.n_replicas
    df['zona'] = df.region_idx.map(lambda k: MACROZONA_DE[NOMBRES_CORTOS[k]] if k >= 0 else 'fuera de las 14 regiones')
    df['region'] = df.region_idx.map(lambda k: NOMBRES_CORTOS[k] if k >= 0 else '')
    df.to_csv(OUT / 'bootstrap_replicas_descartadas_celdas.csv', index=False)

    d14 = df[(df.region_idx >= 0) & np.isfinite(df.obs)]
    def resumen(g):
        return pd.Series(dict(
            celdas=len(g), celdas_con_descartes=int((g.descartadas > 0).sum()),
            pct_celdas_con_descartes=100 * (g.descartadas > 0).mean(),
            celdas_mas_5pct=int((g.pct_descartadas > 5).sum()), celdas_mas_20pct=int((g.pct_descartadas > 20).sum()),
            pct_descartadas_media=g.pct_descartadas.mean(), pct_descartadas_max=g.pct_descartadas.max(),
            celdas_ic_no_calculable=int((g.validas < 50).sum())))
    zonas = pd.concat([d14.groupby(['metrica', 'zona']).apply(resumen).reset_index(),
                       d14.groupby('metrica').apply(resumen).reset_index().assign(zona='Las 14 regiones')])
    zonas['zona'] = pd.Categorical(zonas.zona, MACROZONAS + ['Las 14 regiones'], ordered=True)
    zonas = zonas.sort_values(['metrica', 'zona'])
    zonas.to_csv(OUT / 'bootstrap_replicas_descartadas_zonas.csv', index=False)
    print(zonas.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
