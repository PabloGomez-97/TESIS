"""Ejecuta bootstrap de pregunta7 (dry spells CR2MET vs ALADIN) con paralelizacion."""
import warnings
warnings.filterwarnings('ignore')

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
import cartopy.io.shapereader as shpreader

START_DATE = '1980-01-01'
END_DATE = '2014-12-31'
TAU_CR2MET_REF = 1.0
TAU_ALADIN_DOMINIO = 5.285
BISECTION_TAU_MAX = 15.0
BISECTION_TOL = 1e-4
MIN_SPELLS_FOR_STATS = 30
DOMAIN_MASK_RESOLUTION = '10m'
BOOTSTRAP_ITER = 1000
RANDOM_SEED = 42
ALPHA = 0.05
N_WORKERS = 4
BOOTSTRAP_METRICS = [('mean', 'Duracion media'), ('t99', 't99 (p99)')]
OUT_DIR = Path('outputs_pregunta7_bootstrap')


def load_chile_geometry():
    shp_path = shpreader.natural_earth(
        resolution=DOMAIN_MASK_RESOLUTION, category='cultural', name='admin_0_countries',
    )
    geoms = [rec.geometry for rec in shpreader.Reader(shp_path).records() if rec.attributes['NAME'] == 'Chile']
    return unary_union(geoms)


def build_chile_mask_on_aladin_grid(lat2d, lon2d, geometry):
    prep_geom = prep(geometry)
    ny, nx = lat2d.shape
    mask = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        for i in range(nx):
            if prep_geom.contains(Point(float(lon2d[j, i]), float(lat2d[j, i]))):
                mask[j, i] = True
    return mask


def open_aladin_historical():
    ds = xr.open_mfdataset('./pr1/pr_CHP12_*_historical_*.nc', use_cftime=True, chunks={'time': 365})
    return ds['pr'].sel(time=slice(START_DATE, END_DATE)) * 86400.0


def open_cr2met_historical():
    ds = xr.open_mfdataset('./pr/CR2MET_pr_v2.5_day_*.nc', chunks={'time': 365})
    return ds['pr'].sel(time=slice(START_DATE, END_DATE))


def regrid_cr2met_to_aladin(pr_cr2met, pr_aladin_template):
    return pr_cr2met.interp(lat=pr_aladin_template['lat'], lon=pr_aladin_template['lon'], method='linear')


def normalize_daily_time(da):
    dates = pd.DatetimeIndex([pd.Timestamp(str(t)[:10]) for t in da['time'].values])
    return da.assign_coords(time=dates).groupby('time').mean()


def wet_fraction_1d(pr_series, threshold):
    x = np.asarray(pr_series, dtype=float)
    x = x[np.isfinite(x)]
    return np.nan if x.size == 0 else float((x >= threshold).mean())


def find_threshold_for_target_1d(pr_series, target_fraction, tau_max=BISECTION_TAU_MAX, tol=BISECTION_TOL):
    if not np.isfinite(target_fraction):
        return np.nan
    f0 = wet_fraction_1d(pr_series, 0.0)
    fmax = wet_fraction_1d(pr_series, tau_max)
    if target_fraction > f0 + tol or target_fraction < fmax - tol:
        return np.nan
    lo, hi = 0.0, tau_max
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = wet_fraction_1d(pr_series, mid)
        if abs(fmid - target_fraction) <= tol:
            return mid
        lo, hi = (mid, hi) if fmid > target_fraction else (lo, mid)
    return 0.5 * (lo + hi)


def pixelwise_local_aladin_threshold(pr_cr2, pr_aladin, tau_cr2, mask_da):
    f_target = (pr_cr2 >= tau_cr2).mean(dim='time').where(mask_da)
    stacked = xr.Dataset({
        'pr_ala': pr_aladin.stack(cell=('y', 'x')),
        'f_target': f_target.stack(cell=('y', 'x')),
    }).compute()
    n_cells = stacked.sizes['cell']
    tau_star = np.full(n_cells, np.nan, dtype=np.float32)
    for idx in range(n_cells):
        ft = float(stacked['f_target'].values[idx])
        if np.isfinite(ft):
            tau_star[idx] = find_threshold_for_target_1d(stacked['pr_ala'].values[:, idx], ft)
    tau_da = xr.DataArray(tau_star, coords={'cell': stacked['cell']}, dims=['cell']).unstack('cell')
    return tau_da.assign_coords(lat=mask_da['lat'], lon=mask_da['lon'])


def time_to_year(t):
    if hasattr(t, 'year'):
        return int(t.year)
    return int(pd.Timestamp(str(t)[:10]).year)


def years_from_time(time_coord):
    tvals = time_coord.values
    if hasattr(tvals[0], 'year'):
        return np.array([int(t.year) for t in tvals], dtype=int)
    return pd.to_datetime(tvals).year.to_numpy(dtype=int)


def spell_metric_from_durations(durations, metric_name):
    d = np.asarray(durations, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size < MIN_SPELLS_FOR_STATS:
        return np.nan
    if metric_name == 'mean':
        return float(np.mean(d))
    if metric_name == 't99':
        return float(np.percentile(d, 99))
    return np.nan


def extract_spell_groups_by_year(pr_masked, dry_threshold, mask_da):
    is_dry = (pr_masked < dry_threshold).where(mask_da)
    dry_stacked = is_dry.stack(cell=('y', 'x')).transpose('time', 'cell').compute()
    times = dry_stacked['time'].values
    dry_vals = dry_stacked.values
    n_cells = dry_vals.shape[1]
    groups = [{} for _ in range(n_cells)]
    for idx in range(n_cells):
        col = dry_vals[:, idx]
        if not np.any(col):
            continue
        x = np.asarray(col, dtype=np.bool_)
        padded = np.r_[False, x, False]
        dx = np.diff(padded.astype(np.int8))
        starts, ends = np.where(dx == 1)[0], np.where(dx == -1)[0]
        for s, e in zip(starts, ends):
            duration = int(e - s)
            if duration <= 0:
                continue
            year = time_to_year(times[s])
            groups[idx].setdefault(year, []).append(duration)
    for idx in range(n_cells):
        groups[idx] = {y: np.asarray(v, dtype=np.int16) for y, v in groups[idx].items()}
    return groups, mask_da.stack(cell=('y', 'x')).values


def pool_spells_by_years(group_dict, year_list):
    parts = [group_dict[y] for y in year_list if y in group_dict and group_dict[y].size > 0]
    return np.array([], dtype=np.int16) if not parts else np.concatenate(parts)


def bootstrap_one_cell(groups_cr2, groups_ala, years_arr, metric_name, n_iter, seed):
    rng = np.random.default_rng(seed)
    if not groups_cr2 or not groups_ala:
        return np.nan, np.nan, np.nan, False, False, False
    all_years = [y for y in years_arr if y in groups_cr2 or y in groups_ala]
    if not all_years:
        return np.nan, np.nan, np.nan, False, False, False
    m_cr2 = spell_metric_from_durations(pool_spells_by_years(groups_cr2, all_years), metric_name)
    m_ala = spell_metric_from_durations(pool_spells_by_years(groups_ala, all_years), metric_name)
    if not (np.isfinite(m_cr2) and np.isfinite(m_ala)):
        return np.nan, np.nan, np.nan, False, False, False
    obs_delta = m_ala - m_cr2
    boots = np.empty(n_iter, dtype=np.float64)
    n_years = years_arr.size
    for b in range(n_iter):
        draw = rng.choice(years_arr, size=n_years, replace=True)
        bm_cr2 = spell_metric_from_durations(pool_spells_by_years(groups_cr2, draw), metric_name)
        bm_ala = spell_metric_from_durations(pool_spells_by_years(groups_ala, draw), metric_name)
        boots[b] = bm_ala - bm_cr2 if np.isfinite(bm_cr2) and np.isfinite(bm_ala) else np.nan
    boots = boots[np.isfinite(boots)]
    if boots.size < 50:
        return obs_delta, np.nan, np.nan, False, False, False
    ci_low, ci_high = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    significant = (ci_low > 0) or (ci_high < 0)
    return obs_delta, ci_low, ci_high, significant, ci_low > 0, ci_high < 0


def _bootstrap_worker(args):
    idx, groups_cr2, groups_ala, years_list, metric_name, n_iter, seed = args
    years_arr = np.asarray(years_list, dtype=int)
    out = bootstrap_one_cell(groups_cr2, groups_ala, years_arr, metric_name, n_iter, seed + idx)
    return idx, out


def bootstrap_paired_parallel(groups_cr2, groups_ala, mask_stacked, years_arr, metric_name,
                              n_iter=BOOTSTRAP_ITER, seed=RANDOM_SEED, n_workers=N_WORKERS):
    n_cells = len(groups_cr2)
    obs = np.full(n_cells, np.nan)
    ci_low = np.full(n_cells, np.nan)
    ci_high = np.full(n_cells, np.nan)
    significant = np.zeros(n_cells, dtype=bool)
    sig_higher = np.zeros(n_cells, dtype=bool)
    sig_lower = np.zeros(n_cells, dtype=bool)
    years_list = years_arr.tolist()
    tasks = [
        (idx, groups_cr2[idx], groups_ala[idx], years_list, metric_name, n_iter, seed)
        for idx in range(n_cells)
        if mask_stacked[idx]
    ]
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_bootstrap_worker, t) for t in tasks]
        for fut in as_completed(futures):
            idx, (o, lo, hi, sig, shi, slo) = fut.result()
            obs[idx], ci_low[idx], ci_high[idx] = o, lo, hi
            significant[idx], sig_higher[idx], sig_lower[idx] = sig, shi, slo
            done += 1
            if done % 500 == 0:
                print(f'      {done}/{len(tasks)} celdas...', flush=True)
    return obs, ci_low, ci_high, significant, sig_higher, sig_lower


def summarize_bootstrap(mask_stacked, obs, significant, sig_higher, escenario, metric_label):
    valid = mask_stacked & np.isfinite(obs)
    data = obs[valid]
    sig = significant[valid]
    sig_hi = sig_higher[valid]
    return {
        'Escenario': escenario,
        'Metrica': metric_label,
        'Delta medio espacial (dias)': float(np.mean(data)),
        'Delta mediano espacial (dias)': float(np.median(data)),
        '% celdas significativas': float(np.mean(sig) * 100.0),
        '% ALADIN > CR2MET (sig.)': float(np.mean(sig_hi) * 100.0),
        '% ALADIN < CR2MET (sig.)': float(np.mean(sig & ~sig_hi) * 100.0),
        'Celdas validas': int(data.size),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print('1/4: Cargando datos...', flush=True)
    pr_aladin = open_aladin_historical()
    pr_cr2met_native = open_cr2met_historical()
    chile_geom = load_chile_geometry()
    chile_mask_bool = build_chile_mask_on_aladin_grid(
        pr_aladin['lat'].values, pr_aladin['lon'].values, chile_geom,
    )
    chile_mask = xr.DataArray(
        chile_mask_bool,
        coords={'y': pr_aladin['y'], 'x': pr_aladin['x'], 'lat': pr_aladin['lat'], 'lon': pr_aladin['lon']},
        dims=['y', 'x'],
    )
    pr_cr2met = regrid_cr2met_to_aladin(pr_cr2met_native, pr_aladin)
    pr_cr2met, pr_aladin = xr.align(normalize_daily_time(pr_cr2met), normalize_daily_time(pr_aladin), join='inner')
    pr_cr2met_chile = pr_cr2met.where(chile_mask)
    pr_aladin_chile = pr_aladin.where(chile_mask)
    mask_stacked = chile_mask.stack(cell=('y', 'x')).values
    years_common = np.unique(years_from_time(pr_aladin_chile['time']))
    print(f'   Celdas Chile: {int(chile_mask.sum().values)} | anos: {years_common.size}', flush=True)

    print('2/4: tau* local (Forma C)...', flush=True)
    tau_local_map = pixelwise_local_aladin_threshold(
        pr_cr2met_chile, pr_aladin_chile, TAU_CR2MET_REF, chile_mask,
    )

    scenarios = {
        'A_opcion_B_integrado': {
            'title': 'Forma A — Opcion B (tau* integrado)',
            'thresh_cr2': TAU_CR2MET_REF,
            'thresh_ala': TAU_ALADIN_DOMINIO,
        },
        'B_mismo_umbral': {
            'title': 'Forma B — Mismo umbral en CR2MET y ALADIN',
            'thresh_cr2': TAU_CR2MET_REF,
            'thresh_ala': TAU_CR2MET_REF,
        },
        'C_umbral_local': {
            'title': 'Forma C — tau* local (wet-day matching por pixel)',
            'thresh_cr2': TAU_CR2MET_REF,
            'thresh_ala': tau_local_map,
        },
    }

    summary_rows = []
    print('3/4: Bootstrap por escenario...', flush=True)
    for scen_key, cfg in scenarios.items():
        print(f"\n  {cfg['title']}", flush=True)
        print('    Extrayendo spells CR2MET...', flush=True)
        groups_cr2, _ = extract_spell_groups_by_year(pr_cr2met_chile, cfg['thresh_cr2'], chile_mask)
        print('    Extrayendo spells ALADIN...', flush=True)
        groups_ala, _ = extract_spell_groups_by_year(pr_aladin_chile, cfg['thresh_ala'], chile_mask)
        for metric_key, metric_label in BOOTSTRAP_METRICS:
            print(f'    Bootstrap: {metric_label}...', flush=True)
            obs, _, _, significant, sig_higher, _ = bootstrap_paired_parallel(
                groups_cr2, groups_ala, mask_stacked, years_common, metric_key,
            )
            summary_rows.append(
                summarize_bootstrap(mask_stacked, obs, significant, sig_higher, cfg['title'], metric_label)
            )

    summary_df = pd.DataFrame(summary_rows)
    print('\n4/4: Resultados bootstrap', flush=True)
    print(summary_df.round(3).to_string(index=False), flush=True)

    out_json = OUT_DIR / 'bootstrap_summary.json'
    out_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding='utf-8')
    summary_df.round(3).to_csv(OUT_DIR / 'bootstrap_summary.csv', index=False)
    print(f'\nGuardado: {out_json}', flush=True)
    return summary_df


if __name__ == '__main__':
    main()
