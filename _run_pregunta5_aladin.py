"""Ejecuta el analisis de risk ratios ALADIN hist vs futuro (pregunta 5/6)."""
import warnings
warnings.filterwarnings('ignore')

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.prepared import prep
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

# =====================================================================
# CONFIG
# =====================================================================
HIST_START = '1980-01-01'
HIST_END = '2014-12-31'
FUT_START = '2040-01-01'
FUT_END = '2074-12-31'
FUTURE_SCENARIO = 'ssp585'

DRY_THRESHOLD = 5.285  # tau* ALADIN (Pregunta 6)
EVENT_THRESHOLD = 20
BOOTSTRAP_ITER = 1000
RANDOM_SEED = 42
PERIOD_1 = (1980, 2014)
PERIOD_2 = (2040, 2074)
THRESHOLD_SWEEP = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
SEASON_MONTH_MIN = 3
SEASON_MONTH_MAX = 11
RR_PLOT_YMAX = 4.0

REGION_SPECS = {
    'Coquimbo': {'query': 'Coquimbo', 'color': 'firebrick'},
    "O'Higgins": {'query': 'higgins', 'color': 'darkorange'},
    'La Araucanía': {'query': 'araucan', 'color': 'forestgreen'},
    'Los Lagos': {'query': 'los lagos', 'color': 'steelblue'},
}


def normalize_text(text):
    text = unicodedata.normalize('NFKD', str(text).lower())
    return ''.join(ch for ch in text if not unicodedata.combining(ch))


def get_region_record(query, chile_records):
    query_norm = normalize_text(query)
    for record in chile_records:
        name_norm = normalize_text(record.attributes.get('name', ''))
        if query_norm in name_norm:
            return record
    raise ValueError(f'No se encontro region chilena para: {query}')


def build_region_mask(lat2d, lon2d, polygon):
    prepg = prep(polygon)
    return np.fromiter(
        (
            prepg.contains(Point(float(lon2d[j, i]), float(lat2d[j, i])))
            or polygon.touches(Point(float(lon2d[j, i]), float(lat2d[j, i])))
            for j in range(lat2d.shape[0])
            for i in range(lat2d.shape[1])
        ),
        dtype=bool,
        count=lat2d.size,
    ).reshape(lat2d.shape)


def time_to_year(t):
    if hasattr(t, 'year'):
        return int(t.year)
    return int(pd.Timestamp(t).year)


def time_to_month(t):
    if hasattr(t, 'month'):
        return int(t.month)
    return int(pd.Timestamp(t).month)


def open_aladin_period(start, end, scenario=None):
    if scenario is None:
        files = sorted(Path('pr1').glob('pr_CHP12_*_historical_*.nc'))
    else:
        files = sorted(Path('pr1').glob(f'pr_CHP12_*_{scenario}_*.nc'))
    ds = xr.open_mfdataset([str(p) for p in files], use_cftime=True, chunks={'time': 365})
    return ds['pr'].sel(time=slice(start, end)) * 86400.0


def extract_dry_spells_aladin(pr_da, region_mask, dry_threshold,
                              season_month_min=SEASON_MONTH_MIN, season_month_max=SEASON_MONTH_MAX):
    is_dry = (pr_da < dry_threshold).where(region_mask)
    stacked = is_dry.stack(cell=('y', 'x')).transpose('time', 'cell').compute()
    times = stacked['time'].values
    vals = stacked.values
    records = []
    for idx in range(vals.shape[1]):
        col = vals[:, idx]
        if not np.any(col):
            continue
        x = np.asarray(col, dtype=bool)
        padded = np.r_[False, x, False]
        dx = np.diff(padded.astype(np.int8))
        starts, ends = np.where(dx == 1)[0], np.where(dx == -1)[0]
        for s, e in zip(starts, ends):
            duration = int(e - s)
            if duration <= 0:
                continue
            start_month = time_to_month(times[s])
            if start_month < season_month_min or start_month > season_month_max:
                continue
            records.append({
                'start_year': time_to_year(times[s]),
                'start_month': start_month,
                'duration': duration,
            })
    return pd.DataFrame(records)


def event_probability(spell_df, start_year, end_year, min_duration):
    subset = spell_df[
        (spell_df['start_year'] >= start_year) & (spell_df['start_year'] <= end_year)
    ]
    total_spells = int(len(subset))
    event_spells = int((subset['duration'] >= min_duration).sum())
    probability = np.nan if total_spells == 0 else event_spells / total_spells
    return subset, probability, event_spells, total_spells


def risk_ratio(prob_period_2, prob_period_1):
    if prob_period_1 == 0 or not np.isfinite(prob_period_1):
        return np.nan
    return prob_period_2 / prob_period_1


def bootstrap_rr_by_year(spell_df, years_1, years_2, min_duration, n_iter=BOOTSTRAP_ITER, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    grouped = {
        year: spell_df.loc[spell_df['start_year'] == year, 'duration'].to_numpy()
        for year in sorted(spell_df['start_year'].unique())
    }
    years_1 = np.array(list(years_1))
    years_2 = np.array(list(years_2))
    rr_values = []
    for _ in range(n_iter):
        draw_1 = rng.choice(years_1, size=len(years_1), replace=True)
        draw_2 = rng.choice(years_2, size=len(years_2), replace=True)
        sample_1 = np.concatenate([grouped[y] for y in draw_1 if y in grouped and grouped[y].size > 0])
        sample_2 = np.concatenate([grouped[y] for y in draw_2 if y in grouped and grouped[y].size > 0])
        if sample_1.size == 0 or sample_2.size == 0:
            continue
        p1 = np.mean(sample_1 >= min_duration)
        p2 = np.mean(sample_2 >= min_duration)
        if p1 > 0:
            rr_values.append(p2 / p1)
    return np.array(rr_values)


def bootstrap_rr_curve(spell_df, years_1, years_2, thresholds, n_iter=BOOTSTRAP_ITER, seed=RANDOM_SEED):
    rows = []
    for thr in thresholds:
        _, p1, _, _ = event_probability(spell_df, PERIOD_1[0], PERIOD_1[1], thr)
        _, p2, _, _ = event_probability(spell_df, PERIOD_2[0], PERIOD_2[1], thr)
        rr_boot = bootstrap_rr_by_year(spell_df, years_1, years_2, thr, n_iter=n_iter, seed=seed)
        if rr_boot.size == 0:
            ci_low = ci_high = ci_med = np.nan
        else:
            ci_low, ci_high = np.percentile(rr_boot, [2.5, 97.5])
            ci_med = np.median(rr_boot)
        rr_obs = risk_ratio(p2, p1)
        rows.append({
            'Umbral (dias)': thr,
            'RR': rr_obs,
            'RR bootstrap mediano': ci_med,
            'IC95 inferior': ci_low,
            'IC95 superior': ci_high,
            'Significativo aumento': np.isfinite(ci_low) and ci_low > 1.0,
            'Significativo disminucion': np.isfinite(ci_high) and ci_high < 1.0,
            'P1 (%)': p1 * 100 if np.isfinite(p1) else np.nan,
            'P2 (%)': p2 * 100 if np.isfinite(p2) else np.nan,
        })
    return pd.DataFrame(rows)


def load_region_spell_data(pr_hist, pr_fut, lat2d, lon2d, y_coord, x_coord, record):
    geom = record.geometry
    region_name = record.attributes.get('name', 'region')
    mask = build_region_mask(lat2d, lon2d, geom)
    mask_da = xr.DataArray(
        mask,
        coords={'y': y_coord, 'x': x_coord, 'lat': (['y', 'x'], lat2d), 'lon': (['y', 'x'], lon2d)},
        dims=('y', 'x'),
    )
    spell_hist = extract_dry_spells_aladin(pr_hist, mask_da, DRY_THRESHOLD)
    spell_fut = extract_dry_spells_aladin(pr_fut, mask_da, DRY_THRESHOLD)
    spell_df = pd.concat([spell_hist, spell_fut], ignore_index=True)
    minx, miny, maxx, maxy = geom.bounds
    return {
        'name': region_name,
        'geom': geom,
        'bounds': (minx, miny, maxx, maxy),
        'mask': mask,
        'mask_da': mask_da,
        'spell_df': spell_df,
        'pixels': int(mask.sum()),
    }


def main():
    print('1/4: Cargando ALADIN historico y futuro...')
    pr_hist = open_aladin_period(HIST_START, HIST_END)
    pr_fut = open_aladin_period(FUT_START, FUT_END, scenario=FUTURE_SCENARIO)
    lat2d = pr_hist['lat'].values
    lon2d = pr_hist['lon'].values

    shape_path = shpreader.natural_earth(
        resolution='10m', category='cultural', name='admin_1_states_provinces'
    )
    reader = shpreader.Reader(shape_path)
    chile_regions = [r for r in reader.records() if r.attributes.get('admin') == 'Chile']

    n_y1 = PERIOD_1[1] - PERIOD_1[0] + 1
    n_y2 = PERIOD_2[1] - PERIOD_2[0] + 1
    print(f'Periodo hist: {PERIOD_1[0]}-{PERIOD_1[1]} ({n_y1} anos) | futuro: {PERIOD_2[0]}-{PERIOD_2[1]} ({n_y2} anos, {FUTURE_SCENARIO})')
    print(f'Dia seco ALADIN: pr < {DRY_THRESHOLD} mm/dia | Mar-Nov')

    print('2/4: Extrayendo dry spells por region...')
    region_data = {}
    for label, spec in REGION_SPECS.items():
        record = get_region_record(spec['query'], chile_regions)
        data = load_region_spell_data(
            pr_hist, pr_fut, lat2d, lon2d, pr_hist['y'], pr_hist['x'], record
        )
        region_data[label] = data
        print(f"  {label:16s} | {data['name']:22s} | pixeles: {data['pixels']:4d} | spells: {len(data['spell_df']):,}")

    years_1 = range(PERIOD_1[0], PERIOD_1[1] + 1)
    years_2 = range(PERIOD_2[0], PERIOD_2[1] + 1)

    print(f'3/4: Risk ratio >= {EVENT_THRESHOLD} dias...')
    summary_rows, rr_rows = [], []
    for label in REGION_SPECS:
        spell_df = region_data[label]['spell_df']
        _, prob_1, hits_1, total_1 = event_probability(spell_df, PERIOD_1[0], PERIOD_1[1], EVENT_THRESHOLD)
        _, prob_2, hits_2, total_2 = event_probability(spell_df, PERIOD_2[0], PERIOD_2[1], EVENT_THRESHOLD)
        rr_obs = risk_ratio(prob_2, prob_1)
        rr_boot = bootstrap_rr_by_year(spell_df, years_1, years_2, EVENT_THRESHOLD)
        if rr_boot.size > 0:
            ci_low, ci_med, ci_high = np.percentile(rr_boot, [2.5, 50, 97.5])
        else:
            ci_low = ci_med = ci_high = np.nan
        summary_rows.extend([
            {'Region': label, 'Periodo': f'{PERIOD_1[0]}-{PERIOD_1[1]}', 'Dry spells totales': total_1,
             f'Spells >= {EVENT_THRESHOLD} d': hits_1, 'Probabilidad (%)': prob_1 * 100},
            {'Region': label, 'Periodo': f'{PERIOD_2[0]}-{PERIOD_2[1]}', 'Dry spells totales': total_2,
             f'Spells >= {EVENT_THRESHOLD} d': hits_2, 'Probabilidad (%)': prob_2 * 100},
        ])
        rr_rows.append({
            'Region': label, 'Pixeles': region_data[label]['pixels'], 'RR observado': rr_obs,
            'RR bootstrap mediano': ci_med, 'IC95 inferior': ci_low, 'IC95 superior': ci_high,
            'Cruza RR=1': (ci_low <= 1 <= ci_high) if np.all(np.isfinite([ci_low, ci_high])) else np.nan,
            'Sig aumento': np.isfinite(ci_low) and ci_low > 1.0,
            'Sig disminucion': np.isfinite(ci_high) and ci_high < 1.0,
        })

    summary_table = pd.DataFrame(summary_rows)
    rr_table = pd.DataFrame(rr_rows)
    print('\n--- Probabilidades ---')
    print(summary_table.round(3).to_string(index=False))
    print('\n--- RR @ 20 dias ---')
    print(rr_table.round(3).to_string(index=False))

    print('4/4: Curvas RR vs umbral...')
    threshold_by_region = {}
    for label in REGION_SPECS:
        threshold_by_region[label] = bootstrap_rr_curve(
            region_data[label]['spell_df'], years_1, years_2, THRESHOLD_SWEEP,
            n_iter=BOOTSTRAP_ITER, seed=RANDOM_SEED,
        )

    out_dir = Path('outputs_pregunta5_aladin')
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()
    for ax, (label, spec) in zip(axes, REGION_SPECS.items()):
        tdf = threshold_by_region[label]
        color = spec['color']
        x = tdf['Umbral (dias)'].to_numpy()
        rr = tdf['RR'].to_numpy()
        lo = tdf['IC95 inferior'].to_numpy()
        hi = tdf['IC95 superior'].to_numpy()
        valid = np.isfinite(rr) & np.isfinite(lo) & np.isfinite(hi)
        if valid.any():
            ax.fill_between(x[valid], lo[valid], hi[valid], color=color, alpha=0.25, linewidth=0)
            ax.plot(x[valid], rr[valid], color=color, marker='o', linewidth=2, markersize=5)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.5)
        ax.set_title(label, fontweight='bold', color=color)
        ax.set_xlabel('Umbral de duracion (dias)')
        ax.set_ylabel(f'Risk ratio ({PERIOD_2[0]}-{PERIOD_2[1]} / {PERIOD_1[0]}-{PERIOD_1[1]})')
        ax.set_ylim(0, RR_PLOT_YMAX)
    fig.suptitle(
        f'ALADIN {FUTURE_SCENARIO} | pr < {DRY_THRESHOLD} mm/d | IC95 bootstrap por ano',
        fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_dir / 'rr_curves_aladin.png', dpi=150, bbox_inches='tight')
    plt.close()

    sig_rows = []
    for label in REGION_SPECS:
        t20 = threshold_by_region[label].loc[threshold_by_region[label]['Umbral (dias)'] == EVENT_THRESHOLD].iloc[0]
        sig_rows.append({
            'Region': label,
            'RR @ 20d': t20['RR'],
            'IC95 inf': t20['IC95 inferior'],
            'IC95 sup': t20['IC95 superior'],
            'Significancia': (
                'aumento' if t20['Significativo aumento']
                else ('disminucion' if t20['Significativo disminucion'] else 'no')
            ),
        })
    sig_table = pd.DataFrame(sig_rows)
    print('\n--- Significancia @ 20d ---')
    print(sig_table.round(3).to_string(index=False))

    # Save results JSON for notebook
    results = {
        'summary': summary_table.round(4).to_dict(orient='records'),
        'rr_table': rr_table.round(4).to_dict(orient='records'),
        'sig_table': sig_table.round(4).to_dict(orient='records'),
        'curves': {k: v.round(4).to_dict(orient='records') for k, v in threshold_by_region.items()},
    }
    with open(out_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nFigura guardada en {out_dir / "rr_curves_aladin.png"}')
    return results


if __name__ == '__main__':
    main()
