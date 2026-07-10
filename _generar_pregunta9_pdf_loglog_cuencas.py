"""Genera PDFs log-log por cuenca (criterio i) para §4.2 del paper."""
import warnings
warnings.filterwarnings('ignore')

import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.io.shapereader as shpreader
from PIL import Image
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '_pregunta9_outputs'
PAPER = ROOT / 'paper_figuras'
PRES = ROOT / 'presentacion_tesis' / 'figuras'
CACHE_DURS = OUT / 'pdf_durations_i_global.pkl'

HIST_START, HIST_END = '1980-01-01', '2014-12-31'
FUT_START, FUT_END = '2040-01-01', '2074-12-31'
FUTURE_SCENARIO = 'ssp585'
TAU_CR2MET_REF = 1.0
TAU_ALADIN_DOMINIO = 5.285
BISECTION_TAU_MAX, BISECTION_TOL = 15.0, 1e-4
PDF_BINS, MIN_DURATION = 25, 1
EXCLUDE_BOUNDARY_SPELLS = True

BASIN_SPECS = {
    'Loa':    {'bounds': (-69.8, -68.2, -24.0, -21.5)},
    'Maipo':  {'bounds': (-71.8, -69.8, -34.5, -33.2)},
    'Maule':  {'bounds': (-72.8, -71.0, -36.5, -34.8)},
    'Biobio': {'bounds': (-73.8, -71.5, -38.5, -36.5)},
}

DATASETS_STYLE = {
    'CR2MET (hist)':   {'color': 'black',     'ls': '-',  'lw': 2.2},
    'ALADIN (hist)':   {'color': 'steelblue', 'ls': '-',  'lw': 2.0},
    'ALADIN (futuro)': {'color': 'crimson',   'ls': '--', 'lw': 2.0},
}

SCEN_TITLE = 'Criterio i — R0 global (5.285 mm en ALADIN)'


def load_chile_geometry():
    reader = shpreader.Reader(
        shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
    )
    geoms = [r.geometry for r in reader.records()
             if r.attributes.get('NAME') == 'Chile' or r.attributes.get('ADMIN') == 'Chile']
    return unary_union(geoms)


def build_chile_mask_on_aladin_grid(lat2d, lon2d, geometry):
    prepared = prep(geometry)
    return np.fromiter(
        (prepared.contains(Point(float(x), float(y))) or geometry.touches(Point(float(x), float(y)))
         for y, x in zip(lat2d.ravel(), lon2d.ravel())),
        dtype=bool, count=lat2d.size,
    ).reshape(lat2d.shape)


def build_basin_mask(lat2d, lon2d, chile_mask_bool, bounds):
    lon_min, lon_max, lat_min, lat_max = bounds
    return (
        (lon2d >= lon_min) & (lon2d <= lon_max) &
        (lat2d >= lat_min) & (lat2d <= lat_max) & chile_mask_bool
    )


def open_aladin_period(start, end, scenario=None):
    if scenario is None:
        files = sorted(ROOT.glob('pr1/pr_CHP12_*_historical_*.nc'))
    else:
        files = sorted(ROOT.glob(f'pr1/pr_CHP12_*_{scenario}_*.nc'))
    ds = xr.open_mfdataset([str(p) for p in files], use_cftime=True, chunks={'time': 365})
    return ds['pr'].sel(time=slice(start, end)) * 86400.0


def open_cr2met_period(start, end):
    ds = xr.open_mfdataset(str(ROOT / 'pr/CR2MET_pr_v2.5_day_*.nc'), chunks={'time': 365})
    return ds['pr'].sel(time=slice(start, end))


def wet_fraction_1d(pr_series, threshold):
    x = np.asarray(pr_series, dtype=float)
    x = x[np.isfinite(x)]
    return float((x >= threshold).mean()) if x.size else np.nan


def find_threshold_for_target_1d(pr_series, target_fraction):
    if not np.isfinite(target_fraction):
        return np.nan
    f0, fmax = wet_fraction_1d(pr_series, 0.0), wet_fraction_1d(pr_series, BISECTION_TAU_MAX)
    if target_fraction > f0 + BISECTION_TOL or target_fraction < fmax - BISECTION_TOL:
        return np.nan
    lo, hi = 0.0, BISECTION_TAU_MAX
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = wet_fraction_1d(pr_series, mid)
        if abs(fmid - target_fraction) <= BISECTION_TOL:
            return mid
        lo, hi = (mid, hi) if fmid > target_fraction else (lo, mid)
    return 0.5 * (lo + hi)


def pixelwise_local_aladin_threshold(pr_cr2, pr_aladin, tau_cr2, mask_da):
    f_target = (pr_cr2 >= tau_cr2).mean(dim='time').where(mask_da)
    stacked = xr.Dataset({
        'pr_ala': pr_aladin.stack(cell=('y', 'x')),
        'f_target': f_target.stack(cell=('y', 'x')),
    }).compute()
    tau_star = np.full(stacked.sizes['cell'], np.nan, dtype=np.float32)
    for idx in range(stacked.sizes['cell']):
        ft = float(stacked['f_target'].values[idx])
        if np.isfinite(ft):
            tau_star[idx] = find_threshold_for_target_1d(stacked['pr_ala'].values[:, idx], ft)
    return xr.DataArray(tau_star, coords={'cell': stacked['cell']}, dims=['cell']).unstack('cell')


def dry_bool_1d(col):
    col = np.asarray(col, dtype=float)
    return np.isfinite(col) & (col > 0.5)


def run_lengths_1d(dry_bool):
    dry_bool = np.asarray(dry_bool, dtype=bool)
    if not np.any(dry_bool):
        return np.array([], dtype=np.int32)
    padded = np.r_[False, dry_bool, False]
    dx = np.diff(padded.astype(np.int8))
    return (np.where(dx == -1)[0] - np.where(dx == 1)[0]).astype(np.int32)


def extract_basin_spell_records(pr, dry_threshold, basin_mask):
    is_dry = (pr < dry_threshold).where(basin_mask)
    stacked = is_dry.stack(cell=('y', 'x')).transpose('time', 'cell').compute()
    times = stacked['time'].values
    vals = stacked.values
    mask_flat = basin_mask.stack(cell=('y', 'x')).values
    n_time = vals.shape[0]
    durations = []
    for idx in range(vals.shape[1]):
        if not mask_flat[idx]:
            continue
        dry = dry_bool_1d(vals[:, idx])
        if not np.any(dry):
            continue
        padded = np.r_[False, dry, False]
        dx = np.diff(padded.astype(np.int8))
        starts, ends = np.where(dx == 1)[0], np.where(dx == -1)[0]
        for s, e, dur in zip(starts, ends, ends - starts):
            if EXCLUDE_BOUNDARY_SPELLS and (s == 0 or e == n_time):
                continue
            if dur > 0:
                durations.append(int(dur))
    return np.asarray(durations, dtype=np.int32)


def pdf_from_durations(durations):
    d = np.asarray(durations, dtype=float)
    d = d[np.isfinite(d) & (d >= MIN_DURATION)]
    if d.size < 50:
        return None, None, 0
    lo, hi = max(float(np.min(d)), MIN_DURATION), float(np.max(d))
    if hi <= lo:
        return None, None, 0
    bins = np.logspace(np.log10(lo), np.log10(hi), PDF_BINS + 1)
    counts, edges = np.histogram(d, bins=bins, density=False)
    centers = np.sqrt(edges[:-1] * edges[1:])
    heights = counts / np.diff(edges)
    return centers, heights, int(d.size)


def load_or_compute_durations():
    if CACHE_DURS.exists():
        print(f'Cargando duraciones desde cache: {CACHE_DURS}')
        with CACHE_DURS.open('rb') as f:
            return pickle.load(f)

    print('1/2: Cargando datos y extrayendo spells (puede tardar varios minutos)...')
    pr_ala_hist = open_aladin_period(HIST_START, HIST_END)
    pr_ala_fut = open_aladin_period(FUT_START, FUT_END, scenario=FUTURE_SCENARIO)
    pr_cr2_hist = open_cr2met_period(HIST_START, HIST_END).interp(
        lat=pr_ala_hist['lat'], lon=pr_ala_hist['lon'], method='linear',
    )

    lat2d, lon2d = pr_ala_hist['lat'].values, pr_ala_hist['lon'].values
    chile_mask_bool = build_chile_mask_on_aladin_grid(lat2d, lon2d, load_chile_geometry())
    chile_mask = xr.DataArray(
        chile_mask_bool,
        coords={'y': pr_ala_hist['y'], 'x': pr_ala_hist['x'],
                'lat': pr_ala_hist['lat'], 'lon': pr_ala_hist['lon']},
        dims=['y', 'x'],
    )

    durs_by_basin = {}
    for basin_name, spec in BASIN_SPECS.items():
        print(f'  Cuenca {basin_name}...')
        bm = xr.DataArray(
            build_basin_mask(lat2d, lon2d, chile_mask_bool, spec['bounds']),
            coords=chile_mask.coords, dims=['y', 'x'],
        )
        durs_by_basin[basin_name] = {
            'CR2MET (hist)': extract_basin_spell_records(
                pr_cr2_hist.where(chile_mask), TAU_CR2MET_REF, bm),
            'ALADIN (hist)': extract_basin_spell_records(
                pr_ala_hist.where(chile_mask), TAU_ALADIN_DOMINIO, bm),
            'ALADIN (futuro)': extract_basin_spell_records(
                pr_ala_fut.where(chile_mask), TAU_ALADIN_DOMINIO, bm),
        }

    OUT.mkdir(exist_ok=True)
    with CACHE_DURS.open('wb') as f:
        pickle.dump(durs_by_basin, f)
    print(f'  Cache guardado: {CACHE_DURS}')
    return durs_by_basin


def plot_basin_pdf_loglog(basin_name, durs_dict, save_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for ds_name, style in DATASETS_STYLE.items():
        centers, heights, n_spells = pdf_from_durations(durs_dict[ds_name])
        if centers is None:
            continue
        yvals = np.where(heights > 0, heights, np.nan)
        ax.plot(centers, yvals, label=f'{ds_name} (n={n_spells:,})', **style)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Duracion de dry spell (dias)')
    ax.set_ylabel('f(t)  [integral = n_spells]')
    ax.set_title(
        f'Cuenca {basin_name} — {SCEN_TITLE}\n'
        f'Hist: {HIST_START[:4]}-{HIST_END[:4]} | Futuro: {FUT_START[:4]}-{FUT_END[:4]} ({FUTURE_SCENARIO})',
        fontweight='bold', fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.25)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def build_panel(paths, out_path):
    imgs = [Image.open(p) for p in paths]
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    canvas = Image.new('RGB', (w * 2, h * 2), 'white')
    labels = ['(a) Loa', '(b) Maipo', '(c) Maule', '(d) Biobio']
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        row, col = divmod(i, 2)
        canvas.paste(im, (col * w, row * h))
    canvas.save(out_path)
    for im in imgs:
        im.close()


def main():
    OUT.mkdir(exist_ok=True)
    PAPER.mkdir(exist_ok=True)
    PRES.mkdir(parents=True, exist_ok=True)

    durs_by_basin = load_or_compute_durations()

    print('2/2: Generando PDFs log-log...')
    paths = []
    for basin in BASIN_SPECS:
        fname = f'pdf_{basin}_i_global_loglog.png'
        out = OUT / fname
        plot_basin_pdf_loglog(basin, durs_by_basin[basin], out)
        print(f'  -> {out}')
        paths.append(out)

        # copias
        for dest in (PAPER / fname, PRES / f'fig_p9_{basin.lower()}_i_global_loglog.png'):
            dest.write_bytes(out.read_bytes())

    panel_out = PAPER / 'fig42_pdf_cuencas_loglog_panel.png'
    build_panel(paths, panel_out)
    print(f'  -> {panel_out}')

    pres_panel = PRES / 'fig42_pdf_cuencas_loglog_panel.png'
    pres_panel.write_bytes(panel_out.read_bytes())
    print('Listo.')


if __name__ == '__main__':
    main()
