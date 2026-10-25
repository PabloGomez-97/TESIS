"""Test de sensibilidad: CR2MET interpolado linealmente vs promediado por area en celdas ALADIN.

Promedio por area: para cada celda ALADIN (poligono de bounds_lat/bounds_lon) se promedian las celdas
CR2MET (0.05 deg) cuyo centro cae dentro, ignorando NaN. Se comparan fwet, R0 calibrado, PDFs por
cuenca y mapas de duracion media / t99 (criterios i, ii, iii), con las mismas reglas de spells.
"""
import warnings
warnings.filterwarnings('ignore')

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath
from scipy import sparse

from _run_pregunta7_bootstrap import (
    load_chile_geometry, build_chile_mask_on_aladin_grid, find_threshold_for_target_1d,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '_test_regrillado'
CACHE = OUT / 'cr2met_lineal_vs_area.npz'
START, END = '1980-01-01', '2014-12-31'
TAU_CR2 = 1.0
R0_ACTUAL = 5.285
MIN_SPELLS = 30
BASINS = {
    'Loa': (-69.8, -68.2, -24.0, -21.5),
    'Maipo': (-71.8, -69.8, -34.5, -33.2),
    'Maule': (-72.8, -71.0, -36.5, -34.8),
    'Biobio': (-73.8, -71.5, -38.5, -36.5),
}


def load_aladin():
    files = sorted(str(p) for p in (ROOT / 'pr1').glob('pr_CHP12_*_historical_*.nc'))
    ds = xr.open_mfdataset(files, use_cftime=True, chunks={'time': 365})
    pr = ds['pr'].sel(time=slice(START, END)) * 86400.0
    blat = ds['bounds_lat'].isel(time=0).values
    blon = ds['bounds_lon'].isel(time=0).values
    dates = pd.DatetimeIndex([pd.Timestamp(str(t)[:10]) for t in pr['time'].values])
    return pr, ds['lat'].values, ds['lon'].values, blat, blon, dates


def area_weights(cells, blat, blon, cr_lat, cr_lon):
    """Matriz dispersa (n_cells x n_cr2) con 1 si el centro CR2MET cae en el poligono de la celda."""
    lat0, dlat = cr_lat[0], cr_lat[1] - cr_lat[0]
    lon0, dlon = cr_lon[0], cr_lon[1] - cr_lon[0]
    nlon = cr_lon.size
    rows, cols = [], []
    for k, (j, i) in enumerate(cells):
        poly = np.c_[blon[j, i], blat[j, i]]
        ia = max(int(np.floor((poly[:, 1].min() - lat0) / dlat)), 0)
        ib = min(int(np.ceil((poly[:, 1].max() - lat0) / dlat)) + 1, cr_lat.size)
        ja = max(int(np.floor((poly[:, 0].min() - lon0) / dlon)), 0)
        jb = min(int(np.ceil((poly[:, 0].max() - lon0) / dlon)) + 1, nlon)
        LA, LO = np.meshgrid(cr_lat[ia:ib], cr_lon[ja:jb], indexing='ij')
        inside = MplPath(poly).contains_points(np.c_[LO.ravel(), LA.ravel()])
        idx = (np.arange(ia, ib)[:, None] * nlon + np.arange(ja, jb)[None, :]).ravel()[inside]
        rows += [k] * idx.size
        cols += idx.tolist()
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cells), cr_lat.size * nlon))


def build_cr2met_versions(cells, lat2d, lon2d, blat, blon, dates):
    files = sorted((ROOT / 'pr').glob('CR2MET_pr_v2.5_day_*.nc'))
    files = [f for f in files if 1980 <= int(f.name.split('_')[4]) <= 2014]
    with xr.open_dataset(files[0]) as d0:
        cr_lat, cr_lon = d0['lat'].values, d0['lon'].values
    W = area_weights(cells, blat, blon, cr_lat, cr_lon)
    n_inside = np.asarray(W.sum(axis=1)).ravel()
    jj, ii = np.array(cells).T
    tlat = xr.DataArray(lat2d[jj, ii], dims='cell')
    tlon = xr.DataArray(lon2d[jj, ii], dims='cell')
    lin_parts, area_parts, t_parts = [], [], []
    for n, f in enumerate(files):
        with xr.open_dataset(f) as ds:
            pr = ds['pr'].load()
        lin_parts.append(pr.interp(lat=tlat, lon=tlon, method='linear').values.astype(np.float32))
        flat = pr.values.reshape(pr.sizes['time'], -1)
        valid = np.isfinite(flat)
        s = W @ np.where(valid, flat, 0.0).T
        c = W @ valid.T.astype(np.float64)
        with np.errstate(invalid='ignore', divide='ignore'):
            area_parts.append(np.where(c > 0, s / c, np.nan).T.astype(np.float32))
        t_parts.append(pd.DatetimeIndex(pr['time'].values).normalize())
        if n % 60 == 0:
            print(f'    CR2MET {f.name}', flush=True)
    t = t_parts[0].append(t_parts[1:])
    lin, area = np.concatenate(lin_parts), np.concatenate(area_parts)
    pos = pd.Index(t).get_indexer(dates)
    assert (pos >= 0).all(), 'fechas ALADIN sin CR2MET'
    return lin[pos], area[pos], n_inside


def spells_per_cell(pr, thr):
    """Duraciones por celda (lista de arrays), NaN interrumpe la racha, censuradas excluidas."""
    thr = np.broadcast_to(np.asarray(thr, dtype=float), (pr.shape[1],))
    n_time = pr.shape[0]
    out = []
    for k in range(pr.shape[1]):
        col = pr[:, k]
        x = np.isfinite(col) & (col < thr[k]) if np.isfinite(thr[k]) else np.zeros(n_time, bool)
        dx = np.diff(np.r_[False, x, False].astype(np.int8))
        s, e = np.where(dx == 1)[0], np.where(dx == -1)[0]
        keep = (s > 0) & (e < n_time)
        out.append((e - s)[keep])
    return out


def metrics(durs):
    mean = np.array([d.mean() if d.size >= MIN_SPELLS else np.nan for d in durs])
    t99 = np.array([np.percentile(d, 99) if d.size >= MIN_SPELLS else np.nan for d in durs])
    return mean, t99


def domain_fwet(pr, thr):
    f = np.nanmean(np.where(np.isfinite(pr), pr >= thr, np.nan), axis=0)
    return float(np.nanmean(f))


def calibrate_r0(pr_ala, target):
    lo, hi = 0.0, 15.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if domain_fwet(pr_ala, mid) > target else (lo, mid)
    return 0.5 * (lo + hi)


def local_r0(pr_cr2, pr_ala):
    f_t = np.nanmean(np.where(np.isfinite(pr_cr2), pr_cr2 >= TAU_CR2, np.nan), axis=0)
    return np.array([find_threshold_for_target_1d(pr_ala[:, k], f_t[k]) for k in range(pr_ala.shape[1])])


def main():
    OUT.mkdir(exist_ok=True)
    print('1/4 Cargando ALADIN...', flush=True)
    pr_ala_da, lat2d, lon2d, blat, blon, dates = load_aladin()
    mask = build_chile_mask_on_aladin_grid(lat2d, lon2d, load_chile_geometry())
    cells = list(zip(*np.where(mask)))
    jj, ii = np.array(cells).T
    pr_ala = pr_ala_da.values[:, jj, ii].astype(np.float32)
    print(f'   celdas Chile: {len(cells)} | dias: {len(dates)}', flush=True)

    if CACHE.exists():
        z = np.load(CACHE)
        cr_lin, cr_area, n_inside = z['lin'], z['area'], z['n_inside']
    else:
        print('2/4 Construyendo CR2MET lineal y por area...', flush=True)
        cr_lin, cr_area, n_inside = build_cr2met_versions(cells, lat2d, lon2d, blat, blon, dates)
        np.savez(CACHE, lin=cr_lin, area=cr_area, n_inside=n_inside)

    valid_lin = np.isfinite(cr_lin).any(axis=0)
    valid_area = np.isfinite(cr_area).any(axis=0)
    common = valid_lin & valid_area
    print(f'   celdas validas lineal={valid_lin.sum()} area={valid_area.sum()} comunes={common.sum()} '
          f'| centros CR2MET por celda: mediana {np.median(n_inside[common]):.0f}', flush=True)

    print('3/4 fwet y R0...', flush=True)
    res = {'celdas': {'lineal': int(valid_lin.sum()), 'area': int(valid_area.sum()), 'comunes': int(common.sum())}}
    for name, cr, sel in (('lineal', cr_lin, valid_lin), ('area', cr_area, valid_area),
                          ('lineal_comunes', cr_lin, common), ('area_comunes', cr_area, common)):
        f = domain_fwet(cr[:, sel], TAU_CR2)
        res[name] = {'fwet_cr2_1mm_pct': 100 * f, 'R0': calibrate_r0(pr_ala[:, sel], f)}
        print(f'   {name:15s} fwet CR2MET@1mm = {100*f:.3f} %  ->  R0 = {res[name]["R0"]:.3f} mm', flush=True)

    print('4/4 Spells, mapas y cuencas...', flush=True)
    durs_ala = {'i': spells_per_cell(pr_ala, R0_ACTUAL), 'ii': spells_per_cell(pr_ala, TAU_CR2)}
    m_ala = {k: metrics(v) for k, v in durs_ala.items()}
    rows, basin_rows = [], []
    for name, cr in (('lineal', cr_lin), ('area', cr_area)):
        durs_cr = spells_per_cell(cr, TAU_CR2)
        mc, tc = metrics(durs_cr)
        m_ala['iii_' + name] = metrics(spells_per_cell(pr_ala, local_r0(cr, pr_ala)))
        for crit, (ma, ta) in (('i', m_ala['i']), ('ii', m_ala['ii']), ('iii', m_ala['iii_' + name])):
            for metric, a, c in (('media', ma, mc), ('t99', ta, tc)):
                ok = common & np.isfinite(a) & np.isfinite(c)
                rows.append({'cr2met': name, 'criterio': crit, 'metrica': metric,
                             'CR2MET': c[ok].mean(), 'ALADIN': a[ok].mean(),
                             'Delta': (a[ok] - c[ok]).mean(), 'pix': int(ok.sum())})
        for b, (lo0, lo1, la0, la1) in BASINS.items():
            sel = (lon2d[jj, ii] >= lo0) & (lon2d[jj, ii] <= lo1) & (lat2d[jj, ii] >= la0) & (lat2d[jj, ii] <= la1)
            d = np.concatenate([durs_cr[k] for k in np.where(sel)[0]])
            basin_rows.append({'cuenca': b, 'cr2met': name, 'n_spells': d.size, 'media': d.mean(),
                               'p90': np.percentile(d, 90), 'p99': np.percentile(d, 99), 'max': d.max()})

    df = pd.DataFrame(rows)
    dfb = pd.DataFrame(basin_rows)
    df.to_csv(OUT / 'mapas_lineal_vs_area.csv', index=False)
    dfb.to_csv(OUT / 'cuencas_cr2met_lineal_vs_area.csv', index=False)
    (OUT / 'r0_lineal_vs_area.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(df.round(2).to_string(index=False))
    print(dfb.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
