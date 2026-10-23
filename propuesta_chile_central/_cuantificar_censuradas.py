"""Cuantifica las rachas secas censuradas (tocan el primer o ultimo dia de la ventana) por cuenca.

Usa las mismas reglas que §2.2: NaN interrumpe la racha; umbral 1 mm en CR2MET y R0 = 5.285 mm en
ALADIN (criterio i). CR2MET interpolado linealmente se lee del cache de _test_regrillado_conservativo.py.
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from _run_pregunta7_bootstrap import load_chile_geometry, build_chile_mask_on_aladin_grid
import unidades as U

ROOT = Path(__file__).resolve().parent
CACHE_CR2 = ROOT / '_test_regrillado' / 'cr2met_lineal_vs_area.npz'
OUT_CSV = ROOT / '_test_regrillado' / 'rachas_censuradas_por_cuenca.csv'
R0 = 5.285


def load_aladin(scenario, start, end, jj, ii):
    files = sorted(str(p) for p in (ROOT / 'pr1').glob(f'pr_CHP12_*_{scenario}_*.nc'))
    da = xr.open_mfdataset(files, use_cftime=True, chunks={'time': 365})['pr'].sel(time=slice(start, end))
    return (da.values[:, jj, ii] * 86400.0).astype(np.float32)


def spells_with_censoring(pr, thr):
    """Duraciones completas y censuradas (concatenadas sobre celdas)."""
    n_time = pr.shape[0]
    full, cens = [], []
    for k in range(pr.shape[1]):
        x = np.isfinite(pr[:, k]) & (pr[:, k] < thr)
        dx = np.diff(np.r_[False, x, False].astype(np.int8))
        s, e = np.where(dx == 1)[0], np.where(dx == -1)[0]
        edge = (s == 0) | (e == n_time)
        full.append((e - s)[~edge])
        cens.append((e - s)[edge])
    return np.concatenate(full), np.concatenate(cens)


def main():
    ds = xr.open_mfdataset(sorted(str(p) for p in (ROOT / 'pr1').glob('pr_CHP12_*_historical_*.nc')),
                           use_cftime=True)
    lat2d, lon2d = ds['lat'].values, ds['lon'].values
    mask = build_chile_mask_on_aladin_grid(lat2d, lon2d, load_chile_geometry())
    jj, ii = np.where(mask)
    unit_masks = U.mascaras(lat2d, lon2d, mask)
    series = {
        'CR2MET 1980-2014 (1 mm)': (np.load(CACHE_CR2)['lin'], 1.0),
        'ALADIN 1980-2014 (R0)': (load_aladin('historical', '1980-01-01', '2014-12-31', jj, ii), R0),
        'ALADIN 2040-2074 (R0)': (load_aladin('ssp585', '2040-01-01', '2074-12-31', jj, ii), R0),
    }
    rows = []
    for name, (pr, thr) in series.items():
        for b in U.UNIDADES:
            sel = unit_masks[b][jj, ii]
            full, cens = spells_with_censoring(pr[:, sel], thr)
            rows.append({
                'serie': name, 'cuenca': b, 'celdas': int(sel.sum()),
                'rachas_completas': full.size, 'censuradas': cens.size,
                'pct_censuradas': 100 * cens.size / (full.size + cens.size),
                'p99_completas': np.percentile(full, 99), 'max_completas': int(full.max()),
                'mediana_censuradas': float(np.median(cens)) if cens.size else np.nan,
                'censuradas_sobre_p99': int((cens > np.percentile(full, 99)).sum()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(df.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
