"""Regenera fig_p6_ftau.png con notacion R0 (en vez de tau*)."""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parent
OUTS = [
    ROOT / 'presentacion_tesis' / 'figuras' / 'fig_p6_ftau.png',
    ROOT / 'paper_figuras' / 'fig03_calibracion_R0_ftau.png',
]


def load_chile_geometry():
    reader = shpreader.Reader(
        shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
    )
    geoms = [
        r.geometry for r in reader.records()
        if r.attributes.get('NAME') == 'Chile' or r.attributes.get('ADMIN') == 'Chile'
    ]
    return unary_union(geoms)


def chile_mask_aladin(lat2d, lon2d, geometry):
    prepared = prep(geometry)
    return np.fromiter(
        (
            prepared.contains(Point(float(x), float(y)))
            or geometry.touches(Point(float(x), float(y)))
            for y, x in zip(lat2d.ravel(), lon2d.ravel())
        ),
        dtype=bool,
        count=lat2d.size,
    ).reshape(lat2d.shape)


def domain_fwet(pr, threshold, mask_da):
    f = (pr >= threshold).mean('time') * 100
    v = f.where(mask_da).values.ravel()
    return float(np.nanmean(v[np.isfinite(v)]))


def main():
    print('Cargando datos...')
    pr_ala = xr.open_mfdataset(
        sorted(str(p) for p in (ROOT / 'pr1').glob('pr_CHP12_*_historical_*.nc')),
        use_cftime=True,
        chunks={'time': 365},
    )['pr'].sel(time=slice('1980-01-01', '2014-12-31')) * 86400.0

    pr_cr2 = xr.open_mfdataset(
        sorted(str(p) for p in (ROOT / 'pr').glob('CR2MET_pr_v2.5_day_*.nc')),
        chunks={'time': 365},
    )['pr'].sel(time=slice('1980-01-01', '2014-12-31')).interp(
        lat=pr_ala['lat'], lon=pr_ala['lon'], method='linear',
    )

    geom = load_chile_geometry()
    mask = xr.DataArray(
        chile_mask_aladin(pr_ala['lat'].values, pr_ala['lon'].values, geom),
        dims=['y', 'x'],
    )

    print('Calculando F(R)...')
    grid = np.linspace(0, 8, 161)
    curve_ala = [domain_fwet(pr_ala, t, mask) for t in grid]
    f_cr2_1 = domain_fwet(pr_cr2, 1.0, mask)

    lo, hi = 0.0, 15.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if domain_fwet(pr_ala, mid, mask) > f_cr2_1:
            lo = mid
        else:
            hi = mid
    r0 = 0.5 * (lo + hi)
    print(f'  F_CR2MET(1 mm) = {f_cr2_1:.2f}%')
    print(f'  R0 = {r0:.3f} mm')

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(grid, curve_ala, 'steelblue', lw=2.2, label=r'ALADIN $F(R)$')
    ax.scatter(
        [1.0], [f_cr2_1], s=90, c='firebrick', zorder=5,
        label=f'CR2MET @ 1 mm ($F$={f_cr2_1:.1f}%)',
    )
    ax.scatter(
        [r0], [f_cr2_1], s=100, facecolors='white', edgecolors='firebrick',
        linewidth=2, marker='s', zorder=5,
        label=f'ALADIN $R_0$={r0:.3f} mm',
    )
    ax.axhline(f_cr2_1, color='firebrick', ls=':', alpha=0.6)
    ax.set_xlabel(r'Umbral $R$ (mm/dia)')
    ax.set_ylabel('Fraccion espacial media de wet days Chile (%)')
    ax.set_title(r'Calibracion $R_0$ ALADIN vs CR2MET (1980-2014)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    for path in OUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f'-> {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
