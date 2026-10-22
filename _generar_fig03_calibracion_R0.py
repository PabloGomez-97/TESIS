"""Figura 3: calibracion del umbral R0 de ALADIN frente a CR2MET a 1 mm (1980-2014).

Guarda tambien los valores de la calibracion en resultados_tesis/calibracion_R0.json (R0 sin redondear, F de cada producto
y F de ALADIN con el R0 redondeado a 3 decimales que usan todos los scripts, 5,285 mm).
"""
import warnings
warnings.filterwarnings('ignore')

import json
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
OUTS = [ROOT / 'paper_figuras' / 'fig03_calibracion_R0_ftau.png']
# Con R = 0 todo dia cuenta como humedo (pr >= 0): la curva parte en 0,1 mm para evitar ese artefacto.
R_GRID = np.round(np.arange(0.1, 8.0001, 0.05), 3)


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


def domain_fwet(pr, threshold):
    """pr: array (tiempo, celdas). Media espacial (igual peso por celda) de la fraccion de dias pr >= R."""
    return float(np.nanmean((pr >= threshold).mean(axis=0)) * 100)


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
    mask = chile_mask_aladin(pr_ala['lat'].values, pr_ala['lon'].values, geom)
    jj, ii = np.where(mask)

    print('Calculando F(R)...')
    ala = pr_ala.values[:, jj, ii]
    cr2 = pr_cr2.values[:, jj, ii]
    curve_ala = [domain_fwet(ala, t) for t in R_GRID]
    f_cr2_1 = domain_fwet(cr2, 1.0)
    f_ala_1 = domain_fwet(ala, 1.0)

    lo, hi = 0.0, 15.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if domain_fwet(ala, mid) > f_cr2_1:
            lo = mid
        else:
            hi = mid
    r0 = 0.5 * (lo + hi)
    print(f'  F_CR2MET(1 mm) = {f_cr2_1:.2f}% | F_ALADIN(1 mm) = {f_ala_1:.2f}%')
    print(f'  R0 = {r0:.3f} mm')
    r0_usado = 5.285  # valor redondeado que usan todos los scripts (TAU_ALADIN_DOMINIO)
    calib = {
        'periodo': '1980-2014', 'celdas': int(mask.sum()), 'umbral_cr2met_mm': 1.0,
        'F_cr2met_1mm_pct': f_cr2_1, 'F_aladin_1mm_pct': f_ala_1, 'exceso_aladin_1mm_pp': f_ala_1 - f_cr2_1,
        'R0_biseccion_mm': r0, 'R0_usado_mm': r0_usado, 'F_aladin_R0_usado_pct': domain_fwet(ala, r0_usado),
    }
    out_json = ROOT / 'resultados_tesis' / 'calibracion_R0.json'
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(calib, indent=2), encoding='utf-8')
    print(json.dumps(calib, indent=2))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(R_GRID, curve_ala, 'steelblue', lw=2.2, label=r'ALADIN $F(R)$')
    ax.scatter(
        [1.0], [f_ala_1], s=80, facecolors='steelblue', edgecolors='white', linewidth=1.2, zorder=5,
        label=f'ALADIN a 1 mm ($F$ = {f_ala_1:.1f} %)'.replace('.', ','),
    )
    ax.scatter(
        [1.0], [f_cr2_1], s=90, c='firebrick', zorder=5,
        label=f'CR2MET a 1 mm ($F$ = {f_cr2_1:.1f} %)'.replace('.', ','),
    )
    ax.scatter(
        [r0], [f_cr2_1], s=100, facecolors='white', edgecolors='firebrick',
        linewidth=2, marker='s', zorder=5,
        label=f'ALADIN con $R_0$ = {r0:.3f} mm'.replace('.', ','),
    )
    ax.axhline(f_cr2_1, color='firebrick', ls=':', alpha=0.6)
    ax.annotate(
        '', xy=(1.0, f_cr2_1 + 0.6), xytext=(1.0, f_ala_1 - 0.6),
        arrowprops=dict(arrowstyle='<->', color='0.3', lw=1.3),
    )
    ax.text(1.12, 0.5 * (f_cr2_1 + f_ala_1), f'+{f_ala_1 - f_cr2_1:.1f} p.p.'.replace('.', ',', 1),
            va='center', fontsize=10, color='0.25')
    ax.set_xlabel(r'Umbral $R$ (mm día$^{-1}$)')
    ax.set_ylabel('Fracción media de días húmedos en Chile (%)')
    ax.set_xlim(0, 8.2)
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
