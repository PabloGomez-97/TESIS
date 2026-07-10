"""Figura enfocada: ALADIN 1 mm vs ALADIN R0 calibrado (5.285 mm)."""
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
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper_figuras'
CAL = ROOT / '_pregunta9_calibracion_outputs'
PRES = ROOT / 'presentacion_tesis' / 'figuras'

HIST_START, HIST_END = '1980-01-01', '2014-12-31'
TAU_CR2 = 1.0
TAU_ALADIN_1MM = 1.0
TAU_R0 = 5.285
F_CR2 = 20.94
F_ALA_1 = 30.68

BASINS = ['Loa', 'Maule', 'Biobio']
BASIN_COLORS = {'Loa': '#D4A017', 'Maule': '#E67E22', 'Biobio': '#27AE60'}

SERIES_PDF = {
    'CR2MET 1 mm':        {'color': 'black',   'ls': '-',  'lw': 2.4},
    'ALADIN 1 mm':        {'color': 'crimson', 'ls': '--', 'lw': 2.2},
    'ALADIN tau* global': {'color': 'steelblue', 'ls': '-', 'lw': 2.2},
}


def load_chile_geometry():
    reader = shpreader.Reader(
        shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
    )
    geoms = [r.geometry for r in reader.records()
             if r.attributes.get('NAME') == 'Chile' or r.attributes.get('ADMIN') == 'Chile']
    return unary_union(geoms)


def chile_mask_aladin(lat2d, lon2d, geometry):
    prepared = prep(geometry)
    return np.fromiter(
        (prepared.contains(Point(float(x), float(y))) or geometry.touches(Point(float(x), float(y)))
         for y, x in zip(lat2d.ravel(), lon2d.ravel())),
        dtype=bool, count=lat2d.size,
    ).reshape(lat2d.shape)


def domain_fwet(pr, tau, mask_da):
    f = (pr >= tau).mean('time') * 100
    v = f.where(mask_da).values.ravel()
    return float(np.nanmean(v[np.isfinite(v)]))


def pdf_from_durations(durations, n_bins=25, min_duration=1):
    d = np.asarray(durations, dtype=float)
    d = d[np.isfinite(d) & (d >= min_duration)]
    if d.size < 50:
        return None, None, 0
    lo, hi = max(float(np.min(d)), min_duration), float(np.max(d))
    if hi <= lo:
        return None, None, 0
    bins = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    counts, edges = np.histogram(d, bins=bins, density=False)
    centers = np.sqrt(edges[:-1] * edges[1:])
    heights = counts / np.diff(edges)
    return centers, heights, int(d.size)


def load_calibracion_durations():
    cache = CAL / 'durations_calibracion.pkl'
    if cache.exists():
        with cache.open('rb') as f:
            return pickle.load(f)

    # Reutilizar logica de pregunta9_calibracion si no hay cache
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'cal', ROOT / 'pregunta9_calibracion_umbral_cuencas.ipynb'
    )
    raise FileNotFoundError(
        f'Falta {cache}. Ejecuta pregunta9_calibracion_umbral_cuencas.ipynb primero.'
    )


def load_durations_from_notebook_outputs():
    """Carga duraciones reconstruyendo desde CSV + notebook cache si existe."""
    cache = ROOT / '_pregunta9_calibracion_outputs' / 'spell_durations_calibracion.pkl'
    if cache.exists():
        with cache.open('rb') as f:
            return pickle.load(f)
    return None


def extract_and_cache_durations():
    """Extrae spells para las 3 series x 3 cuencas (rapido con datos ya en disco)."""
    from _generar_pregunta9_pdf_loglog_cuencas import (
        BASIN_SPECS, build_basin_mask, extract_basin_spell_records,
        load_chile_geometry, build_chile_mask_on_aladin_grid,
        open_aladin_period, open_cr2met_period, TAU_ALADIN_DOMINIO, TAU_CR2MET_REF,
    )

    cache = CAL / 'spell_durations_calibracion.pkl'
    if cache.exists():
        with cache.open('rb') as f:
            return pickle.load(f)

    print('Extrayendo duraciones (1 mm vs R0)...')
    pr_ala = open_aladin_period(HIST_START, HIST_END)
    pr_cr2 = open_cr2met_period(HIST_START, HIST_END).interp(
        lat=pr_ala['lat'], lon=pr_ala['lon'], method='linear',
    )
    lat2d, lon2d = pr_ala['lat'].values, pr_ala['lon'].values
    chile_bool = build_chile_mask_on_aladin_grid(lat2d, lon2d, load_chile_geometry())
    chile_mask = xr.DataArray(
        chile_bool,
        coords={'y': pr_ala['y'], 'x': pr_ala['x'], 'lat': pr_ala['lat'], 'lon': pr_ala['lon']},
        dims=['y', 'x'],
    )

    data = {}
    for basin, spec in BASIN_SPECS.items():
        if basin not in BASINS:
            continue
        bm = xr.DataArray(
            build_basin_mask(lat2d, lon2d, chile_bool, spec['bounds']),
            coords=chile_mask.coords, dims=['y', 'x'],
        )
        data[basin] = {
            'CR2MET 1 mm': extract_basin_spell_records(
                pr_cr2.where(chile_mask), TAU_CR2MET_REF, bm),
            'ALADIN 1 mm': extract_basin_spell_records(
                pr_ala.where(chile_mask), TAU_ALADIN_1MM, bm),
            'ALADIN tau* global': extract_basin_spell_records(
                pr_ala.where(chile_mask), TAU_ALADIN_DOMINIO, bm),
        }

    CAL.mkdir(exist_ok=True)
    with cache.open('wb') as f:
        pickle.dump(data, f)
    return data


def panel_fwet(ax):
    pr_ala = (
        xr.open_mfdataset(sorted(str(p) for p in (ROOT / 'pr1').glob('pr_CHP12_*_historical_*.nc')),
                          use_cftime=True, chunks={'time': 365})
        ['pr'].sel(time=slice(HIST_START, HIST_END)) * 86400.0
    )
    pr_cr2 = (
        xr.open_mfdataset(str(ROOT / 'pr' / 'CR2MET_pr_v2.5_day_*.nc'), chunks={'time': 365})
        ['pr'].sel(time=slice(HIST_START, HIST_END))
        .interp(lat=pr_ala['lat'], lon=pr_ala['lon'], method='linear')
    )
    geom = load_chile_geometry()
    mask = xr.DataArray(
        chile_mask_aladin(pr_ala['lat'].values, pr_ala['lon'].values, geom),
        dims=['y', 'x'],
    )

    grid = np.linspace(0, 8, 161)
    curve = [domain_fwet(pr_ala, t, mask) for t in grid]
    f1_ala = domain_fwet(pr_ala, TAU_ALADIN_1MM, mask)
    f_r0 = domain_fwet(pr_ala, TAU_R0, mask)

    ax.plot(grid, curve, color='steelblue', lw=2.2, label='ALADIN F(R)')
    ax.scatter([TAU_CR2], [F_CR2], s=100, c='black', zorder=5,
               label=f'CR2MET @ 1 mm ({F_CR2:.1f}%)')
    ax.scatter([TAU_ALADIN_1MM], [f1_ala], s=100, c='crimson', marker='x', lw=2, zorder=5,
               label=f'ALADIN @ 1 mm ({f1_ala:.1f}%)')
    ax.scatter([TAU_R0], [f_r0], s=110, facecolors='white', edgecolors='steelblue',
               linewidth=2.2, marker='s', zorder=5,
               label=f'ALADIN @ R0={TAU_R0} mm ({f_r0:.1f}%)')
    ax.axhline(F_CR2, color='black', ls=':', alpha=0.5, lw=1)
    ax.annotate('', xy=(TAU_R0, F_CR2), xytext=(TAU_ALADIN_1MM, f1_ala),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(2.8, 26.5, f'+{f1_ala - F_CR2:.1f} pp\nsin calibrar', fontsize=8, color='crimson')
    ax.text(4.2, 19.5, 'R0 iguala\nwet days', fontsize=8, color='steelblue')
    ax.set_xlabel('Umbral R (mm/dia)')
    ax.set_ylabel('Fraccion espacial media de wet days (%)')
    ax.set_title('(a) Por que calibrar R0', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 7)


def panel_bars(ax, df):
    basins = [b for b in BASINS if b in df['cuenca'].unique()]
    x = np.arange(len(basins))
    w = 0.25
    series_order = ['CR2MET 1 mm', 'ALADIN 1 mm', 'ALADIN tau* global']
    colors = ['black', 'crimson', 'steelblue']
    for i, (serie, col) in enumerate(zip(series_order, colors)):
        vals = [df[(df['cuenca'] == b) & (df['serie'] == serie)]['mean_dias'].iloc[0] for b in basins]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=serie.replace('tau* global', 'R0 global'),
                      color=col, edgecolor='white')
        for bar, b in zip(bars, basins):
            n = df[(df['cuenca'] == b) & (df['serie'] == serie)]['n_spells'].iloc[0]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'n={n/1000:.0f}k', ha='center', va='bottom', fontsize=6, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(basins)
    ax.set_ylabel('Duracion media de dry spells (dias)')
    ax.set_title('(b) Efecto en duracion media (1980-2014)', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)


def panel_pdf_loa(ax, durs):
    for name, style in SERIES_PDF.items():
        centers, heights, n = pdf_from_durations(durs[name])
        if centers is None:
            continue
        y = np.where(heights > 0, heights, np.nan)
        ax.plot(centers, y, label=f'{name.replace("tau* global", "R0 global")} (n={n:,})', **style)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Duracion (dias)')
    ax.set_ylabel('f(t)  [integral = n_spells]')
    ax.set_title('(c) Cuenca Loa — PDF log-log', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, which='both', alpha=0.25)


def main():
    OUT.mkdir(exist_ok=True)
    PRES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CAL / 'resumen_calibracion_umbral.csv')

    # Panel PDF: intentar cache de duraciones
    cache = CAL / 'spell_durations_calibracion.pkl'
    if not cache.exists():
        try:
            extract_and_cache_durations()
        except Exception as e:
            print(f'  aviso: no se pudo extraer duraciones ({e}); panel PDF omitido')

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.32, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_fwet(ax_a)
    panel_bars(ax_b, df)

    if cache.exists():
        with cache.open('rb') as f:
            all_durs = pickle.load(f)
        panel_pdf_loa(ax_c, all_durs['Loa'])
    else:
        ax_c.text(0.5, 0.5, 'Ejecutar script con datos para panel PDF', ha='center', va='center',
                  transform=ax_c.transAxes)

    fig.suptitle(
        'Diferencia entre ALADIN @ 1 mm y ALADIN @ R0 calibrado (5.285 mm)\n'
        'Sin R0: mas wet days (+9.7 pp) -> spells mas cortos y mas numerosos',
        fontweight='bold', fontsize=12, y=0.98,
    )

    out = OUT / 'fig_umbral_1mm_vs_R0_comparacion.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'-> {out}')

    pres = PRES / 'fig_umbral_1mm_vs_R0_comparacion.png'
    pres.write_bytes(out.read_bytes())
    print(f'-> {pres}')

    # Version solo PDF 3 curvas x 3 cuencas
    if cache.exists():
        with cache.open('rb') as f:
            all_durs = pickle.load(f)
        fig2, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, basin in zip(axes, BASINS):
            for name, style in SERIES_PDF.items():
                centers, heights, n = pdf_from_durations(all_durs[basin][name])
                if centers is None:
                    continue
                y = np.where(heights > 0, heights, np.nan)
                ax.plot(centers, y, label=f'{name.split()[0]} {name.split()[-1]}' if 'CR2MET' in name
                        else ('ALADIN 1mm' if '1 mm' in name else 'ALADIN R0'), **style)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Duracion (dias)')
            ax.set_title(f'Cuenca {basin}', fontweight='bold')
            ax.grid(True, which='both', alpha=0.25)
        axes[0].set_ylabel('f(t)  [integral = n_spells]')
        axes[0].legend(fontsize=7)
        fig2.suptitle(
            'PDF dry spells: CR2MET 1 mm vs ALADIN 1 mm vs ALADIN R0=5.285 mm | 1980-2014',
            fontweight='bold', fontsize=11,
        )
        plt.tight_layout()
        out2 = OUT / 'fig_umbral_1mm_vs_R0_pdf_panel.png'
        fig2.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f'-> {out2}')


if __name__ == '__main__':
    main()
