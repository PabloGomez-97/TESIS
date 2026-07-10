"""Figura §3.2 — sesgos espaciales de persistencia CR2MET vs ALADIN."""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper_figuras'
OUT.mkdir(exist_ok=True)
CACHE = ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz'
BOOT_CSV = ROOT / 'outputs_pregunta7_bootstrap' / 'bootstrap_summary.csv'
RESUMEN_CSV = ROOT / '_pregunta7_outputs' / 'resumen_pregunta7.csv'

CHILE_EXTENT = [-76, -65, -55, -17]
CRITERIA_LABELS = {
    'A_opcion_B_integrado': 'i — R₀ global',
    'B_mismo_umbral': 'ii — 1 mm ambos',
    'C_umbral_local': 'iii — R₀ local',
}
CRITERIA_ORDER = ['A_opcion_B_integrado', 'B_mismo_umbral', 'C_umbral_local']
COLORS_CRIT = ['#2E86AB', '#C0392B', '#27AE60']


def load_criterion_i_maps():
    z = np.load(CACHE, allow_pickle=True)
    chile_mask_bool = z['chile_mask']
    coords = {
        'y': z['y'],
        'x': z['x'],
        'lat': (['y', 'x'], z['lat']),
        'lon': (['y', 'x'], z['lon']),
    }
    chile_mask = xr.DataArray(chile_mask_bool, coords=coords, dims=['y', 'x'])

    def _da(arr):
        return xr.DataArray(arr, coords=coords, dims=['y', 'x'])

    cr2_mean, ala_mean = _da(z['mean_cr2']), _da(z['mean_ala'])
    cr2_t99, ala_t99 = _da(z['t99_cr2']), _da(z['t99_ala'])
    return {
        'chile_mask': chile_mask,
        'mean': (cr2_mean, ala_mean, ala_mean - cr2_mean),
        't99': (cr2_t99, ala_t99, ala_t99 - cr2_t99),
    }


def draw_spatial_comparison_maps(maps, *, suptitle=None, figsize=(11, 9)):
    """Mapas 2x3 con colorbars debajo de cada fila (sin solapamiento)."""
    row_specs = [
        ('mean', 'Duracion media (dias)'),
        ('t99', 't99 — p99 (dias)'),
    ]
    col_titles = ['CR2MET (regrillado)', 'ALADIN historico', 'Delta = ALADIN - CR2MET']
    mask = maps['chile_mask']

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.02, hspace=0.42)
    row_axes = []

    for row_i, (key, row_label) in enumerate(row_specs):
        cr2, ala, dlt = maps[key]
        vals = np.concatenate([
            cr2.where(mask).values.ravel(),
            ala.where(mask).values.ravel(),
        ])
        vals = vals[np.isfinite(vals)]
        vmin = float(np.nanpercentile(vals, 2))
        vmax = float(np.nanpercentile(vals, 98))

        dvals = dlt.where(mask).values.ravel()
        dvals = dvals[np.isfinite(dvals)]
        dv = max(float(np.nanpercentile(np.abs(dvals), 98)), 0.5)

        axes_row = []
        im_main = im_delta = None
        for col_i, (da, title) in enumerate(zip([cr2, ala, dlt], col_titles)):
            ax = fig.add_subplot(gs[row_i, col_i], projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.4)
            ax.set_extent(CHILE_EXTENT)
            if col_i < 2:
                im_main = da.plot.pcolormesh(
                    ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
                    cmap='viridis', vmin=vmin, vmax=vmax, add_colorbar=False,
                )
            else:
                im_delta = da.plot.pcolormesh(
                    ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
                    cmap='RdBu_r', vmin=-dv, vmax=dv, add_colorbar=False,
                )
            ax.set_title(title, fontsize=9.5, fontweight='bold', pad=4)
            if col_i == 0:
                ax.text(
                    -0.12, 0.5, row_label, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='center', ha='center', rotation=90,
                )
            axes_row.append(ax)

        cbar_m = fig.colorbar(
            im_main, ax=axes_row[:2], orientation='horizontal',
            fraction=0.046, pad=0.07, aspect=50, shrink=0.92,
        )
        cbar_m.set_label(row_label, fontsize=8)
        cbar_m.ax.tick_params(labelsize=7)

        cbar_d = fig.colorbar(
            im_delta, ax=axes_row[2], orientation='horizontal',
            fraction=0.046, pad=0.07, aspect=30, shrink=0.92,
        )
        cbar_d.set_label('Delta (dias)', fontsize=8)
        cbar_d.ax.tick_params(labelsize=7)

        row_axes.append(axes_row)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight='bold', y=0.98)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.06)
    return fig


def load_delta_by_criterion():
    df = pd.read_csv(RESUMEN_CSV)
    rows = []
    for esc in CRITERIA_ORDER:
        sub = df[(df['escenario'] == esc) & (df['dataset'].str.contains('DELTA', na=False))]
        for _, r in sub.iterrows():
            metric = 'mean' if 'media' in r['metrica'].lower() else 't99'
            rows.append({
                'criterio': CRITERIA_LABELS[esc],
                'metric': metric,
                'delta': r['mean_espacial'],
            })
    return pd.DataFrame(rows)


def add_criteria_bars(fig, gs_bot, boot_df):
    ax = fig.add_subplot(gs_bot[0, 0])
    deltas = load_delta_by_criterion()
    x = np.arange(len(CRITERIA_ORDER))
    width = 0.35

    mean_vals = [deltas[(deltas['criterio'] == CRITERIA_LABELS[e]) & (deltas['metric'] == 'mean')]['delta'].iloc[0]
                 for e in CRITERIA_ORDER]
    t99_vals = [deltas[(deltas['criterio'] == CRITERIA_LABELS[e]) & (deltas['metric'] == 't99')]['delta'].iloc[0]
                for e in CRITERIA_ORDER]

    bars1 = ax.bar(x - width / 2, mean_vals, width, label='Duracion media', color='#34495E', edgecolor='white')
    bars2 = ax.bar(x + width / 2, t99_vals, width, label='t99 (p99)', color='#8E44AD', edgecolor='white')

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([CRITERIA_LABELS[e] for e in CRITERIA_ORDER], fontsize=9)
    ax.set_ylabel('Delta espacial medio\n(ALADIN - CR2MET, dias)', fontsize=9)
    ax.set_title(
        'Sensibilidad al criterio de umbral — promedio espacial sobre Chile',
        fontsize=10, fontweight='bold',
    )
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(axis='y', alpha=0.25)

    # Anotaciones P48-P51
    notes = []
    row_a = boot_df[boot_df['Escenario'].str.contains('Opcion B') & boot_df['Metrica'].str.contains('media')].iloc[0]
    notes.append(
        f"Criterio i: media {mean_vals[0]:.1f} d; "
        f"{row_a['% ALADIN < CR2MET (sig.)']:.0f}% celdas con ALADIN sig. mas corto"
    )
    notes.append(f"Criterio ii: media {mean_vals[1]:.1f} d (sesgo umbral + modelo)")
    notes.append(f"Criterio iii: t99 {t99_vals[2]:.1f} d (mejora vs i, sesgo residual)")

    ax.text(
        0.02, 0.02, '\n'.join(notes), transform=ax.transAxes,
        fontsize=7.5, va='bottom', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#CCCCCC'),
    )

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, -10 if h < 0 else 4), textcoords='offset points',
                ha='center', va='top' if h < 0 else 'bottom', fontsize=7,
            )


def build_figure():
    if not CACHE.exists():
        raise FileNotFoundError(
            f'No existe {CACHE}. Ejecuta primero _generar_figuras_paper_faltantes.py'
        )

    maps = load_criterion_i_maps()
    boot_df = pd.read_csv(BOOT_CSV)

    fig = plt.figure(figsize=(14, 12))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.3, 1.0], hspace=0.22)
    gs_maps = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[0], hspace=0.40, wspace=0.05)
    gs_bars = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])

    row_specs = [('mean', 'Duracion media (dias)'), ('t99', 't99 — p99 (dias)')]
    col_titles = ['CR2MET (regrillado)', 'ALADIN historico', 'Delta = ALADIN - CR2MET']
    mask = maps['chile_mask']

    for row_i, (key, row_label) in enumerate(row_specs):
        cr2, ala, dlt = maps[key]
        vals = np.concatenate([cr2.where(mask).values.ravel(), ala.where(mask).values.ravel()])
        vals = vals[np.isfinite(vals)]
        vmin, vmax = float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98))
        dvals = dlt.where(mask).values.ravel()
        dvals = dvals[np.isfinite(dvals)]
        dv = max(float(np.nanpercentile(np.abs(dvals), 98)), 0.5)
        axes_row = []
        im_main = im_delta = None
        for col_i, (da, title) in enumerate(zip([cr2, ala, dlt], col_titles)):
            ax = fig.add_subplot(gs_maps[row_i, col_i], projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.4)
            ax.set_extent(CHILE_EXTENT)
            if col_i < 2:
                im_main = da.plot.pcolormesh(
                    ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
                    cmap='viridis', vmin=vmin, vmax=vmax, add_colorbar=False,
                )
            else:
                im_delta = da.plot.pcolormesh(
                    ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
                    cmap='RdBu_r', vmin=-dv, vmax=dv, add_colorbar=False,
                )
            ax.set_title(title, fontsize=9, fontweight='bold', pad=3)
            if col_i == 0:
                ax.text(-0.10, 0.5, row_label, transform=ax.transAxes,
                        fontsize=9, fontweight='bold', va='center', ha='center', rotation=90)
            axes_row.append(ax)

        cb_m = fig.colorbar(im_main, ax=axes_row[:2], orientation='horizontal', fraction=0.05, pad=0.05, aspect=45)
        cb_m.set_label(row_label, fontsize=7)
        cb_d = fig.colorbar(im_delta, ax=axes_row[2], orientation='horizontal', fraction=0.05, pad=0.05, aspect=28)
        cb_d.set_label('Delta (dias)', fontsize=7)

    add_criteria_bars(fig, gs_bars, boot_df)

    fig.suptitle(
        'Sesgo de persistencia de dry spells: ALADIN vs CR2MET (1980-2014)\n'
        'Mapas: criterio i (R0 = 5.285 mm) | Barras: comparacion criterios i / ii / iii',
        fontsize=12, fontweight='bold', y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.08)

    path = OUT / 'fig32_persistencia_biases_sec3_2.png'
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> {path}')
    return path


def build_spatial_only():
    """Version solo mapas (layout corregido)."""
    maps = load_criterion_i_maps()
    fig = draw_spatial_comparison_maps(
        maps,
        suptitle=(
            'Persistencia de dry spells — CR2MET vs ALADIN\n'
            'Criterio i (R0 = 5.285 mm) | 1980-2014'
        ),
        figsize=(11, 9.5),
    )
    path = OUT / 'fig32_persistencia_spatial_only.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> {path}')
    return path


if __name__ == '__main__':
    print('Generando figura §3.2...')
    build_figure()
    build_spatial_only()
    print('Listo.')
