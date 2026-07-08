"""Genera las 3 figuras faltantes del outline del paper."""
import warnings
warnings.filterwarnings('ignore')

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper_figuras'
OUT.mkdir(exist_ok=True)

CHILE_EXTENT = [-76, -65, -55, -17]
START_DATE = '1980-01-01'
END_DATE = '2014-12-31'
TAU_CR2MET_REF = 1.0
TAU_ALADIN_DOMINIO = 5.285
MIN_SPELLS_FOR_STATS = 30

BASIN_SPECS = {
    'Loa': {'bounds': (-69.8, -68.2, -24.0, -21.5), 'color': '#D4A017'},
    'Maipo': {'bounds': (-71.8, -69.8, -34.5, -33.2), 'color': '#C0392B'},
    'Maule': {'bounds': (-72.8, -71.0, -36.5, -34.8), 'color': '#E67E22'},
    'Biobio': {'bounds': (-73.8, -71.5, -38.5, -36.5), 'color': '#27AE60'},
}

REGION_SPECS = {
    'Coquimbo': {'query': 'coquimbo', 'color': '#8B0000'},
    "O'Higgins": {'query': 'higgins', 'color': '#FF8C00'},
    'La Araucania': {'query': 'araucan', 'color': '#228B22'},
    'Los Lagos': {'query': 'los lagos', 'color': '#4682B4'},
}


def normalize_text(text):
    text = unicodedata.normalize('NFKD', str(text).lower())
    return ''.join(ch for ch in text if not unicodedata.combining(ch))


def load_chile_geometry():
    reader = shpreader.Reader(
        shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
    )
    geoms = [
        r.geometry for r in reader.records()
        if r.attributes.get('NAME') == 'Chile' or r.attributes.get('ADMIN') == 'Chile'
    ]
    return unary_union(geoms)


def get_chile_admin1_records():
    reader = shpreader.Reader(
        shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
    )
    records = []
    for r in reader.records():
        attrs = r.attributes
        if (
            attrs.get('adm0_name') == 'Chile'
            or attrs.get('admin') == 'Chile'
            or 'chile' in normalize_text(attrs.get('adm0_name', ''))
        ):
            records.append(r)
    return records


def find_region_record(query, records):
    q = normalize_text(query)
    for rec in records:
        name = normalize_text(rec.attributes.get('name', ''))
        if q in name:
            return rec
    return None


def build_chile_mask_on_aladin_grid(lat2d, lon2d, geometry):
    prepared = prep(geometry)
    return np.fromiter(
        (
            prepared.contains(Point(float(x), float(y))) or geometry.touches(Point(float(x), float(y)))
            for y, x in zip(lat2d.ravel(), lon2d.ravel())
        ),
        dtype=bool,
        count=lat2d.size,
    ).reshape(lat2d.shape)


# ---------------------------------------------------------------------------
# Fig 2 — Dominio
# ---------------------------------------------------------------------------
def fig02_dominio():
    print('Fig 2: mapa de dominio...')
    chile_geom = load_chile_geometry()
    admin_records = get_chile_admin1_records()

    fig, ax = plt.subplots(figsize=(7.5, 14), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(CHILE_EXTENT)
    ax.add_feature(cfeature.LAND, facecolor='#F5F5F0', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#E8F4FC', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=2)

    for name, spec in REGION_SPECS.items():
        rec = find_region_record(spec['query'], admin_records)
        if rec is None:
            continue
        ax.add_geometries(
            [rec.geometry], ccrs.PlateCarree(),
            facecolor=spec['color'], edgecolor='white', linewidth=0.6,
            alpha=0.35, zorder=1,
        )
        rep = rec.geometry.representative_point()
        ax.text(
            rep.x, rep.y, name.replace('La Araucania', 'Araucania'),
            transform=ccrs.PlateCarree(), fontsize=7, ha='center', va='center',
            color='#1a1a1a', fontweight='bold', zorder=4,
        )

    for name, spec in BASIN_SPECS.items():
        lon_min, lon_max, lat_min, lat_max = spec['bounds']
        rect = Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=2.2, edgecolor=spec['color'], facecolor='none',
            transform=ccrs.PlateCarree(), zorder=5,
        )
        ax.add_patch(rect)
        cx, cy = 0.5 * (lon_min + lon_max), 0.5 * (lat_min + lat_max)
        ax.text(
            cx, cy, name, transform=ccrs.PlateCarree(),
            fontsize=9, ha='center', va='center', color=spec['color'],
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75),
            zorder=6,
        )

    legend_items = [
        mpatches.Patch(facecolor=spec['color'], edgecolor='white', alpha=0.5, label=f'Region {n}')
        for n, spec in REGION_SPECS.items()
    ]
    legend_items += [
        Line2D([0], [0], color=spec['color'], lw=2.5, label=f'Cuenca {n}')
        for n, spec in BASIN_SPECS.items()
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=7, framealpha=0.9, ncol=1)

    ax.set_title(
        'Dominio de estudio — Chile continental\n'
        'Cuencas hidrograficas (PDFs) y regiones administrativas (risk ratios)',
        fontweight='bold', fontsize=11,
    )
    plt.tight_layout()
    path = OUT / 'fig02_dominio_cuencas_regiones.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path}')
    return path


# ---------------------------------------------------------------------------
# Fig 4 — Panel calibracion (combina PNGs existentes)
# ---------------------------------------------------------------------------
def fig04_calibracion_panel():
    print('Fig 4: panel calibracion umbral...')
    src_dir = ROOT / '_pregunta9_calibracion_outputs'
    basins = ['Loa', 'Maule', 'Biobio']
    images = []
    for b in basins:
        p = src_dir / f'pdf_calibracion_{b}_logy.png'
        if not p.exists():
            raise FileNotFoundError(f'Falta {p}. Ejecuta pregunta9_calibracion_umbral_cuencas.ipynb')
        images.append(Image.open(p))

    fig, axes = plt.subplots(3, 1, figsize=(10, 16))
    for ax, img, b in zip(axes, images, basins):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Cuenca {b}', fontweight='bold', fontsize=11, pad=6)

    fig.suptitle(
        'Efecto de la calibracion del umbral en PDFs de dry spells (1980-2014)\n'
        'CR2MET 1 mm | ALADIN 1 mm | ALADIN R0 global (5.285 mm) | ALADIN R0 local',
        fontweight='bold', fontsize=12, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUT / 'fig04_calibracion_umbral_panel.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    for im in images:
        im.close()
    print(f'  -> {path}')
    return path


# ---------------------------------------------------------------------------
# Fig 5 — Mapas pregunta 7 (solo criterio integrado, mean + t99)
# ---------------------------------------------------------------------------
def run_lengths_1d(bool_series):
    x = np.asarray(bool_series, dtype=np.bool_)
    if x.size == 0:
        return np.array([], dtype=np.int16)
    padded = np.r_[False, x, False]
    dx = np.diff(padded.astype(np.int8))
    starts = np.where(dx == 1)[0]
    ends = np.where(dx == -1)[0]
    return (ends - starts).astype(np.int16)


def metrics_from_durations(durations):
    durations = np.asarray(durations, dtype=float)
    durations = durations[np.isfinite(durations) & (durations > 0)]
    n = int(durations.size)
    if n < MIN_SPELLS_FOR_STATS:
        return {'mean': np.nan, 't99': np.nan, 'n_spells': n}
    return {
        'mean': float(np.mean(durations)),
        't99': float(np.percentile(durations, 99)),
        'n_spells': n,
    }


def pixelwise_spell_metrics(pr_masked, dry_threshold, mask_da):
    is_dry = (pr_masked < dry_threshold).where(mask_da)
    dry_stacked = is_dry.stack(cell=('y', 'x')).transpose('time', 'cell').compute()
    dry_vals = dry_stacked.values
    out = {
        'mean': np.full(dry_vals.shape[1], np.nan, dtype=np.float32),
        't99': np.full(dry_vals.shape[1], np.nan, dtype=np.float32),
    }
    for idx in range(dry_vals.shape[1]):
        col = dry_vals[:, idx]
        if not np.any(col):
            continue
        m = metrics_from_durations(run_lengths_1d(col))
        out['mean'][idx] = m['mean']
        out['t99'][idx] = m['t99']
    maps = {}
    for key in ['mean', 't99']:
        da_1d = xr.DataArray(out[key], coords={'cell': dry_stacked['cell']}, dims=['cell'])
        da_2d = da_1d.unstack('cell')
        maps[key] = da_2d.assign_coords(lat=mask_da['lat'], lon=mask_da['lon'])
    return maps


def plot_three_panel_maps(cr2_da, ala_da, delta_da, chile_mask, *, title, cbar_label, save_path,
                          cmap_main='viridis', cmap_delta='RdBu_r', vmin=None, vmax=None, dv=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    for ax in axes:
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_extent(CHILE_EXTENT)

    cr2_da.plot.pcolormesh(
        ax=axes[0], x='lon', y='lat', transform=ccrs.PlateCarree(), cmap=cmap_main,
        vmin=vmin, vmax=vmax, add_colorbar=False,
    )
    axes[0].set_title('CR2MET (regrillado)', fontweight='bold')

    im1 = ala_da.plot.pcolormesh(
        ax=axes[1], x='lon', y='lat', transform=ccrs.PlateCarree(), cmap=cmap_main,
        vmin=vmin, vmax=vmax, add_colorbar=False,
    )
    axes[1].set_title('ALADIN historico', fontweight='bold')

    if dv is None:
        vals = delta_da.where(chile_mask).values.ravel()
        vals = vals[np.isfinite(vals)]
        dv = float(np.nanpercentile(np.abs(vals), 98)) if vals.size else 1.0
        dv = max(dv, 0.5)

    im2 = delta_da.plot.pcolormesh(
        ax=axes[2], x='lon', y='lat', transform=ccrs.PlateCarree(), cmap=cmap_delta,
        vmin=-dv, vmax=dv, add_colorbar=False,
    )
    axes[2].set_title('Delta = ALADIN - CR2MET', fontweight='bold')

    cbar0 = fig.colorbar(im1, ax=axes[:2].ravel().tolist(), orientation='horizontal', shrink=0.85, pad=0.08)
    cbar0.set_label(cbar_label)
    cbar1 = fig.colorbar(im2, ax=[axes[2]], orientation='horizontal', shrink=0.85, pad=0.08)
    cbar1.set_label('Delta (dias)')

    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig05_mapas_persistencia():
    print('Fig 5: mapas persistencia dry spells (pregunta 7, criterio integrado)...')
    cache = ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz'

    if cache.exists():
        print('  Cargando metricas desde cache...')
        z = np.load(cache, allow_pickle=True)
        lat = z['lat']
        lon = z['lon']
        chile_mask_bool = z['chile_mask']
        mean_cr2 = z['mean_cr2']
        mean_ala = z['mean_ala']
        t99_cr2 = z['t99_cr2']
        t99_ala = z['t99_ala']
        y = z['y']
        x = z['x']
    else:
        print('  Cargando datos (puede tardar varios minutos)...')
        pr_aladin = (
            xr.open_mfdataset(str(ROOT / 'pr1' / 'pr_CHP12_*_historical_*.nc'), use_cftime=True, chunks={'time': 365})
            ['pr'].sel(time=slice(START_DATE, END_DATE)) * 86400.0
        )
        pr_cr2met_native = (
            xr.open_mfdataset(str(ROOT / 'pr' / 'CR2MET_pr_v2.5_day_*.nc'), chunks={'time': 365})
            ['pr'].sel(time=slice(START_DATE, END_DATE))
        )
        chile_geom = load_chile_geometry()
        lat2d = pr_aladin['lat'].values
        lon2d = pr_aladin['lon'].values
        chile_mask_bool = build_chile_mask_on_aladin_grid(lat2d, lon2d, chile_geom)
        chile_mask = xr.DataArray(
            chile_mask_bool,
            coords={'y': pr_aladin['y'], 'x': pr_aladin['x'], 'lat': pr_aladin['lat'], 'lon': pr_aladin['lon']},
            dims=['y', 'x'],
        )
        pr_cr2met = pr_cr2met_native.interp(lat=pr_aladin['lat'], lon=pr_aladin['lon'], method='linear')
        pr_cr2met_chile = pr_cr2met.where(chile_mask)
        pr_aladin_chile = pr_aladin.where(chile_mask)

        print('  Calculando metricas CR2MET...')
        m_cr2 = pixelwise_spell_metrics(pr_cr2met_chile, TAU_CR2MET_REF, chile_mask)
        print('  Calculando metricas ALADIN...')
        m_ala = pixelwise_spell_metrics(pr_aladin_chile, TAU_ALADIN_DOMINIO, chile_mask)

        mean_cr2 = m_cr2['mean'].values
        mean_ala = m_ala['mean'].values
        t99_cr2 = m_cr2['t99'].values
        t99_ala = m_ala['t99'].values
        lat = lat2d
        lon = lon2d
        y = pr_aladin['y'].values
        x = pr_aladin['x'].values

        np.savez(
            cache,
            lat=lat, lon=lon, y=y, x=x, chile_mask=chile_mask_bool,
            mean_cr2=mean_cr2, mean_ala=mean_ala, t99_cr2=t99_cr2, t99_ala=t99_ala,
        )
        print(f'  Cache guardado: {cache}')

    chile_mask = xr.DataArray(
        chile_mask_bool,
        coords={'y': y, 'x': x, 'lat': (['y', 'x'], lat), 'lon': (['y', 'x'], lon)},
        dims=['y', 'x'],
    )

    def _da(arr):
        return xr.DataArray(arr, coords=chile_mask.coords, dims=['y', 'x'])

    metrics_cr2 = {'mean': _da(mean_cr2), 't99': _da(t99_cr2)}
    metrics_ala = {'mean': _da(mean_ala), 't99': _da(t99_ala)}
    metrics_delta = {
        'mean': metrics_ala['mean'] - metrics_cr2['mean'],
        't99': metrics_ala['t99'] - metrics_cr2['t99'],
    }

    threshold_note = (
        f'Dia seco: CR2MET < {TAU_CR2MET_REF:g} mm/dia | '
        f'ALADIN < {TAU_ALADIN_DOMINIO:g} mm/dia (R0 integrado)'
    )
    paths = []
    for key, label in [('mean', 'Duracion media'), ('t99', 't99 (p99)')]:
        cr2 = metrics_cr2[key]
        ala = metrics_ala[key]
        dlt = metrics_delta[key]
        vals = np.concatenate([
            cr2.where(chile_mask).values.ravel(),
            ala.where(chile_mask).values.ravel(),
        ])
        vals = vals[np.isfinite(vals)]
        vmin = float(np.nanpercentile(vals, 2)) if vals.size else None
        vmax = float(np.nanpercentile(vals, 98)) if vals.size else None
        fname = 'fig05_mapas_persistencia_mean.png' if key == 'mean' else 'fig05_mapas_persistencia_t99.png'
        path = OUT / fname
        plot_three_panel_maps(
            cr2, ala, dlt, chile_mask,
            title=f'{label} de dry spells (dias) | {START_DATE[:4]}-{END_DATE[:4]}\n{threshold_note}',
            cbar_label=f'{label} (dias)',
            save_path=path,
            vmin=vmin, vmax=vmax,
        )
        print(f'  -> {path}')
        paths.append(path)

    combined = OUT / 'fig05_mapas_persistencia_combined.png'
    imgs = [Image.open(p) for p in paths]
    w = max(im.width for im in imgs)
    h = sum(im.height for im in imgs)
    canvas = Image.new('RGB', (w, h), 'white')
    yoff = 0
    for im in imgs:
        canvas.paste(im, (0, yoff))
        yoff += im.height
    canvas.save(combined)
    for im in imgs:
        im.close()
    print(f'  -> {combined}')
    paths.append(combined)
    return paths


def main():
    print('Generando figuras faltantes del paper...\n')
    p1 = fig02_dominio()
    p2 = fig04_calibracion_panel()
    p3 = fig05_mapas_persistencia()
    print('\nListo.')
    print(f'Carpeta: {OUT.resolve()}')
    return p1, p2, p3


if __name__ == '__main__':
    main()
