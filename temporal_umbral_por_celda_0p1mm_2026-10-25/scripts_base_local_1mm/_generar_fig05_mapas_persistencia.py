"""Figura 5: mapas de persistencia de dry spells, umbral local principal (CR2MET regrillado, ALADIN historico y diferencia), 1980-2014.

Los tres paneles de cada fila (CR2MET, ALADIN y Delta) muestran las mismas celdas: las celdas con al menos 30 rachas en ambos productos.
Dibuja la duracion media y el percentil 99 (t99) por celda (>= 30 rachas, rachas censuradas excluidas) leidos de la cache
_pregunta7_outputs/metrics_integrado_cache.npz que genera _run_pregunta7_bootstrap.py. No recalcula nada.
Las celdas de Chile de la grilla ALADIN CHP12 recibida cubren 17,7-45,9 S: el mapa se recorta a esa zona.
Salida: paper_figuras/fig05_mapas_persistencia_criterio_i.png
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper_figuras'
OUT.mkdir(exist_ok=True)
CACHE = ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz'


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


# Las celdas de Chile de la grilla ALADIN CHP12 entregada cubren 17,7-45,9 S: el mapa se recorta a esa zona.
MAP_EXTENT = [-76, -66, -46.5, -17]


def draw_spatial_comparison_maps(maps, *, suptitle=None, figsize=(11, 13)):
    """Mapas 2x3 con colorbars verticales al costado de cada fila, para no tapar el sur de Chile."""
    row_specs = [
        ('mean', 'Duración media (días)', 'Δ duración media (días)'),
        ('t99', 't99 (días)', 'Δ t99 (días)'),
    ]
    col_titles = ['CR2MET', 'ALADIN histórico', 'Δ = ALADIN − CR2MET']
    # Los tres paneles de cada fila usan las mismas celdas: Chile y >= 30 rachas en CR2MET y en ALADIN (cantidad determinada por los datos).
    mask = maps['chile_mask']

    fig = plt.figure(figsize=figsize)
    # Columnas: mapa CR2MET | mapa ALADIN | barra comun | espacio | mapa delta | barra delta
    gs = gridspec.GridSpec(
        2, 6, figure=fig, width_ratios=[1, 1, 0.07, 0.42, 1, 0.07], wspace=0.08, hspace=0.10,
    )
    letters = iter('abcdef')

    for row_i, (key, main_label, delta_label) in enumerate(row_specs):
        cr2, ala, dlt = maps[key]
        mask = maps['chile_mask'] & np.isfinite(cr2) & np.isfinite(ala)
        vals = np.concatenate([cr2.where(mask).values.ravel(), ala.where(mask).values.ravel()])
        vals = vals[np.isfinite(vals)]
        vmin, vmax = float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98))
        dvals = dlt.where(mask).values.ravel()
        dvals = dvals[np.isfinite(dvals)]
        dv = max(float(np.nanpercentile(np.abs(dvals), 98)), 0.5)

        im_main = im_delta = None
        for da, title, col in zip([cr2, ala, dlt], col_titles, [0, 1, 4]):
            ax = fig.add_subplot(gs[row_i, col], projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='#EEEEEE', zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.4)
            ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
            is_delta = col == 4
            im = da.where(mask).plot.pcolormesh(
                ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(), add_colorbar=False,
                cmap='RdBu_r' if is_delta else 'viridis',
                vmin=-dv if is_delta else vmin, vmax=dv if is_delta else vmax,
            )
            if is_delta:
                im_delta = im
            else:
                im_main = im
            ax.set_title(f'({next(letters)}) {title}', fontsize=10, fontweight='bold', pad=4)

        cax_m = fig.add_subplot(gs[row_i, 2])
        cb_m = fig.colorbar(im_main, cax=cax_m, extend='both')
        cb_m.set_label(main_label, fontsize=9)
        cb_m.ax.tick_params(labelsize=8)
        cax_m.set_box_aspect(18)

        cax_d = fig.add_subplot(gs[row_i, 5])
        cb_d = fig.colorbar(im_delta, cax=cax_d, extend='both')
        cb_d.set_label(delta_label, fontsize=9)
        cb_d.ax.tick_params(labelsize=8)
        cax_d.set_box_aspect(18)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight='bold', y=0.995)
    fig.subplots_adjust(left=0.02, right=0.93, top=0.97, bottom=0.02)
    return fig


def main():
    if not CACHE.exists():
        raise FileNotFoundError(f'No existe {CACHE}. Ejecuta primero _run_pregunta7_bootstrap.py')
    fig = draw_spatial_comparison_maps(load_criterion_i_maps())
    path = OUT / 'fig05_mapas_persistencia_criterio_i.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'-> {path}')


if __name__ == '__main__':
    main()
