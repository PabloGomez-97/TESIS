"""Figura S2: mapa de R0 local de ALADIN (criterio iii, §2.2 y §2.7), 1980-2014.

En cada pixel, R0 local es el umbral que iguala la fraccion de dias humedos de ALADIN a la de
CR2MET con 1 mm. Mascara, regrillado y biseccion identicos a la Figura 6 y la Figura S1.
"""
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

import _generar_pregunta9_pdf_loglog_cuencas as f6

CACHE = f6.ROOT / '_pregunta7_outputs' / 'R0_local_map.npz'
FIG = f6.PAPER / 'figS2_mapa_R0_local.png'


def compute_r0_local():
    if CACHE.exists():
        print(f'Cargando R0 local desde cache: {CACHE}')
        z = np.load(CACHE)
        return z['lat'], z['lon'], z['r0'], z['chile']

    pr_ala = f6.open_aladin_period(f6.HIST_START, f6.HIST_END)
    pr_cr2 = f6.open_cr2met_period(f6.HIST_START, f6.HIST_END).interp(
        lat=pr_ala['lat'], lon=pr_ala['lon'], method='linear',
    )
    lat2d, lon2d = pr_ala['lat'].values, pr_ala['lon'].values
    chile_bool = f6.build_chile_mask_on_aladin_grid(lat2d, lon2d, f6.load_chile_geometry())
    chile = xr.DataArray(chile_bool, dims=['y', 'x'],
                         coords={'y': pr_ala['y'], 'x': pr_ala['x'],
                                 'lat': pr_ala['lat'], 'lon': pr_ala['lon']})
    print(f'Biseccion en {int(chile_bool.sum())} pixeles...')
    r0 = f6.pixelwise_local_aladin_threshold(pr_cr2.where(chile), pr_ala.where(chile),
                                             f6.TAU_CR2MET_REF, chile).values
    np.savez(CACHE, lat=lat2d, lon=lon2d, r0=r0, chile=chile_bool)
    return lat2d, lon2d, r0, chile_bool


def main():
    lat, lon, r0, chile = compute_r0_local()
    valid = chile & np.isfinite(r0)
    vals = r0[valid]
    print(f'Pixeles Chile: {int(chile.sum())} | con R0 local: {int(valid.sum())} | '
          f'sin solucion: {int((chile & ~np.isfinite(r0)).sum())}')
    print(f'R0 local: mediana {np.median(vals):.2f} | p10 {np.percentile(vals, 10):.2f} | '
          f'p90 {np.percentile(vals, 90):.2f} | min {vals.min():.2f} | max {vals.max():.2f} mm/d')
    print(f'% pixeles con R0 local < {f6.TAU_ALADIN_DOMINIO}: {np.mean(vals < f6.TAU_ALADIN_DOMINIO) * 100:.1f}')

    fig = plt.figure(figsize=(5.2, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-76.5, -66, -56, -17], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', linewidth=0.6)
    pcm = ax.pcolormesh(lon, lat, np.where(valid, r0, np.nan), cmap='YlOrRd',
                        vmin=0, vmax=14, shading='auto', transform=ccrs.PlateCarree())
    cb = fig.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.03, fraction=0.04, shrink=0.9)
    cb.set_label('R₀ local de ALADIN (mm/día)')
    cb.ax.axvline(f6.TAU_ALADIN_DOMINIO, color='k', lw=1.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax.set_title('R₀ local de ALADIN (1980–2014)', fontweight='bold')
    fig.savefig(FIG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'-> {FIG}')


if __name__ == '__main__':
    main()
