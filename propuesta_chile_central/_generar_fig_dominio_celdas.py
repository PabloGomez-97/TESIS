"""Dominio de estudio: las 3 587 celdas de ALADIN CHP12 dentro de Chile (17,7-45,9 S), las 4 regiones de Chile central y,
como referencia, la franja 32-40 S de Latoja et al. (2025). Entrada: _pregunta7_outputs/metrics_integrado_cache.npz (lat, lon, chile_mask).
Salida: paper_figuras/fig_dominio_celdas.png"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import unidades as U

ROOT = Path(__file__).resolve().parent
z = np.load(ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz')
lat, lon, ch = z['lat'], z['lon'], z['chile_mask']
zonas = dict(U.mascaras(lat, lon, ch))
fig = plt.figure(figsize=(5.6, 11.5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-77, -64.5, -47.5, -16.5], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN, facecolor='#e6f2fa'); ax.add_feature(cfeature.LAND, facecolor='#f3f3ee')
ax.add_feature(cfeature.COASTLINE, linewidth=0.6); ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':')
ax.scatter(lon[ch], lat[ch], s=2.2, c='#9a9a94', marker='s', linewidths=0, transform=ccrs.PlateCarree(), label=f'Celdas ALADIN en Chile ({int(ch.sum()):,})'.replace(',', ' '))
for k, m in zonas.items():
    ax.scatter(lon[m], lat[m], s=3.2, c=U.UNIDADES[k]['color'], marker='s', linewidths=0, transform=ccrs.PlateCarree())
for y in (32, 40):
    ax.plot([-77, -64.5], [-y, -y], color='#c0392b', lw=1, ls='--', transform=ccrs.PlateCarree())
ax.text(-64.7, -36, 'Franja de\nLatoja et al.\n(2025)\n32°–40°S', color='#c0392b', fontsize=8.5, ha='right', va='center', transform=ccrs.PlateCarree())
for y in (lat[ch].min(), lat[ch].max()):
    ax.plot([-77, -64.5], [y, y], color='#333', lw=0.8, ls=':', transform=ccrs.PlateCarree())
ax.text(-76.8, lat[ch].max() + 0.25, f'{abs(lat[ch].max()):.1f}°S'.replace('.', ','), fontsize=8.5, transform=ccrs.PlateCarree())
ax.text(-76.8, lat[ch].min() + 0.25, f'{abs(lat[ch].min()):.1f}°S'.replace('.', ','), fontsize=8.5, transform=ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, ylocs=range(-46, -16, 4), xlocs=[-76, -72, -68])
gl.top_labels = gl.right_labels = False
h = [mp.Patch(color='#9a9a94', label=f'Dominio: {int(ch.sum()):,} celdas ALADIN en Chile'.replace(',', ' '))]
h += [mp.Patch(color=v['color'], label=v['etiqueta']) for v in U.UNIDADES.values()]
ax.legend(handles=h, loc='lower right', fontsize=7.5, framealpha=0.95)
ax.set_title('Dominio de estudio y regiones de Chile central\n(grilla ALADIN CHP12, ≈ 12 km)', fontsize=10.5, fontweight='bold')
out = ROOT / 'paper_figuras' / 'fig_dominio_celdas.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out, int(ch.sum()), lat[ch].min(), lat[ch].max())
