"""Paso 1.1 — Interpolacion bilineal de CR2MET v2.5 (lluvia diaria `pr`, 0,05 deg) a los centros de las 502 celdas del dominio de ALADIN.

Entrada: salidas/paso_0_1_mascara_dominio.npz (centros de celda y mascara del dominio), pr/CR2MET_pr_v2.5_day_*.nc (1980-2014).
Metodo: interpolacion lineal en latitud y longitud (xarray .interp, method='linear'), punto a punto, dia por dia. Se recorta
CR2MET a la caja del dominio mas un margen de 0,5 deg (>> 2 celdas CR2MET) solo para ahorrar memoria: no afecta el resultado.
Salidas: salidas/paso_1_1_cr2met_interpolado_dominio.nc, salidas/paso_1_1_chequeos.txt, figuras/paso_1_1_interpolacion_dia.png
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INI, FIN = '1980-01-01', '2014-12-31'
MARGEN = 0.5

m = np.load(ROOT / 'salidas' / 'paso_0_1_mascara_dominio.npz')
dom, region_id, nombres = m['dominio'], m['region_id'], list(m['nombres'])
jj, ii = np.where(dom)
lat_c, lon_c = m['lat'][jj, ii], m['lon'][jj, ii]
rid = region_id[jj, ii]
ncel = lat_c.size

# 1) CR2MET, 1980-2014, recortado a la caja del dominio + margen
ds = xr.open_mfdataset(sorted(glob.glob(str(ROOT / 'pr' / 'CR2MET_pr_v2.5_day_*.nc'))), chunks={'time': 366}, combine='by_coords')
pr = ds['pr'].sel(time=slice(INI, FIN))
caja = dict(lat=slice(lat_c.min() - MARGEN, lat_c.max() + MARGEN), lon=slice(lon_c.min() - MARGEN, lon_c.max() + MARGEN))
pr = pr.sel(**caja).load()
print('CR2MET recortado:', pr.shape, float(pr.lat.min()), float(pr.lat.max()), float(pr.lon.min()), float(pr.lon.max()))
ndias = pr.sizes['time']

# 2) Interpolacion bilineal a los 502 centros (punto a punto)
tgt_lat = xr.DataArray(lat_c, dims='cell')
tgt_lon = xr.DataArray(lon_c, dims='cell')
interp = pr.interp(lat=tgt_lat, lon=tgt_lon, method='linear').astype('float32')
interp = interp.assign_coords(cell=np.arange(ncel), lat_celda=('cell', lat_c), lon_celda=('cell', lon_c), region_id=('cell', rid))
interp.name = 'pr'
interp.attrs.update(units='mm/dia', description='CR2MET v2.5 interpolado linealmente (bilineal) a los centros de celda de ALADIN CHP12 del dominio Chile central')
interp.to_netcdf(ROOT / 'salidas' / 'paso_1_1_cr2met_interpolado_dominio.nc')

# 3) Chequeos
chk = []
t = pd.DatetimeIndex(pr.time.values)
chk.append(f'Periodo: {t[0].date()} a {t[-1].date()}; dias: {ndias}; dias unicos: {t.nunique()}; esperado 12784 (35 anios con 9 bisiestos)')
chk.append(f'Celdas: {ncel}')
# NaN
nan_raw = int(np.isnan(pr.values).sum())
v = interp.values
nan_int = int(np.isnan(v).sum())
chk.append(f'NaN en CR2MET recortado (caja completa, incluye mar/Argentina si corresponde): {nan_raw}')
chk.append(f'NaN en la serie interpolada (dias x celdas = {v.size}): {nan_int}; celdas con algun NaN: {int(np.isnan(v).any(axis=0).sum())}')
chk.append(f'Rango de pr interpolada: min {np.nanmin(v):.3f} mm, max {np.nanmax(v):.1f} mm; valores negativos: {int((v < 0).sum())}')

# vecinos: verificacion manual de una celda y un dia (bilineal "a mano")
la, lo = pr.lat.values, pr.lon.values
rng = np.random.default_rng(1)
dif_max = 0.0
for k in rng.choice(ncel, 5, replace=False):
    d = int(rng.integers(0, ndias))
    i0 = np.searchsorted(la, lat_c[k]) - 1
    j0 = np.searchsorted(lo, lon_c[k]) - 1
    wy = (lat_c[k] - la[i0]) / (la[i0 + 1] - la[i0])
    wx = (lon_c[k] - lo[j0]) / (lo[j0 + 1] - lo[j0])
    c = pr.values[d, i0:i0 + 2, j0:j0 + 2]
    a_mano = (1 - wy) * (1 - wx) * c[0, 0] + (1 - wy) * wx * c[0, 1] + wy * (1 - wx) * c[1, 0] + wy * wx * c[1, 1]
    dif_max = max(dif_max, abs(a_mano - v[d, k]))
chk.append(f'Bilineal calculada a mano frente a xarray (5 celdas x dias al azar): diferencia maxima {dif_max:.2e} mm')

# comparacion con vecino mas cercano y promedio de las 4 vecinas (efecto del suavizado)
nn = pr.sel(lat=tgt_lat, lon=tgt_lon, method='nearest').values
chk.append(f'Correlacion diaria interpolada-vecino mas cercano (todas las celdas): {np.corrcoef(v.ravel(), nn.ravel())[0, 1]:.4f}')
chk.append(f'Lluvia anual media del dominio: interpolada {np.nanmean(v) * 365.25:.0f} mm; vecino mas cercano {np.nanmean(nn) * 365.25:.0f} mm')
filas = []
for r, nom in enumerate(nombres):
    mk = rid == r
    filas.append((nom, mk.sum(), np.nanmean(v[:, mk]) * 365.25, 100 * np.nanmean(v[:, mk] >= 1.0), 100 * np.nanmean(nn[:, mk] >= 1.0)))
chk.append('Por region: celdas | lluvia anual interpolada (mm) | dias >= 1 mm interpolados (%) | dias >= 1 mm vecino mas cercano (%)')
for f in filas:
    chk.append(f'  {f[0]:<14} {int(f[1]):>4} | {f[2]:7.0f} | {f[3]:5.2f} | {f[4]:5.2f}')
chk.append(f'  Dominio        {ncel:>4} | {np.nanmean(v) * 365.25:7.0f} | {100 * np.nanmean(v >= 1.0):5.2f} | {100 * np.nanmean(nn >= 1.0):5.2f}')
(ROOT / 'salidas' / 'paso_1_1_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))

# 4) Figura: el dia de mayor lluvia media del dominio, campo fino de CR2MET y valores interpolados en las celdas
dmax = int(np.nanargmax(np.nanmean(v, axis=1)))
vmax = float(np.nanpercentile(pr.values[dmax], 99.5))
fig, axs = plt.subplots(1, 2, figsize=(11, 6.5), sharex=True, sharey=True)
ext = [lon_c.min() - 0.3, lon_c.max() + 0.3, lat_c.min() - 0.3, lat_c.max() + 0.3]
sel_la = (la > ext[2]) & (la < ext[3]); sel_lo = (lo > ext[0]) & (lo < ext[1])
im = axs[0].pcolormesh(lo[sel_lo], la[sel_la], pr.values[dmax][np.ix_(sel_la, sel_lo)], cmap='Blues', vmin=0, vmax=vmax, shading='nearest')
axs[0].set_title(f'CR2MET original (0,05°, ≈ 5 km)\n{t[dmax].date()}', fontsize=10)
axs[1].scatter(lon_c, lat_c, c=v[dmax], cmap='Blues', vmin=0, vmax=vmax, s=34, marker='s')
axs[1].set_title(f'CR2MET interpolado a los {ncel} centros de celda de ALADIN\n(mismo día, misma escala)', fontsize=10)
for ax in axs:
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect(1 / np.cos(np.deg2rad(34))); ax.set_xlabel('longitud')
axs[0].set_ylabel('latitud')
fig.colorbar(im, ax=axs, shrink=0.8, label='mm/día')
(ROOT / 'figuras').mkdir(exist_ok=True)
fig.savefig(ROOT / 'figuras' / 'paso_1_1_interpolacion_dia.png', dpi=160, bbox_inches='tight')
print('figura lista; dia mostrado:', t[dmax].date())
