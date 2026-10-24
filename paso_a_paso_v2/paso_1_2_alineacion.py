"""Paso 1.2 (v2) — Alinear periodos y calendarios.

- ALADIN CHP12 (CNRM-ESM2-1, r1i1p1f2): historico 1979-2014 -> se recorta a 1980-2014; SSP5-8.5 y SSP3-7.0, tramos 2040-2064 y 2065-2100 -> se pegan
  y se recortan a 2040-2074. Solo las celdas del dominio (972, paso 0.1). Unidad: kg m-2 s-1 -> mm/dia (x 86400; 1 kg m-2 = 1 mm).
- ALADIN marca cada dia a las 12:00; CR2MET a las 00:00. Se alinea por FECHA (anio-mes-dia).
- Se comprueba que las fechas de ALADIN historico y de CR2MET interpolado (paso 1.1) coinciden una a una.

Entradas: salidas/paso_0_1_mascara_dominio.npz, salidas/paso_1_1_cr2met_interpolado_dominio.nc, pr1/pr_CHP12_*.nc
Salidas: salidas/paso_1_2_aladin_hist_1980_2014.nc, salidas/paso_1_2_aladin_ssp585_2040_2074.nc, salidas/paso_1_2_aladin_ssp370_2040_2074.nc,
         salidas/paso_1_2_chequeos.txt
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
m = np.load(ROOT / 'salidas' / 'paso_0_1_mascara_dominio.npz')
dom, region_id, nombres = m['dominio'], m['region_id'], list(m['nombres'])
jj, ii = np.where(dom)
lat_c, lon_c, rid = m['lat'][jj, ii], m['lon'][jj, ii], region_id[jj, ii]
ncel = jj.size
chk = []


def fechas(time_cftime):
    """Fecha (datetime64[D]) de cada instante, ignorando la hora."""
    return np.array([np.datetime64(f'{x.year:04d}-{x.month:02d}-{x.day:02d}', 'D') for x in time_cftime])


def leer(archivos, ini, fin):
    """Concatena archivos de ALADIN, recorta por fecha y devuelve (fechas, pr en mm/dia [tiempo, celda])."""
    out_f, out_v = [], []
    for f in archivos:
        ds = xr.open_dataset(f, use_cftime=True)
        fe = fechas(ds['time'].values)
        sel = (fe >= np.datetime64(ini)) & (fe <= np.datetime64(fin))
        if not sel.any():
            continue
        # misma grilla que la del paso 0.1
        assert np.array_equal(ds['lat'].values[jj, ii], lat_c) and np.array_equal(ds['lon'].values[jj, ii], lon_c), f
        pr = ds['pr'].isel(time=np.where(sel)[0], y=xr.DataArray(jj, dims='cell'), x=xr.DataArray(ii, dims='cell')).load().values.astype('float64')
        out_f.append(fe[sel]); out_v.append((pr * 86400.0).astype('float32'))
        chk.append(f'  leido {Path(f).name}: {sel.sum()} dias usados ({fe[sel][0]} a {fe[sel][-1]}); unidad original {ds["pr"].attrs.get("units")}')
    return np.concatenate(out_f), np.concatenate(out_v, axis=0)


def revisar(nombre, fe, v, esperado):
    """Fechas consecutivas, sin repetidos, sin NaN, sin negativos; resumen de lluvia."""
    d = np.diff(fe.astype('datetime64[D]').astype(int))
    bis = int(((pd.DatetimeIndex(fe).month == 2) & (pd.DatetimeIndex(fe).day == 29)).sum())
    chk.append(f'{nombre}: {len(fe)} dias (esperado {esperado}); {fe[0]} a {fe[-1]}; fechas unicas {len(np.unique(fe))}; saltos distintos de 1 dia: {int((d != 1).sum())}; 29-feb: {bis}')
    chk.append(f'   NaN: {int(np.isnan(v).sum())}; negativos: {int((v < 0).sum())}; maximo diario {v.max():.1f} mm; promedio de dias >= 0,1 mm: {100 * np.mean(v >= 0.1):.2f} %')
    anual = [f'{n} {np.mean(v[:, rid == r]) * 365.25:.0f}' for r, n in enumerate(nombres)]
    chk.append(f'   lluvia anual media (mm) por region: ' + '; '.join(anual) + f'; dominio {np.mean(v) * 365.25:.0f}')


def guardar(fe, v, nombre_archivo, descr):
    da = xr.DataArray(v, dims=('time', 'cell'), coords=dict(time=pd.to_datetime(fe), cell=np.arange(ncel),
                      lat_celda=('cell', lat_c), lon_celda=('cell', lon_c), region_id=('cell', rid)), name='pr',
                      attrs=dict(units='mm/dia', description=descr))
    da.to_netcdf(ROOT / 'salidas' / nombre_archivo)


pr1 = ROOT / 'pr1'
hist_f = sorted(glob.glob(str(pr1 / 'pr_CHP12_*_historical_*_chile.nc')))
chk.append('ALADIN historico (1980-2014):')
fe_h, v_h = leer(hist_f, '1980-01-01', '2014-12-31')
revisar('ALADIN historico', fe_h, v_h, 12784)
guardar(fe_h, v_h, 'paso_1_2_aladin_hist_1980_2014.nc', 'ALADIN CHP12 historico, celdas del dominio, mm/dia, fechas diarias')

for esc, tag in (('ssp585', 'SSP5-8.5'), ('ssp370', 'SSP3-7.0')):
    fs = sorted(glob.glob(str(pr1 / f'pr_CHP12_*_{esc}_*_chile.nc')))
    chk.append(f'ALADIN {tag} (2040-2074):')
    fe_f, v_f = leer(fs, '2040-01-01', '2074-12-31')
    revisar(f'ALADIN {tag}', fe_f, v_f, 12784)
    guardar(fe_f, v_f, f'paso_1_2_aladin_{esc}_2040_2074.nc', f'ALADIN CHP12 {tag}, celdas del dominio, mm/dia, fechas diarias')

# Coincidencia de fechas ALADIN historico <-> CR2MET interpolado (paso 1.1)
cr = xr.open_dataset(ROOT / 'salidas' / 'paso_1_1_cr2met_interpolado_dominio.nc')
fe_c = cr['time'].values.astype('datetime64[D]')
chk.append('CR2MET (paso 1.1) frente a ALADIN historico:')
chk.append(f'   mismas fechas, una a una: {bool(np.array_equal(fe_c, fe_h))} ({len(fe_c)} frente a {len(fe_h)}); horas de CR2MET: {np.unique(pd.DatetimeIndex(cr["time"].values).hour)}')
chk.append(f'   celdas y regiones coinciden: {bool(np.array_equal(cr["lat_celda"].values, lat_c) and np.array_equal(cr["region_id"].values, rid))}')
# efecto de un desfase de un dia: correlacion diaria con y sin desfase (prueba de que la alineacion por fecha es la que mejor concuerda)
a, c = v_h.mean(axis=1), cr['pr'].values.mean(axis=1)
corr = {k: np.corrcoef(a[max(0, k):len(a) + min(0, k)], c[max(0, -k):len(c) + min(0, -k)])[0, 1] for k in (-2, -1, 0, 1, 2)}
chk.append('   correlacion diaria (promedio del dominio) con ALADIN adelantado/atrasado k dias: ' + '; '.join(f'k={k:+d}: {r:.3f}' for k, r in corr.items()))
(ROOT / 'salidas' / 'paso_1_2_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))
