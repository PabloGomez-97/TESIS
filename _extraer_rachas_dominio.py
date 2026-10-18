"""Extrae TODAS las rachas secas de cada celda del dominio y las guarda en cache.

Es la base comun de PDFs, Tabla 3, risk ratios y mapas: se extraen una sola vez, con una sola regla.
Reglas: dia seco = pr < umbral; se excluyen las rachas censuradas (las que empiezan el primer dia
o terminan el ultimo dia de la ventana); la racha se etiqueta con el anio y mes de su inicio.

Salida: resultados_tesis/rachas_<conjunto>.npz con arreglos cell, start_year, start_month, duration
y resultados_tesis/grilla.npz con lat, lon, mascara de Chile e indice de region.
"""
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from geografia_tesis import region_index

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'
R0 = 5.285            # umbral de dia humedo calibrado de ALADIN (mm/dia)
R_CR2MET = 1.0
HIST = ('1980-01-01', '2014-12-31')
FUT = ('2040-01-01', '2074-12-31')

# nombre -> (producto, periodo, escenario, umbral)
CONJUNTOS = {
    'cr2met_hist':       ('cr2met', HIST, None, R_CR2MET),
    'aladin_hist':       ('aladin', HIST, None, R0),
    'aladin_hist_1mm':   ('aladin', HIST, None, 1.0),
    'aladin_ssp585':     ('aladin', FUT, 'ssp585', R0),
    'aladin_ssp585_1mm': ('aladin', FUT, 'ssp585', 1.0),
    'aladin_ssp370':     ('aladin', FUT, 'ssp370', R0),
}


def open_aladin(periodo, escenario):
    patron = 'pr_CHP12_*_historical_*.nc' if escenario is None else f'pr_CHP12_*_{escenario}_*.nc'
    archivos = sorted(str(p) for p in (ROOT / 'pr1').glob(patron))
    ds = xr.open_mfdataset(archivos, use_cftime=True, chunks={'time': 365})
    return ds['pr'].sel(time=slice(*periodo)) * 86400.0


def open_cr2met(periodo, lat2d_da, lon2d_da):
    archivos = sorted(str(p) for p in (ROOT / 'pr').glob('CR2MET_pr_v2.5_day_*.nc'))
    ds = xr.open_mfdataset(archivos, chunks={'time': 365})
    return ds['pr'].sel(time=slice(*periodo)).interp(lat=lat2d_da, lon=lon2d_da, method='linear')


def a_celdas(da, jj, ii):
    return da.isel(y=xr.DataArray(jj, dims='p'), x=xr.DataArray(ii, dims='p')).compute().values


def extraer(pr, umbral, anios, meses):
    """pr: (tiempo, celdas). Devuelve arreglos de rachas no censuradas."""
    n = pr.shape[0]
    secos = (pr < umbral)
    cell, sy, sm, dur = [], [], [], []
    for c in range(pr.shape[1]):
        x = np.r_[False, secos[:, c], False].astype(np.int8)
        dx = np.diff(x)
        s, e = np.where(dx == 1)[0], np.where(dx == -1)[0]
        ok = (s > 0) & (e < n)          # excluye censuradas
        s, e = s[ok], e[ok]
        cell.append(np.full(s.size, c, dtype=np.int32))
        sy.append(anios[s].astype(np.int16))
        sm.append(meses[s].astype(np.int8))
        dur.append((e - s).astype(np.int32))
    return (np.concatenate(cell), np.concatenate(sy), np.concatenate(sm), np.concatenate(dur))


def main(solo=None):
    OUT.mkdir(exist_ok=True)
    ref = open_aladin(HIST, None)
    lat2d, lon2d = ref['lat'].values, ref['lon'].values
    mask, idx = region_index(lat2d, lon2d)
    jj, ii = np.where(mask)
    np.savez(OUT / 'grilla.npz', lat=lat2d, lon=lon2d, mask=mask, region_idx=idx, jj=jj, ii=ii)
    print(f'Celdas Chile: {mask.sum()} | en las 14 regiones: {(idx >= 0).sum()}', flush=True)

    for nombre, (producto, periodo, esc, umbral) in CONJUNTOS.items():
        if solo and nombre not in solo:
            continue
        print(f'Extrayendo {nombre}...', flush=True)
        if producto == 'aladin':
            da = open_aladin(periodo, esc)
        else:
            da = open_cr2met(periodo, ref['lat'], ref['lon'])
        pr = a_celdas(da, jj, ii)
        anios = da['time'].dt.year.values
        meses = da['time'].dt.month.values
        assert not np.isnan(pr).any(), f'{nombre}: hay NaN'
        cell, sy, sm, dur = extraer(pr, umbral, anios, meses)
        np.savez(OUT / f'rachas_{nombre}.npz', cell=cell, start_year=sy, start_month=sm, duration=dur,
                 n_dias=pr.shape[0], umbral=umbral)
        print(f'   {nombre}: {dur.size} rachas | media {dur.mean():.2f} d | max {dur.max()} d', flush=True)


if __name__ == '__main__':
    main(solo=sys.argv[1:] or None)
