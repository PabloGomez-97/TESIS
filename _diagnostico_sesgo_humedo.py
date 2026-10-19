"""Diagnostico del sesgo humedo de ALADIN: ¿el exceso de precipitacion esta en ALADIN o en CR2MET?

Tres comprobaciones sobre 1980-2014 (precipitacion anual media):
 1. Incertidumbre interna de CR2MET (pr_sd, desviacion estandar del ensamble de 50 miembros) por region.
 2. Razon ALADIN/CR2MET por posicion oeste-centro-este (terciles dentro de cada banda de latitud de 0,25 grados y region):
    si el exceso fuera un artefacto de CR2MET en altura, la razon crecería mucho hacia la cordillera.
 3. Valores en la celda mas cercana a siete ciudades (orden de magnitud conocido; VERIFICAR contra estaciones).

Salidas: resultados_tesis/sesgo_humedo_pr_sd_regiones.csv, sesgo_humedo_terciles.csv, sesgo_humedo_macrozonas.csv, sesgo_humedo_ciudades.csv
"""
import warnings
warnings.filterwarnings('ignore')

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import _extraer_rachas_dominio as E
from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'
CIUDADES = {'Arica': (-18.48, -70.31), 'Antofagasta': (-23.65, -70.40), 'La Serena': (-29.91, -71.25),
            'Santiago': (-33.45, -70.68), 'Concepción': (-36.83, -73.05), 'Temuco': (-38.74, -72.60),
            'Puerto Montt': (-41.47, -72.94)}


def main():
    g = np.load(OUT / 'grilla.npz')
    jj, ii = g['jj'], g['ii']
    ref = E.open_aladin(E.HIST, None)
    ala2d = ref.mean('time').compute().values * 365
    ds = xr.open_mfdataset(sorted(glob.glob(str(ROOT / 'pr' / 'CR2MET_pr_v2.5_day_*.nc'))), chunks={'time': 365})
    sel = slice('1980-01-01', '2014-12-31')
    interp = lambda da: da.compute().interp(lat=ref['lat'], lon=ref['lon'], method='linear').values
    cr2d = interp(ds['pr'].sel(time=sel).mean('time')) * 365
    sd2d = interp(ds['pr_sd'].sel(time=sel).mean('time'))
    ala, cr, sd = ala2d[jj, ii], cr2d[jj, ii], sd2d[jj, ii]
    reg, lat, lon = g['region_idx'][jj, ii], g['lat'][jj, ii], g['lon'][jj, ii]

    filas = []
    for k, n in enumerate(NOMBRES_CORTOS):
        m = reg == k
        filas.append(dict(region=n, cr2met_mm=cr[m].mean(), aladin_mm=ala[m].mean(), razon=ala[m].mean() / cr[m].mean(),
                          cr2met_pr_sd_mm_dia=sd[m].mean(), sd_sobre_media=sd[m].mean() / (cr[m].mean() / 365)))
    pd.DataFrame(filas).to_csv(OUT / 'sesgo_humedo_pr_sd_regiones.csv', index=False)

    ter = np.full(len(reg), -1)
    for k in range(len(NOMBRES_CORTOS)):
        idx = np.where(reg == k)[0]
        bandas = np.floor(lat[idx] / 0.25)
        for b in np.unique(bandas):
            mm = idx[bandas == b]
            r = np.argsort(np.argsort(lon[mm])) / max(1, len(mm) - 1)
            ter[mm] = np.minimum(2, (r * 3).astype(int))
    filas = []
    for z in MACROZONAS:
        ks = [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == z]
        for t, nom in enumerate(['oeste', 'centro', 'este']):
            m = np.isin(reg, ks) & (ter == t)
            filas.append(dict(macrozona=z, posicion=nom, celdas=int(m.sum()), cr2met_mm=cr[m].mean(),
                              aladin_mm=ala[m].mean(), razon=ala[m].mean() / cr[m].mean()))
    pd.DataFrame(filas).to_csv(OUT / 'sesgo_humedo_terciles.csv', index=False)

    # Resumen por macrozona y dominio (Tabla A.4): totales, razon, razon por posicion y dispersion interna de CR2MET.
    filas = []
    unidades = [(z, [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == z]) for z in MACROZONAS]
    unidades.append(('14 regiones', list(range(len(NOMBRES_CORTOS)))))
    for z, ks in unidades:
        m = np.isin(reg, ks)
        fila = dict(unidad=z, celdas=int(m.sum()), cr2met_mm=cr[m].mean(), aladin_mm=ala[m].mean(),
                    razon=ala[m].mean() / cr[m].mean(), sd_sobre_media=sd[m].mean() / (cr[m].mean() / 365))
        for t, nom in enumerate(['oeste', 'centro', 'este']):
            mt = m & (ter == t)
            fila[f'razon_{nom}'] = ala[mt].mean() / cr[mt].mean()
        filas.append(fila)
    pd.DataFrame(filas).to_csv(OUT / 'sesgo_humedo_macrozonas.csv', index=False)

    la2, lo2 = ref['lat'].values, ref['lon'].values
    filas = []
    for n, (a, b) in CIUDADES.items():
        d = (la2 - a) ** 2 + (lo2 - b) ** 2
        j, i = np.unravel_index(np.argmin(d), d.shape)
        filas.append(dict(ciudad=n, lat_celda=float(la2[j, i]), lon_celda=float(lo2[j, i]), cr2met_mm=cr2d[j, i],
                          aladin_mm=ala2d[j, i], razon=ala2d[j, i] / cr2d[j, i]))
    pd.DataFrame(filas).to_csv(OUT / 'sesgo_humedo_ciudades.csv', index=False)
    for f in ('sesgo_humedo_macrozonas', 'sesgo_humedo_ciudades'):
        print(f); print(pd.read_csv(OUT / f'{f}.csv').round(2).to_string(index=False))


if __name__ == '__main__':
    main()
