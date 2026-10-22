"""Fraccion de la precipitacion anual que cae entre marzo y noviembre en las 4 regiones de los risk ratios.

Respalda (o cuestiona) la restriccion de los risk ratios a rachas que empiezan entre marzo y noviembre (§2.6).
Para cada region (mismas mascaras que _run_pregunta5_aladin.py) se calcula la precipitacion mensual media de la region
(promedio espacial de la climatologia mensual por celda, 1980-2014) y la fraccion mar-nov del total anual.
CR2MET se interpola linealmente a la grilla ALADIN (igual que en los demas scripts); ALADIN usa pr x 86 400.

Entrada: pr/ y pr1/ (historico 1980-2014). Salida: resultados_tesis/estacionalidad_marnov_regiones.csv
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.io.shapereader as shpreader

import _run_pregunta5_aladin as rr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis' / 'estacionalidad_marnov_regiones.csv'
DIAS_MES = 30.4375


def climatologia_mensual(da, mask):
    """Precipitacion mensual media (mm/mes) promediada sobre las celdas de la region: vector de 12."""
    sel = da.where(mask)
    clim = sel.groupby('time.month').mean('time')
    return clim.mean(dim=('y', 'x')).compute().values * DIAS_MES


def main():
    aladin = rr.open_aladin_period(rr.HIST_START, rr.HIST_END)
    cr2 = xr.open_mfdataset(str(ROOT / 'pr' / 'CR2MET_pr_v2.5_day_*.nc'), chunks={'time': 365})['pr'].sel(
        time=slice(rr.HIST_START, rr.HIST_END)).interp(lat=aladin['lat'], lon=aladin['lon'], method='linear')
    cr2 = cr2.assign_coords(time=pd.DatetimeIndex(cr2['time'].values).normalize())
    aladin = aladin.assign_coords(time=pd.DatetimeIndex([pd.Timestamp(str(t)[:10]) for t in aladin['time'].values]))
    reader = shpreader.Reader(shpreader.natural_earth(resolution='10m', category='cultural',
                                                      name='admin_1_states_provinces'))
    chile = [r for r in reader.records() if r.attributes.get('admin') == 'Chile']
    lat2d, lon2d = aladin['lat'].values, aladin['lon'].values
    filas = []
    for nombre, spec in rr.REGION_SPECS.items():
        rec = rr.get_region_record(spec['query'], chile)
        m = rr.build_region_mask(lat2d, lon2d, rec.geometry)
        mask = xr.DataArray(m, coords={'y': aladin['y'], 'x': aladin['x']}, dims=('y', 'x'))
        fila = {'region': nombre, 'celdas': int(m.sum())}
        for prod, da in (('CR2MET', cr2), ('ALADIN', aladin)):
            mens = climatologia_mensual(da, mask)
            fila[f'{prod}_total_mm_anio'] = mens.sum()
            fila[f'{prod}_marnov_mm'] = mens[2:11].sum()
            fila[f'{prod}_fraccion_marnov_pct'] = 100 * mens[2:11].sum() / mens.sum()
        filas.append(fila)
        print(fila, flush=True)
    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(filas).round(3).to_csv(OUT, index=False)


if __name__ == '__main__':
    main()
