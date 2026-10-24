"""Fraccion de la precipitacion anual que cae entre marzo y noviembre en las 4 regiones de los risk ratios.

Respalda (o cuestiona) la restriccion de los risk ratios a rachas que empiezan entre marzo y noviembre (§2.6).
Para cada region (mismas mascaras que _run_pregunta5_aladin.py) se suman los dias reales de cada mes y se promedian
los acumulados mensuales de 1980-2014. Se respeta la duracion de los meses y los anios bisiestos, sin asumir meses
de igual duracion. Se calcula el promedio espacial por celda y la fraccion mar-nov del total anual.
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


def climatologia_mensual(da, mask):
    """Media de acumulados mensuales de calendario (mm/mes), con igual peso por celda.

    Los meses con dias sin dato no se convierten en acumulados parciales; fuera de
    la mascara se mantienen los NaN. La suma en float64 evita acumular en float32.
    """
    fechas = pd.DatetimeIndex(da['time'].values)
    esperadas = pd.date_range(f'{fechas[0].year}-01-01', f'{fechas[-1].year}-12-31', freq='D')
    if not fechas.equals(esperadas):
        raise ValueError('Se requieren anios completos de datos diarios, sin huecos ni duplicados.')
    sel = da.where(mask).astype(np.float64)
    mensuales = sel.resample(time='MS').sum('time', skipna=False)
    clim = mensuales.groupby('time.month').mean('time')
    return clim.mean(dim=('y', 'x')).compute().values


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
