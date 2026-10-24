"""Paso 1.2 (complemento) — Conteo de dias-celda >= 1 mm en CR2MET interpolado y ALADIN historico (dominio, 1980-2014).
Un 'dia-celda' es un dia en una celda: 12 784 dias x 502 celdas = 6 417 568 casos. Dia humedo: pr >= 1 mm.
Es una inspeccion; la fraccion oficial se calcula en los pasos 2.1 y 2.2.
Entradas: salidas/paso_1_1_cr2met_interpolado_dominio.nc, salidas/paso_1_2_aladin_hist_1980_2014.nc. Salida: salidas/paso_1_2_conteo_dias_humedos.csv"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
ROOT = Path(__file__).resolve().parent
filas = []
for nombre, f in (('CR2MET interpolado', 'paso_1_1_cr2met_interpolado_dominio.nc'), ('ALADIN historico', 'paso_1_2_aladin_hist_1980_2014.nc')):
    v = xr.open_dataset(ROOT / 'salidas' / f)['pr'].values
    h = int((v >= 1.0).sum())
    filas.append(dict(serie=nombre, dias=v.shape[0], celdas=v.shape[1], dias_celda_total=int(v.size), dias_celda_humedos_1mm=h,
                      fraccion_pct=100 * h / v.size, dias_con_exactamente_1mm=int((v == 1.0).sum()), nan=int(np.isnan(v).sum())))
df = pd.DataFrame(filas)
df.to_csv(ROOT / 'salidas' / 'paso_1_2_conteo_dias_humedos.csv', index=False)
print(df.round(4).to_string(index=False))
