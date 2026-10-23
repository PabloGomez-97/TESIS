"""Duracion media espacial de las rachas secas (1980-2014) de CR2MET (1 mm), ALADIN (1 mm) y ALADIN (R0 = 5,285 mm) sobre UN MISMO
conjunto de celdas: las 3 526 validas (>= 30 rachas) en CR2MET (1 mm) y ALADIN (R0), las del criterio i de la Tabla S1.
CR2MET y ALADIN (R0) salen de la cache de mapas del analisis de dominio completo; ALADIN (1 mm) se recalcula con las funciones de
_run_pregunta7_bootstrap.py (mismas reglas: censuradas excluidas, >= 30 rachas). Si ALADIN (1 mm) no tiene >= 30 rachas en alguna de esas celdas, se informa.
Salida: resultados_tesis/duracion_media_celdas_comunes_i.csv"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import _run_pregunta7_bootstrap as b

z = np.load(ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz')
ch = z['chile_mask']
com = np.isfinite(z['mean_cr2']) & np.isfinite(z['mean_ala']) & ch
pa = b.open_aladin_historical()
g = b.load_chile_geometry()
m = b.build_chile_mask_on_aladin_grid(pa['lat'].values, pa['lon'].values, g)
mk = xr.DataArray(m, coords={'y': pa['y'], 'x': pa['x'], 'lat': pa['lat'], 'lon': pa['lon']}, dims=['y', 'x'])
pa = b.normalize_daily_time(pa)
groups, _ = b.extract_spell_groups_by_year(pa.where(mk), 1.0, mk)
ny, nx = m.shape
a1 = b.full_period_metric(groups, mk.stack(cell=('y', 'x')).values, 'mean').reshape(ny, nx)
n_nan = int((com & ~np.isfinite(a1)).sum())
filas = [('CR2MET, 1 mm', float(z['mean_cr2'][com].mean())),
         ('ALADIN, 1 mm', float(a1[com & np.isfinite(a1)].mean())),
         ('ALADIN, R0 = 5,285 mm', float(z['mean_ala'][com].mean()))]
df = pd.DataFrame(filas, columns=['serie', 'duracion_media_dias'])
df['celdas'] = int(com.sum()); df['celdas_sin_dato_aladin_1mm'] = n_nan
df.to_csv(ROOT / 'resultados_tesis' / 'duracion_media_celdas_comunes_i.csv', index=False)
print(df.round(3).to_string(index=False))
