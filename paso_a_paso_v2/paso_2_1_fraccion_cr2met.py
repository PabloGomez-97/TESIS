"""Paso 2.1 (v2) — Fraccion de dias humedos de CR2MET con 0,1 mm en el dominio (1980-2014): el numero objetivo que ALADIN debera igualar (paso 2.3).

Definicion: dia humedo = pr >= 0,1 mm/dia; dia seco = pr < 0,1 mm/dia (misma convencion que el analisis de rachas: seco si pr < umbral).
Fraccion del dominio = (dias humedos de todas las celdas) / (dias x celdas) = 972 celdas x 12 784 dias, sin NaN, asi que es el promedio simple de las
fracciones de las celdas. Se informa tambien por region, por celda (distribucion), por mes y por anio.
Entrada: salidas/paso_1_1_cr2met_interpolado_dominio.nc
Salidas: salidas/paso_2_1_fraccion_humeda_cr2met.csv, salidas/paso_2_1_objetivo_F.json, salidas/paso_2_1_chequeos.txt,
         figuras/paso_2_1_fraccion_humeda_cr2met.png
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
UMBRAL = 0.1
cr = xr.open_dataset(ROOT / 'salidas' / 'paso_1_1_cr2met_interpolado_dominio.nc')
m = np.load(ROOT / 'salidas' / 'paso_0_1_mascara_dominio.npz')
nombres = list(m['nombres'])
p = cr['pr'].values.astype('float64')           # (dias, celdas) en mm/dia
t = pd.DatetimeIndex(cr['time'].values)
rid = cr['region_id'].values
lat, lon = cr['lat_celda'].values, cr['lon_celda'].values
nd, nc = p.shape
chk = [f'Matriz: {nd} dias x {nc} celdas = {p.size} celda-dias; NaN: {int(np.isnan(p).sum())}']

hum = p >= UMBRAL
F_dom = hum.mean() * 100
F_cel = hum.mean(axis=0) * 100
chk.append(f'Fraccion de dias humedos (pr >= {UMBRAL} mm) del dominio: {F_dom:.4f} % ({int(hum.sum())} de {hum.size} celda-dias)')
chk.append(f'Promedio simple de las fracciones por celda: {F_cel.mean():.4f} % (debe coincidir: mismos dias en todas las celdas)')
# sensibilidad a la definicion en el borde: dias con valor exactamente 1,000
exacto = int((p == UMBRAL).sum())
chk.append(f'Celda-dias con pr exactamente {UMBRAL} mm (los afectaria usar > en vez de >=): {exacto}; fraccion con > {UMBRAL}: {(p > UMBRAL).mean() * 100:.4f} %')
# sensibilidad a redondeo del valor: 0,95 mm y 1,05 mm
for u in (0.05, 0.095, 0.105, 0.2):
    chk.append(f'Para ver cuan sensible es al umbral exacto: fraccion con pr >= {u} mm: {(p >= u).mean() * 100:.3f} %')

filas = []
for r, n in enumerate(nombres):
    k = rid == r
    filas.append(dict(nivel='region', nombre=n, celdas=int(k.sum()), F_pct=hum[:, k].mean() * 100,
                      F_celda_min=F_cel[k].min(), F_celda_mediana=np.median(F_cel[k]), F_celda_max=F_cel[k].max()))
filas.append(dict(nivel='dominio', nombre='Dominio (unión)', celdas=nc, F_pct=F_dom, F_celda_min=F_cel.min(),
                  F_celda_mediana=np.median(F_cel), F_celda_max=F_cel.max()))
df = pd.DataFrame(filas)
mes = pd.Series(hum.mean(axis=1) * 100, index=t).groupby(t.month).mean()
anio = pd.Series(hum.mean(axis=1) * 100, index=t).groupby(t.year).mean()
for mm_, v in mes.items():
    df.loc[len(df)] = dict(nivel='mes', nombre=f'mes {mm_:02d}', celdas=nc, F_pct=v, F_celda_min=np.nan, F_celda_mediana=np.nan, F_celda_max=np.nan)
for yy, v in anio.items():
    df.loc[len(df)] = dict(nivel='anio', nombre=str(yy), celdas=nc, F_pct=v, F_celda_min=np.nan, F_celda_mediana=np.nan, F_celda_max=np.nan)
df.to_csv(ROOT / 'salidas' / 'paso_2_1_fraccion_humeda_cr2met.csv', index=False)
json.dump(dict(periodo='1980-2014', celdas=nc, umbral_mm=UMBRAL, definicion='dia humedo: pr >= umbral', F_cr2met_dominio_pct=float(F_dom),
               F_por_region_pct={n: float(hum[:, rid == r].mean() * 100) for r, n in enumerate(nombres)}),
          open(ROOT / 'salidas' / 'paso_2_1_objetivo_F.json', 'w'), indent=2, ensure_ascii=False)

chk.append('Por region (%): ' + '; '.join(f'{n} {hum[:, rid == r].mean() * 100:.2f}' for r, n in enumerate(nombres)))
chk.append(f'Por celda: min {F_cel.min():.2f} %, mediana {np.median(F_cel):.2f} %, max {F_cel.max():.2f} %')
chk.append('Por mes (%): ' + '; '.join(f'{k:02d}: {v:.1f}' for k, v in mes.items()))
chk.append(f'Por anio: min {anio.min():.2f} % ({anio.idxmin()}), max {anio.max():.2f} % ({anio.idxmax()}), desviacion estandar {anio.std():.2f} pp')
mar_nov = (t.month >= 3) & (t.month <= 11)
chk.append(f'Fraccion marzo-noviembre: {hum[mar_nov].mean() * 100:.2f} %; diciembre-febrero: {hum[~mar_nov].mean() * 100:.2f} %; porcentaje de los dias humedos que caen en mar-nov: {100 * hum[mar_nov].sum() / hum.sum():.1f} %')
(ROOT / 'salidas' / 'paso_2_1_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))

fig, axs = plt.subplots(1, 3, figsize=(16, 6.5), gridspec_kw={'width_ratios': [1.1, 1, 1]})
ax = axs[0]
sc = ax.scatter(lon, lat, c=F_cel, s=26, marker='s', cmap='viridis')
ax.set_aspect(1 / np.cos(np.deg2rad(34))); ax.set_xlabel('longitud'); ax.set_ylabel('latitud')
ax.set_title('Fracción de días húmedos por celda (%)\nCR2MET, ≥ 0,1 mm, 1980–2014', fontsize=10)
fig.colorbar(sc, ax=ax, shrink=0.7)
ax = axs[1]
cols = ['#7B68B5', '#2C7FB8', '#2AA198', '#253494', '#8C6D31', '#D95F02', '#B2182B']
vals = [hum[:, rid == r].mean() * 100 for r in range(len(nombres))] + [F_dom]
ax.bar(range(len(vals)), vals, color=cols + ['#888888'])
for i, v in enumerate(vals):
    ax.text(i, v + 0.3, f'{v:.2f}'.replace('.', ','), ha='center', fontsize=9)
ax.set_xticks(range(len(vals))); ax.set_xticklabels([n.replace('Metropolitana', 'Metrop.').replace('La Araucanía', 'Araucanía') for n in nombres] + ['Dominio'], fontsize=8, rotation=30)
ax.set_ylabel('% de días húmedos'); ax.set_title('Por región y para el dominio', fontsize=10)
ax = axs[2]
ax.bar(mes.index, mes.values, color='#555555')
ax.set_xticks(range(1, 13)); ax.set_xlabel('mes'); ax.set_ylabel('% de días húmedos')
ax.set_title('Por mes (promedio de las 972 celdas)', fontsize=10)
fig.tight_layout()
(ROOT / 'figuras').mkdir(exist_ok=True)
fig.savefig(ROOT / 'figuras' / 'paso_2_1_fraccion_humeda_cr2met.png', dpi=150)
