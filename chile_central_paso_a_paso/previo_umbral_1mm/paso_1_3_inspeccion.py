"""Paso 1.3 — Inspeccion (no va al paper): lluvia diaria de CR2MET y de ALADIN lado a lado.

Panel A y B: dos celdas del dominio (la mas cercana a Santiago y la mas cercana a Talca), marzo-noviembre de un anio tipico.
  Anio tipico = el anio 1980-2014 cuya lluvia anual media del dominio en CR2MET es la mediana de los 35 anios (regla fijada antes de mirar).
Panel C: fraccion de dias del dominio (todas las celdas, 1980-2014) en cada tramo de intensidad diaria, CR2MET frente a ALADIN.
Entradas: salidas/paso_1_1_cr2met_interpolado_dominio.nc, salidas/paso_1_2_aladin_hist_1980_2014.nc
Salidas: figuras/paso_1_3_series_e_intensidad.png, salidas/paso_1_3_intensidad_por_tramo.csv, salidas/paso_1_3_chequeos.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
cr = xr.open_dataset(ROOT / 'salidas' / 'paso_1_1_cr2met_interpolado_dominio.nc')
al = xr.open_dataset(ROOT / 'salidas' / 'paso_1_2_aladin_hist_1980_2014.nc')
assert np.array_equal(cr['time'].values.astype('datetime64[D]'), al['time'].values.astype('datetime64[D]'))
t = pd.DatetimeIndex(cr['time'].values)
c, a = cr['pr'].values, al['pr'].values
lat, lon = cr['lat_celda'].values, cr['lon_celda'].values
chk = []

# anio tipico: mediana de la lluvia anual del dominio en CR2MET
anual = pd.Series(c.mean(axis=1), index=t).groupby(t.year).sum()
tipico = int(anual.index[np.argsort(anual.values)[len(anual) // 2]])
chk.append(f'Anio tipico (mediana de la lluvia anual media del dominio en CR2MET, 35 anios): {tipico} ({anual[tipico]:.0f} mm; mediana {anual.median():.0f} mm)')


def celda_cercana(la, lo):
    d2 = (lat - la) ** 2 + ((lon - lo) * np.cos(np.deg2rad(la))) ** 2
    return int(np.argmin(d2))


refs = {'Santiago': (-33.45, -70.67), 'Talca': (-35.43, -71.66)}
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.1], hspace=0.45)
mk = (t.year == tipico) & (t.month >= 3) & (t.month <= 11)
for k, (nombre, (la, lo)) in enumerate(refs.items()):
    j = celda_cercana(la, lo)
    ax = fig.add_subplot(gs[k])
    x = t[mk]
    ax.bar(x - pd.Timedelta(hours=5), c[mk, j], width=0.4, color='#555555', label='CR2MET (interpolado)')
    ax.bar(x + pd.Timedelta(hours=5), a[mk, j], width=0.4, color='#d4564e', label='ALADIN')
    ax.axhline(1.0, color='k', lw=0.6, ls=':')
    ax.text(x[0], 1.2, '1 mm', fontsize=8)
    ax.set_title(f'Celda cercana a {nombre} (lat {lat[j]:.2f}, lon {lon[j]:.2f}); marzo–noviembre de {tipico}', fontsize=10, loc='left')
    ax.set_ylabel('mm/día'); ax.legend(loc='upper right', fontsize=8)
    nc, na = int((c[mk, j] >= 1).sum()), int((a[mk, j] >= 1).sum())
    ax.text(0.01, 0.9, f'días ≥ 1 mm: CR2MET {nc}, ALADIN {na} (de {int(mk.sum())})', transform=ax.transAxes, fontsize=9, va='top')
    chk.append(f'{nombre}: celda {j}, region {int(cr["region_id"].values[j])}; marzo-noviembre {tipico}: dias >= 1 mm CR2MET {nc}, ALADIN {na} de {int(mk.sum())}')

# panel C: intensidad por tramos
bordes = [0, 0.1, 1, 5, 10, 25, np.inf]
etq = ['< 0,1', '0,1 – 1', '1 – 5', '5 – 10', '10 – 25', '≥ 25']
fc = np.array([np.mean((c >= lo) & (c < hi)) * 100 for lo, hi in zip(bordes[:-1], bordes[1:])])
fa = np.array([np.mean((a >= lo) & (a < hi)) * 100 for lo, hi in zip(bordes[:-1], bordes[1:])])
pd.DataFrame({'tramo_mm_dia': etq, 'CR2MET_pct_dias': fc, 'ALADIN_pct_dias': fa}).to_csv(ROOT / 'salidas' / 'paso_1_3_intensidad_por_tramo.csv', index=False)
ax = fig.add_subplot(gs[2])
xx = np.arange(len(etq))
ax.bar(xx - 0.2, fc, 0.4, color='#555555', label='CR2MET (interpolado)')
ax.bar(xx + 0.2, fa, 0.4, color='#d4564e', label='ALADIN')
for i in range(len(etq)):
    ax.text(xx[i] - 0.2, fc[i] + 0.8, f'{fc[i]:.1f}'.replace('.', ','), ha='center', fontsize=8)
    ax.text(xx[i] + 0.2, fa[i] + 0.8, f'{fa[i]:.1f}'.replace('.', ','), ha='center', fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels(etq); ax.set_xlabel('lluvia del día (mm)'); ax.set_ylabel('% de los días del dominio, 1980–2014')
ax.set_title('Panel C. ¿Cuántos días caen en cada tramo de intensidad? (502 celdas × 12 784 días)', fontsize=10, loc='left')
ax.legend(fontsize=8)
(ROOT / 'figuras').mkdir(exist_ok=True)
fig.savefig(ROOT / 'figuras' / 'paso_1_3_series_e_intensidad.png', dpi=150, bbox_inches='tight')

chk.append('Dias del dominio por tramo de intensidad (%), CR2MET | ALADIN: ' + '; '.join(f'{e}: {x:.1f} | {y:.1f}' for e, x, y in zip(etq, fc, fa)))
chk.append(f'Suma de tramos: CR2MET {fc.sum():.2f} %, ALADIN {fa.sum():.2f} %')
chk.append(f'Dias >= 1 mm: CR2MET {fc[2:].sum():.2f} %, ALADIN {fa[2:].sum():.2f} %; dias con 0,1-1 mm: CR2MET {fc[1]:.2f} %, ALADIN {fa[1]:.2f} %')
(ROOT / 'salidas' / 'paso_1_3_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))
