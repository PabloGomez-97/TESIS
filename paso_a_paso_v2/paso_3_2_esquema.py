"""Paso 3.2 — Figura de la definicion operativa de racha seca con DATOS REALES de una celda (cercana a Santiago, celda 845).

Panel A: 1-ene a 15-abr de 1980, CR2MET (umbral 0,1 mm) y ALADIN (umbral R0): dias secos (gris) y humedos (azul), rachas y su duracion; la primera toca el primer dia del periodo
         (censurada, se excluye).
Panel B: una racha real de CR2MET que cruza el cambio de anio (empieza en diciembre y termina en enero): su anio es el de inicio.
Entradas: salidas/paso_1_1_cr2met_interpolado_dominio.nc, salidas/paso_1_2_aladin_hist_1980_2014.nc, salidas/paso_2_3_R0.json, salidas/paso_3_1_rachas_{completas,censuradas}_{cr2met_hist,aladin_hist_i}.npz
Salidas: figuras/paso_3_2_esquema_rachas.png, salidas/paso_3_2_ejemplos.txt
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
S = ROOT / 'salidas'
R0 = json.load(open(S / 'paso_2_3_R0.json'))['R0_usado_mm']
cr = xr.open_dataset(S / 'paso_1_1_cr2met_interpolado_dominio.nc')
al = xr.open_dataset(S / 'paso_1_2_aladin_hist_1980_2014.nc')
t = pd.DatetimeIndex(cr['time'].values)
j = int(np.argmin((cr['lat_celda'].values + 33.45) ** 2 + (cr['lon_celda'].values + 70.67) ** 2))
pc, pa = cr['pr'].values[:, j].astype('float64'), al['pr'].values[:, j].astype('float64')


def spells(nombre):
    out = []
    for tag, cen in (('completas', False), ('censuradas', True)):
        z = np.load(S / f'paso_3_1_rachas_{tag}_{nombre}.npz')
        m = z['celda'] == j
        out += [(int(a), int(b), cen) for a, b in zip(z['inicio'][m], z['duracion'][m])]
    return sorted(out)


sp_c, sp_a = spells('cr2met_hist'), spells('aladin_hist_i')
txt = [f'Celda {j} (lat {cr["lat_celda"].values[j]:.2f}, lon {cr["lon_celda"].values[j]:.2f}); R0 = {R0} mm']


def dibuja(ax, serie, umbral, sp, i0, i1, color_h, titulo, etiqueta):
    idx = np.arange(i0, i1)
    x = t[idx]
    seco = serie[idx] < umbral
    ax.bar(x, np.where(seco, 0, serie[idx]), width=0.9, color=color_h)
    ax.axhline(umbral, color='k', lw=0.8, ls=':')
    ax.set_ylabel('mm/día'); ax.set_title(titulo, fontsize=10, loc='left')
    ymax = max(serie[idx].max(), umbral) * 1.0
    for a, b, cen in sp:
        if a + b <= i0 or a >= i1: continue
        a2, b2 = max(a, i0), min(a + b, i1)
        ax.axvspan(t[a2] - pd.Timedelta(hours=12), t[b2 - 1] + pd.Timedelta(hours=12), ymin=0, ymax=0.09, color='#d4564e' if cen else '#888888', alpha=0.85)
        ax.text(t[a2] + (t[b2 - 1] - t[a2]) / 2, ymax * 0.12, f'{b} d' + ('\n(censurada)' if cen else ''), ha='center', va='bottom', fontsize=8,
                color='#a33' if cen else '#333')
    ax.set_xlim(x[0] - pd.Timedelta(days=1), x[-1] + pd.Timedelta(days=1)); ax.set_ylim(0, ymax * 1.15)
    ax.text(0.995, 0.93, etiqueta, transform=ax.transAxes, ha='right', va='top', fontsize=8)


fig, axs = plt.subplots(3, 1, figsize=(12, 10.5), gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.5})
i0, i1 = 0, int(np.where(t == pd.Timestamp('1980-04-15'))[0][0]) + 1
dibuja(axs[0], pc, 0.1, sp_c, i0, i1, '#3f77d1', 'A1. CR2MET, 1-ene a 15-abr de 1980: día seco si llovió menos de 0,1 mm', 'barras azules = días húmedos (≥ 0,1 mm); sin barra = día seco')
dibuja(axs[1], pa, R0, sp_a, i0, i1, '#3f77d1', f'A2. ALADIN (criterio i), misma celda y fechas: día seco si llovió menos de R₀ = {R0} mm'.replace('.', ','), 'barras azules = días húmedos (≥ R₀); los días con menos lluvia que R₀ cuentan como secos y no se dibujan')
for nom, sp in (('CR2MET', sp_c), ('ALADIN i', sp_a)):
    txt.append(f'{nom}, rachas que empiezan entre el 1-ene y el 15-abr de 1980 (inicio, duracion, censurada): ' + str([(str(t[a].date()), b, c) for a, b, c in sp if a < i1]))
# Panel B: racha real de CR2MET que cruza un cambio de anio (empieza en dic, termina en ene), la mas larga de la celda con inicio entre 1-dic y 31-dic
cand = [(a, b) for a, b, c in sp_c if not c and t[a].month == 12 and (t[min(a + b, len(t) - 1)].year != t[a].year) and b >= 12]
a, b = max(cand, key=lambda z: z[1])
dibuja(axs[2], pc, 0.1, sp_c, a - 15, a + b + 15, '#3f77d1', f'B. CR2MET, una racha real que cruza el cambio de año: empieza el {t[a].date()} y dura {b} días', 'su año es el del día de inicio')
axs[2].axvline(pd.Timestamp(f'{t[a].year + 1}-01-01') - pd.Timedelta(hours=12), color='#c0392b', lw=1.2)
axs[2].text(pd.Timestamp(f'{t[a].year + 1}-01-01'), axs[2].get_ylim()[1] * 0.97, f' 1-ene {t[a].year + 1}', color='#c0392b', fontsize=8, va='top')
axs[2].text(0.01, 0.93, f'se cuenta como racha de {t[a].year}', transform=axs[2].transAxes, fontsize=9, va='top', color='#a33')
txt.append(f'Panel B: racha de CR2MET que empieza el {t[a].date()} y dura {b} d (termina el {t[a + b - 1].date()}): anio asignado {t[a].year}')
fig.text(0.01, 0.005, 'Barras grises/rojas bajo cada panel: rachas secas con su duración. Gris = completa (se usa). Rojo = censurada (toca el primer o el último día del período: se excluye). Datos reales de una celda cercana a Santiago.', fontsize=8.5, color='#444')
(ROOT / 'figuras').mkdir(exist_ok=True)
fig.savefig(ROOT / 'figuras' / 'paso_3_2_esquema_rachas.png', dpi=150, bbox_inches='tight')
(S / 'paso_3_2_ejemplos.txt').write_text('\n'.join(txt), encoding='utf-8')
print('\n'.join(txt))
