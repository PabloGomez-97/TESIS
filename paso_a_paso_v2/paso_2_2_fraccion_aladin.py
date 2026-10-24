"""Paso 2.2 (v2) — Fraccion de dias humedos de ALADIN con 0,1 mm en el dominio (1980-2014) y su exceso frente a CR2MET (paso 2.1).

Definicion igual a la del paso 2.1: humedo = pr >= 0,1 mm/dia. Entradas: salidas/paso_1_2_aladin_hist_1980_2014.nc,
salidas/paso_1_1_cr2met_interpolado_dominio.nc, salidas/paso_2_1_objetivo_F.json.
Salidas: salidas/paso_2_2_fraccion_humeda_aladin.csv, salidas/paso_2_2_chequeos.txt, figuras/paso_2_2_fraccion_humeda_aladin.png
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
al = xr.open_dataset(ROOT / 'salidas' / 'paso_1_2_aladin_hist_1980_2014.nc')
cr = xr.open_dataset(ROOT / 'salidas' / 'paso_1_1_cr2met_interpolado_dominio.nc')
objetivo = json.load(open(ROOT / 'salidas' / 'paso_2_1_objetivo_F.json'))
nombres = list(np.load(ROOT / 'salidas' / 'paso_0_1_mascara_dominio.npz')['nombres'])
assert np.array_equal(al['time'].values.astype('datetime64[D]'), cr['time'].values.astype('datetime64[D]'))
assert np.array_equal(al['lat_celda'].values, cr['lat_celda'].values)
a = al['pr'].values.astype('float64'); c = cr['pr'].values.astype('float64')
t = pd.DatetimeIndex(al['time'].values)
rid = al['region_id'].values
lat, lon = al['lat_celda'].values, al['lon_celda'].values
ha, hc = a >= UMBRAL, c >= UMBRAL
chk = [f'ALADIN: {a.shape[0]} dias x {a.shape[1]} celdas; NaN {int(np.isnan(a).sum())}']
Fa, Fc = ha.mean() * 100, hc.mean() * 100
chk.append(f'F ALADIN (>= {UMBRAL} mm), dominio: {Fa:.4f} %; F CR2MET (paso 2.1): {Fc:.4f} % (json: {objetivo["F_cr2met_dominio_pct"]:.4f} %)')
assert abs(Fc - objetivo['F_cr2met_dominio_pct']) < 1e-9
chk.append(f'Exceso de ALADIN: {Fa - Fc:.2f} puntos porcentuales; razon ALADIN/CR2MET: {Fa / Fc:.2f}; dias humedos de ALADIN por cada 100 de CR2MET: {100 * Fa / Fc:.0f}')
chk.append(f'Celda-dias con pr exactamente {UMBRAL} mm en ALADIN: {int((a == UMBRAL).sum())}')
Fa_c, Fc_c = ha.mean(axis=0) * 100, hc.mean(axis=0) * 100
filas = []
for r, n in enumerate(nombres):
    k = rid == r
    filas.append(dict(nivel='region', nombre=n, F_cr2met_pct=hc[:, k].mean() * 100, F_aladin_pct=ha[:, k].mean() * 100,
                      exceso_pp=(ha[:, k].mean() - hc[:, k].mean()) * 100))
filas.append(dict(nivel='dominio', nombre='Dominio (unión)', F_cr2met_pct=Fc, F_aladin_pct=Fa, exceso_pp=Fa - Fc))
mes_a = pd.Series(ha.mean(axis=1) * 100, index=t).groupby(t.month).mean()
mes_c = pd.Series(hc.mean(axis=1) * 100, index=t).groupby(t.month).mean()
for k in mes_a.index:
    filas.append(dict(nivel='mes', nombre=f'mes {k:02d}', F_cr2met_pct=mes_c[k], F_aladin_pct=mes_a[k], exceso_pp=mes_a[k] - mes_c[k]))
df = pd.DataFrame(filas)
df.to_csv(ROOT / 'salidas' / 'paso_2_2_fraccion_humeda_aladin.csv', index=False)

chk.append('Por region (CR2MET | ALADIN | exceso, %/pp): ' + '; '.join(f'{f["nombre"]} {f["F_cr2met_pct"]:.2f} | {f["F_aladin_pct"]:.2f} | {f["exceso_pp"]:+.2f}' for f in filas[:len(nombres)]))
chk.append(f'Por celda, ALADIN: min {Fa_c.min():.2f} %, mediana {np.median(Fa_c):.2f} %, max {Fa_c.max():.2f} %; exceso por celda: min {(Fa_c - Fc_c).min():+.2f}, mediana {np.median(Fa_c - Fc_c):+.2f}, max {(Fa_c - Fc_c).max():+.2f} pp; celdas con ALADIN > CR2MET: {int((Fa_c > Fc_c).sum())} de {Fa_c.size}')
chk.append('Por mes, exceso (pp): ' + '; '.join(f'{k:02d}: {mes_a[k] - mes_c[k]:+.1f}' for k in mes_a.index))
chk.append(f'Correlacion entre celdas del patron espacial de F (ALADIN frente a CR2MET): {np.corrcoef(Fa_c, Fc_c)[0, 1]:.3f}')
# cuanto del exceso es "llovizna" (0,1-1 mm) y cuanto es lluvia >= 1 mm: dias con 0,1 <= pr < 1 en ALADIN
chk.append(f'Dias con 0,1 <= pr < 1 mm: ALADIN {np.mean((a >= 0.1) & (a < 1)) * 100:.2f} %, CR2MET {np.mean((c >= 0.1) & (c < 1)) * 100:.2f} % (con el umbral de 0,1 mm SI cuentan como humedos)')
(ROOT / 'salidas' / 'paso_2_2_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))

fig, axs = plt.subplots(1, 4, figsize=(20, 6.2), gridspec_kw={'width_ratios': [1, 1, 0.9, 1]})
vmax = max(Fa_c.max(), Fc_c.max())
for ax, v, tit in ((axs[0], Fc_c, 'CR2MET'), (axs[1], Fa_c, 'ALADIN')):
    sc = ax.scatter(lon, lat, c=v, s=22, marker='s', cmap='viridis', vmin=0, vmax=vmax)
    ax.set_aspect(1 / np.cos(np.deg2rad(34))); ax.set_title(f'{tit}: % de días ≥ 0,1 mm por celda', fontsize=10); ax.set_xlabel('longitud')
    fig.colorbar(sc, ax=ax, shrink=0.7, location='bottom', pad=0.12)
axs[0].set_ylabel('latitud')
ax = axs[2]
dif = Fa_c - Fc_c
sc = ax.scatter(lon, lat, c=dif, s=22, marker='s', cmap='Reds', vmin=0, vmax=dif.max())
ax.set_aspect(1 / np.cos(np.deg2rad(34))); ax.set_title('Exceso de ALADIN (puntos porcentuales)', fontsize=10); ax.set_xlabel('longitud')
fig.colorbar(sc, ax=ax, shrink=0.7, location='bottom', pad=0.12)
ax = axs[3]
x = np.arange(len(nombres) + 1); w = 0.38
vc = [f['F_cr2met_pct'] for f in filas[:len(nombres) + 1]]; va = [f['F_aladin_pct'] for f in filas[:len(nombres) + 1]]
ax.bar(x - w / 2, vc, w, color='#555555', label='CR2MET'); ax.bar(x + w / 2, va, w, color='#d4564e', label='ALADIN')
for i in range(len(nombres) + 1):
    ax.text(x[i] - w / 2, vc[i] + 0.5, f'{vc[i]:.1f}'.replace('.', ','), ha='center', fontsize=8)
    ax.text(x[i] + w / 2, va[i] + 0.5, f'{va[i]:.1f}'.replace('.', ','), ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([n.replace('Metropolitana', 'Metrop.').replace('La Araucanía', 'Araucanía') for n in nombres] + ['Dominio'], fontsize=8, rotation=30)
ax.set_ylabel('% de días ≥ 0,1 mm'); ax.set_title('Por región, 0,1 mm en ambos', fontsize=10); ax.legend(fontsize=8)
fig.tight_layout()
(ROOT / 'figuras').mkdir(exist_ok=True)
fig.savefig(ROOT / 'figuras' / 'paso_2_2_fraccion_humeda_aladin.png', dpi=140, bbox_inches='tight')
