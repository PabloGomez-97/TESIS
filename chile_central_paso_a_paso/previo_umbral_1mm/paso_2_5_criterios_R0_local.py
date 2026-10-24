"""Paso 2.5 — Los tres criterios de umbral y el R0 local por celda.

  i   : CR2MET 1 mm, ALADIN R0 global (paso 2.3: un solo umbral para todo el dominio)
  ii  : 1 mm en ambos (sin calibrar)
  iii : CR2MET 1 mm, ALADIN R0 local: en cada celda, el umbral de ALADIN que deja a esa celda con la misma fraccion de dias humedos que CR2MET (1 mm) en esa celda

R0 local por biseccion, celda por celda (vectorizada): se busca el umbral cuyo conteo de dias humedos es k_c = (dias >= 1 mm de CR2MET en la celda).
Control: el k-esimo valor mayor de ALADIN en cada celda.
Entradas: salidas/paso_1_1_cr2met_interpolado_dominio.nc, salidas/paso_1_2_aladin_hist_1980_2014.nc, salidas/paso_2_3_R0.json
Salidas: salidas/paso_2_5_criterios.csv, salidas/paso_2_5_R0_local_por_celda.csv, salidas/paso_2_5_chequeos.txt, figuras/paso_2_5_R0_local.png
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
al = xr.open_dataset(ROOT / 'salidas' / 'paso_1_2_aladin_hist_1980_2014.nc')
cr = xr.open_dataset(ROOT / 'salidas' / 'paso_1_1_cr2met_interpolado_dominio.nc')
R0 = json.load(open(ROOT / 'salidas' / 'paso_2_3_R0.json'))['R0_usado_mm']
nombres = list(np.load(ROOT / 'salidas' / 'paso_0_1_mascara_dominio.npz')['nombres'])
a = al['pr'].values.astype('float64'); c = cr['pr'].values.astype('float64')
rid, lat, lon = al['region_id'].values, al['lat_celda'].values, al['lon_celda'].values
N, nc = a.shape
chk = []

k = (c >= 1.0).sum(axis=0)                      # dias humedos de CR2MET por celda (el objetivo de cada celda, en conteo)
chk.append(f'Objetivo por celda: dias >= 1 mm de CR2MET, min {k.min()}, mediana {int(np.median(k))}, max {k.max()} (de {N}); k > 0 en todas: {bool(np.all(k > 0))}')

# biseccion vectorizada: busca, en cada celda, R0 tal que #(a >= R0) = k
lo, hi = np.zeros(nc), np.full(nc, 100.0)
for it in range(1, 80):
    mid = 0.5 * (lo + hi)
    cnt = (a >= mid[None, :]).sum(axis=0)
    # demasiados dias humedos (cnt > k): subir el umbral; si no, bajarlo
    lo = np.where(cnt > k, mid, lo)
    hi = np.where(cnt > k, hi, mid)
    if np.all(cnt == k):
        break
R0_loc = hi    # extremo con conteo <= k; al converger el conteo es exactamente k (0.5*(lo+hi) cae sobre el valor de empate y suma un dia)
cnt = (a >= R0_loc[None, :]).sum(axis=0)
chk.append(f'Biseccion por celda: {it} iteraciones; celdas con conteo exacto (= objetivo): {int((cnt == k).sum())} de {nc}')
# control exacto: cualquier umbral en (v_{k+1}, v_k] deja exactamente k dias humedos; v_k = k-esimo valor mayor de ALADIN en la celda
vk = np.array([np.partition(a[:, j], N - k[j])[N - k[j]] for j in range(nc)])
vk1 = np.array([np.partition(a[:, j], N - k[j] - 1)[N - k[j] - 1] for j in range(nc)])
chk.append(f'Control por cuantil: R0 local dentro de (v_(k+1), v_k] en {int(((R0_loc > vk1) & (R0_loc <= vk)).sum())} de {nc} celdas; diferencia maxima con v_k: {np.abs(R0_loc - vk).max():.3f} mm (es el ancho del intervalo en que el conteo no cambia)')
Fc_cell = k / N * 100
Fa_loc = cnt / N * 100
chk.append(f'F por celda con R0 local: max |F_ALADIN - F_CR2MET| = {np.abs(Fa_loc - Fc_cell).max():.4f} pp; dominio: ALADIN {cnt.sum() / a.size * 100:.4f} % frente a CR2MET {k.sum() / c.size * 100:.4f} %')
q = lambda x, p: np.percentile(x, p)
chk.append(f'R0 local (mm): min {R0_loc.min():.2f}, p10 {q(R0_loc, 10):.2f}, mediana {np.median(R0_loc):.2f}, p90 {q(R0_loc, 90):.2f}, max {R0_loc.max():.2f}; R0 global {R0}')
chk.append(f'Celdas con R0 local menor que el global: {int((R0_loc < R0).sum())}; mayor: {int((R0_loc > R0).sum())}; con R0 local < 1 mm: {int((R0_loc < 1).sum())}')
for r, n in enumerate(nombres):
    m = rid == r
    chk.append(f'  {n:<14} R0 local: mediana {np.median(R0_loc[m]):.2f} mm (p10 {q(R0_loc[m], 10):.2f}, p90 {q(R0_loc[m], 90):.2f}); celdas con local > global: {int((R0_loc[m] > R0).sum())} de {int(m.sum())}')
# cuanto se desvia cada celda si se usa el R0 global (anticipo de lo que verá el paso 4.5)
Fa_glob = (a >= R0).mean(axis=0) * 100
chk.append(f'Con R0 global, F_ALADIN - F_CR2MET por celda (pp): min {(Fa_glob - Fc_cell).min():+.2f}, mediana {np.median(Fa_glob - Fc_cell):+.2f}, max {(Fa_glob - Fc_cell).max():+.2f}; celdas con exceso > +5 pp: {int(((Fa_glob - Fc_cell) > 5).sum())}; con defecto < -5 pp: {int(((Fa_glob - Fc_cell) < -5).sum())}')
chk.append(f'Correlacion entre R0 local y lluvia anual de CR2MET por celda: {np.corrcoef(R0_loc, c.mean(axis=0) * 365.25)[0, 1]:.3f}; con la fraccion humeda de CR2MET: {np.corrcoef(R0_loc, Fc_cell)[0, 1]:.3f}')

pd.DataFrame({'celda': np.arange(nc), 'region': [nombres[i] for i in rid], 'lat': lat, 'lon': lon, 'dias_humedos_cr2met_1mm': k,
              'F_cr2met_pct': Fc_cell, 'R0_local_mm': R0_loc, 'F_aladin_R0_local_pct': Fa_loc, 'F_aladin_R0_global_pct': Fa_glob}).to_csv(
    ROOT / 'salidas' / 'paso_2_5_R0_local_por_celda.csv', index=False)
pd.DataFrame([
    dict(criterio='i', umbral_cr2met_mm='1', umbral_aladin_mm=f'{R0} (global, un valor para todo el dominio)', uso='análisis principal'),
    dict(criterio='ii', umbral_cr2met_mm='1', umbral_aladin_mm='1', uso='sensibilidad: sesgo sin calibrar'),
    dict(criterio='iii', umbral_cr2met_mm='1', umbral_aladin_mm=f'local por celda (mediana {np.median(R0_loc):.2f}, de {R0_loc.min():.2f} a {R0_loc.max():.2f})', uso='sensibilidad espacial'),
]).to_csv(ROOT / 'salidas' / 'paso_2_5_criterios.csv', index=False)
(ROOT / 'salidas' / 'paso_2_5_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))

fig, axs = plt.subplots(1, 3, figsize=(16, 6.2), gridspec_kw={'width_ratios': [1.1, 1, 1]})
ax = axs[0]
sc = ax.scatter(lon, lat, c=R0_loc, s=28, marker='s', cmap='YlOrRd')
ax.set_aspect(1 / np.cos(np.deg2rad(34))); ax.set_xlabel('longitud'); ax.set_ylabel('latitud')
ax.set_title('R₀ local de ALADIN por celda (mm/día)', fontsize=10)
fig.colorbar(sc, ax=ax, shrink=0.7, location='bottom', pad=0.12).ax.axvline(R0, color='k', lw=2)
ax = axs[1]
ax.hist(R0_loc, bins=30, color='#d98c4b', edgecolor='white')
ax.axvline(R0, color='#3f77d1', lw=2, label=f'R₀ global = {R0:.3f} mm'.replace('.', ','))
ax.axvline(np.median(R0_loc), color='k', ls='--', label=f'mediana local = {np.median(R0_loc):.2f} mm'.replace('.', ','))
ax.set_xlabel('R₀ (mm/día)'); ax.set_ylabel('número de celdas'); ax.legend(fontsize=9); ax.set_title('Distribución de R₀ local (502 celdas)', fontsize=10)
ax = axs[2]
cols = ['#7B68B5', '#2C7FB8', '#2AA198', '#253494']
bp = ax.boxplot([R0_loc[rid == r] for r in range(4)], positions=range(4), widths=0.55, patch_artist=True, showfliers=True)
for p, cc in zip(bp['boxes'], cols):
    p.set_facecolor(cc); p.set_alpha(0.7)
ax.axhline(R0, color='#3f77d1', lw=2)
ax.set_xticks(range(4)); ax.set_xticklabels([n.replace('Metropolitana', 'Metrop.') for n in nombres]); ax.set_ylabel('R₀ local (mm/día)')
ax.set_title('Por región (línea azul: R₀ global)', fontsize=10)
fig.tight_layout()
(ROOT / 'figuras').mkdir(exist_ok=True)
fig.savefig(ROOT / 'figuras' / 'paso_2_5_R0_local.png', dpi=140, bbox_inches='tight')
