"""Paso 4.5 — Frecuencia de dias humedos por region (1980-2014): ¿el R0 global iguala la frecuencia de CR2MET dentro de cada region?

Series comparadas (dia humedo = pr >= umbral): CR2MET (0,1 mm); ALADIN con R0 global (criterio i); ALADIN con 0,1 mm (criterio ii); ALADIN con R0 local (criterio iii, igual a CR2MET por construccion en cada celda).
Diferencia = F_ALADIN - F_CR2MET en puntos porcentuales (pp), por region y para el dominio, con IC95 del bootstrap por anios (los mismos sorteos del paso 4.2: independientes entre productos).
F de una zona = dias humedos de todas sus celdas / (dias x celdas); en cada replica F* = suma ponderada de los dias humedos por anio / suma ponderada de las celda-dias por anio.
Entradas: salidas/paso_1_1_cr2met_interpolado_dominio.nc, salidas/paso_1_2_aladin_hist_1980_2014.nc, salidas/paso_2_3_R0.json, salidas/paso_2_5_R0_local_por_celda.csv, salidas/paso_4_2_deltas_replicas.npz
Salidas: salidas/paso_4_5_frecuencia_por_region.csv, salidas/paso_4_5_chequeos.txt, figuras/paso_4_5_frecuencia_por_region.png
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
R0_loc = pd.read_csv(S / 'paso_2_5_R0_local_por_celda.csv')['R0_local_mm'].values
cr = xr.open_dataset(S / 'paso_1_1_cr2met_interpolado_dominio.nc')
al = xr.open_dataset(S / 'paso_1_2_aladin_hist_1980_2014.nc')
rep = np.load(S / 'paso_4_2_deltas_replicas.npz'); W_c, W_a = rep['W_cr2met'], rep['W_aladin']
nombres = ['Valparaíso', 'Metropolitana', "O'Higgins", 'Maule', 'Ñuble', 'Biobío', 'La Araucanía']
rid = cr['region_id'].values
t = pd.DatetimeIndex(cr['time'].values)
assert np.array_equal(t, pd.DatetimeIndex(al['time'].values))
c, a = cr['pr'].values.astype('float64'), al['pr'].values.astype('float64')
anio = t.year.values - 1980
chk = [f'{W_c.shape[0]} replicas del paso 4.2; {rid.size} celdas; R0 global = {R0} mm']

series = {'CR2MET 0,1 mm': c >= 0.1, 'ALADIN R0 global (i)': a >= R0, 'ALADIN 0,1 mm (ii)': a >= 0.1, 'ALADIN R0 local (iii)': a >= R0_loc[None, :]}
zonas = [('Dominio', np.ones(rid.size, bool))] + [(n, rid == r) for r, n in enumerate(nombres)]


def por_anio(h, k):
    """dias humedos y celda-dias por anio (35) para una zona."""
    hh = h[:, k]
    nh = np.array([hh[anio == y].sum() for y in range(35)], dtype=float)
    nt = np.array([(anio == y).sum() * k.sum() for y in range(35)], dtype=float)
    return nh, nt


filas = []
for zn, k in zonas:
    F, boot = {}, {}
    for nom, h in series.items():
        nh, nt = por_anio(h, k)
        F[nom] = 100 * nh.sum() / nt.sum()
        W = W_c if nom.startswith('CR2MET') else W_a
        boot[nom] = 100 * (W @ nh) / (W @ nt)
    f = dict(zona=zn, celdas=int(k.sum()))
    for nom in series:
        f[f'F_{nom}'] = F[nom]
    for nom in ('ALADIN R0 global (i)', 'ALADIN 0,1 mm (ii)', 'ALADIN R0 local (iii)'):
        d = F[nom] - F['CR2MET 0,1 mm']; db = boot[nom] - boot['CR2MET 0,1 mm']
        f[f'dif_{nom}'] = d; f[f'ic_inf_{nom}'] = np.percentile(db, 2.5); f[f'ic_sup_{nom}'] = np.percentile(db, 97.5)
    filas.append(f)
df = pd.DataFrame(filas)
df.to_csv(S / 'paso_4_5_frecuencia_por_region.csv', index=False)
# controles: dominio de CR2MET = 27,1856 % (paso 2.1) y de ALADIN con R0 = 27,1856 % (paso 2.3)
chk.append(f'Control con los pasos 2.1 y 2.3: dominio CR2MET {df.iloc[0]["F_CR2MET 0,1 mm"]:.4f} % (2.1: 27,1856 %); ALADIN con R0 global {df.iloc[0]["F_ALADIN R0 global (i)"]:.4f} % (2.3: 27,1856 %); ALADIN 0,1 mm {df.iloc[0]["F_ALADIN 0,1 mm (ii)"]:.4f} % (2.2: 48,9115 %)')
chk.append(f'Control criterio iii: dominio ALADIN con R0 local {df.iloc[0]["F_ALADIN R0 local (iii)"]:.4f} % (igual a CR2MET por construccion)')
chk.append('Zona | CR2MET | ALADIN R0 global | dif. [IC95] pp | ALADIN 0,1 mm | dif. pp')
for f in filas:
    chk.append(f'  {f["zona"]:<14} {f["F_CR2MET 0,1 mm"]:6.2f} | {f["F_ALADIN R0 global (i)"]:6.2f} | {f["dif_ALADIN R0 global (i)"]:+6.2f} [{f["ic_inf_ALADIN R0 global (i)"]:+6.2f}; {f["ic_sup_ALADIN R0 global (i)"]:+6.2f}] | {f["F_ALADIN 0,1 mm (ii)"]:6.2f} | {f["dif_ALADIN 0,1 mm (ii)"]:+6.2f}')
(S / 'paso_4_5_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))

fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={'width_ratios': [1.35, 1]})
ax = axs[0]
x = np.arange(len(df)); w = 0.27
ax.bar(x - w, df['F_CR2MET 0,1 mm'], w, color='#555555', label='CR2MET (0,1 mm)')
ax.bar(x, df['F_ALADIN R0 global (i)'], w, color='#3f77d1', label=f'ALADIN, R₀ global = {R0:.3f} mm (criterio i)'.replace('.', ','))
ax.bar(x + w, df['F_ALADIN 0,1 mm (ii)'], w, color='#d4564e', label='ALADIN, 0,1 mm (criterio ii)')
ax.set_xticks(x); ax.set_xticklabels([z.replace('Metropolitana', 'Metrop.') for z in df['zona']], rotation=20); ax.set_ylabel('% de días húmedos'); ax.legend(fontsize=8)
ax.set_title('Fracción de días húmedos por región, 1980–2014', fontsize=10)
ax = axs[1]
for i, f in df.iterrows():
    y = len(df) - 1 - i
    d, lo, hi = f['dif_ALADIN R0 global (i)'], f['ic_inf_ALADIN R0 global (i)'], f['ic_sup_ALADIN R0 global (i)']
    ax.barh(y, d, color='#3f77d1', height=0.6); ax.plot([lo, hi], [y, y], 'k', lw=1.4)
    ax.text(max(hi, 0) + 0.15, y, f'{d:+.2f} pp'.replace('.', ','), va='center', fontsize=8)
ax.axvline(0, color='k', lw=1); ax.set_yticks(range(len(df))); ax.set_yticklabels(list(reversed(list(df['zona']))), fontsize=9)
ax.set_xlabel('ALADIN (R₀ global) − CR2MET, puntos porcentuales'); ax.set_title('¿El R₀ global iguala la frecuencia dentro de cada región?\n(barra negra = IC95 del bootstrap por años)', fontsize=10)
ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 1.5)
fig.tight_layout()
fig.savefig(ROOT / 'figuras' / 'paso_4_5_frecuencia_por_region.png', dpi=140)
