"""Paso 2.4 (v2) — Figura de calibracion: F(R) = fraccion de dias humedos en funcion del umbral R, para ALADIN y CR2MET (dominio de 972 celdas, 1980-2014),
con el objetivo (CR2MET con 0,1 mm), R0 de la biseccion y los tres casos comparados.
Entradas: salidas/paso_1_1_cr2met_interpolado_dominio.nc, salidas/paso_1_2_aladin_hist_1980_2014.nc, salidas/paso_2_1_objetivo_F.json, salidas/paso_2_3_R0.json
Salidas: salidas/paso_2_4_curvas_F_umbral.csv, figuras/paso_2_4_calibracion_R0.png
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
a = xr.open_dataset(ROOT / 'salidas' / 'paso_1_2_aladin_hist_1980_2014.nc')['pr'].values.ravel().astype('float64')
c = xr.open_dataset(ROOT / 'salidas' / 'paso_1_1_cr2met_interpolado_dominio.nc')['pr'].values.ravel().astype('float64')
obj = json.load(open(ROOT / 'salidas' / 'paso_2_1_objetivo_F.json'))
r0 = json.load(open(ROOT / 'salidas' / 'paso_2_3_R0.json'))
F_obj, R0 = obj['F_cr2met_dominio_pct'], r0['R0_usado_mm']
umbrales = np.r_[np.arange(0.1, 2, 0.1), np.arange(2, 30.01, 0.25)]
Fa = np.array([np.mean(a >= u) * 100 for u in umbrales])
Fc = np.array([np.mean(c >= u) * 100 for u in umbrales])
pd.DataFrame({'umbral_mm': umbrales, 'F_aladin_pct': Fa, 'F_cr2met_pct': Fc}).to_csv(ROOT / 'salidas' / 'paso_2_4_curvas_F_umbral.csv', index=False)
assert np.all(np.diff(Fa) <= 0) and np.all(np.diff(Fc) <= 0)   # F decrece con el umbral

fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.5, 1]})
ax = axs[0]
ax.plot(umbrales, Fa, color='#d4564e', lw=2, label='ALADIN')
ax.plot(umbrales, Fc, color='#555555', lw=2, label='CR2MET (interpolado)')
ax.axhline(F_obj, color='#555555', ls=':', lw=1)
ax.axvline(R0, color='#3f77d1', ls='--', lw=1.2)
ax.plot([0.1], [obj['F_cr2met_dominio_pct']], 'o', color='#555555', ms=8, zorder=5)
ax.plot([0.1], [np.mean(a >= 0.1) * 100], 'o', color='#d4564e', ms=8, zorder=5)
ax.plot([R0], [np.mean(a >= R0) * 100], 'o', color='#3f77d1', ms=9, zorder=5)
ax.annotate(f'CR2MET, 0,1 mm: {F_obj:.2f} %'.replace('.', ','), (0.1, F_obj), xytext=(8, -16), textcoords='offset points', fontsize=9, color='#333')
ax.annotate(f'ALADIN, 0,1 mm: {np.mean(a >= 0.1) * 100:.2f} %'.replace('.', ','), (0.1, np.mean(a >= 0.1) * 100), xytext=(10, 4), textcoords='offset points', fontsize=9, color='#a33')
ax.annotate(f'R₀ = {R0:.3f} mm\nALADIN: {np.mean(a >= R0) * 100:.2f} %'.replace('.', ','), (R0, np.mean(a >= R0) * 100), xytext=(12, 40), textcoords='offset points', fontsize=9, color='#2a55a0',
            arrowprops=dict(arrowstyle='-', color='#3f77d1'))
ax.set_xlim(0, 25); ax.set_ylim(0, 65)
ax.set_xlabel('umbral de día húmedo (mm/día)'); ax.set_ylabel('fracción de días húmedos (%)')
ax.set_title('Calibración de R₀: se busca el umbral donde ALADIN iguala el objetivo (línea punteada)', fontsize=10, loc='left')
ax.legend(loc='upper right'); ax.grid(alpha=0.25)
ax = axs[1]
vals = [F_obj, np.mean(a >= 0.1) * 100, np.mean(a >= R0) * 100]
labs = ['CR2MET\n0,1 mm', 'ALADIN\n0,1 mm', f'ALADIN\nR₀ = {R0:.3f} mm'.replace('.', ',')]
ax.bar(range(3), vals, color=['#8a8984', '#d4564e', '#3f77d1'], width=0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 0.6, f'{v:.2f}'.replace('.', ','), ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_ylabel('fracción de días húmedos (%)'); ax.set_ylim(0, 58)
ax.set_title('Resultado de la calibración (972 celdas, 1980–2014)', fontsize=10, loc='left')
fig.tight_layout()
(ROOT / 'figuras').mkdir(exist_ok=True)
fig.savefig(ROOT / 'figuras' / 'paso_2_4_calibracion_R0.png', dpi=160)
print('ok', f'F_obj={F_obj:.4f}', f'R0={R0}', f'F_aladin(R0)={vals[2]:.4f}', f'F_aladin(0,1mm)={vals[1]:.4f}')
print('ALADIN por encima de CR2MET en todos los umbrales dibujados:', bool(np.all(Fa > Fc)))
print('F con el umbral de 0,1 mm: CR2MET %.2f %%, ALADIN %.2f %%; F en 25 mm: CR2MET %.2f %%, ALADIN %.2f %%' % (Fc[0], Fa[0], Fc[np.argmin(abs(umbrales-25))], Fa[np.argmin(abs(umbrales-25))]))
