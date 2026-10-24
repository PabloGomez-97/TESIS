"""Paso 2.3 — Calibrar R0 por biseccion: el umbral de ALADIN que da la misma fraccion de dias humedos que CR2MET con 1 mm.

Objetivo: F_CR2MET del dominio (paso 2.1) = salidas/paso_2_1_objetivo_F.json (502 celdas, 1980-2014, pr >= 1 mm).
Funcion a igualar: F_ALADIN(R0) = fraccion de celda-dias de ALADIN historico (502 x 12 784) con pr >= R0. Es decreciente en R0.
Biseccion sobre R0 en [0, 50] mm hasta |F(R0) - objetivo| <= 1e-6 (fraccion). R0 usado = R0 de la biseccion redondeado a 3 decimales
(misma convencion que el analisis anterior). Control independiente: el cuantil exacto de la distribucion de lluvia de ALADIN.
Salidas: salidas/paso_2_3_R0.json, salidas/paso_2_3_biseccion.csv (historial de la biseccion), salidas/paso_2_3_chequeos.txt
"""
from pathlib import Path
import json
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent
obj = json.load(open(ROOT / 'salidas' / 'paso_2_1_objetivo_F.json'))
F_obj = obj['F_cr2met_dominio_pct'] / 100.0
a = xr.open_dataset(ROOT / 'salidas' / 'paso_1_2_aladin_hist_1980_2014.nc')['pr'].values.ravel().astype('float64')
N = a.size
chk = [f'Objetivo F_CR2MET (dominio, 1 mm): {F_obj * 100:.6f} %; celda-dias de ALADIN: {N}; NaN: {int(np.isnan(a).sum())}']


def F(r0):
    return float(np.mean(a >= r0))


chk.append(f'F_ALADIN(0) = {F(0.0) * 100:.3f} % ; F_ALADIN(1 mm) = {F(1.0) * 100:.4f} % ; F_ALADIN(50 mm) = {F(50.0) * 100:.5f} % (el objetivo cae entre F(50) y F(0): se puede biseccionar)')
assert F(50.0) < F_obj < F(0.0)

lo, hi = 0.0, 50.0
hist = []
TOL = 1e-6
for it in range(1, 101):
    mid = 0.5 * (lo + hi)
    f = F(mid)
    hist.append((it, lo, hi, mid, f * 100, (f - F_obj) * 100))
    if abs(f - F_obj) <= TOL:
        break
    if f > F_obj:
        lo = mid      # demasiados dias humedos: hay que subir el umbral
    else:
        hi = mid
R0_bis = mid
R0_usado = round(R0_bis, 3)
chk.append(f'Biseccion: {it} iteraciones; R0 = {R0_bis:.6f} mm; F(R0) = {F(R0_bis) * 100:.6f} % (diferencia con el objetivo {(F(R0_bis) - F_obj) * 100:+.2e} pp)')
chk.append(f'R0 usado (redondeado a 3 decimales) = {R0_usado} mm; F(R0 usado) = {F(R0_usado) * 100:.6f} % (diferencia {(F(R0_usado) - F_obj) * 100:+.2e} pp)')

# control independiente: el R0 exacto es el k-esimo valor mayor, con k = round(F_obj * N)
k = int(round(F_obj * N))
R0_q = float(np.partition(a, N - k)[N - k])
chk.append(f'Control por cuantil (k = {k} celda-dias mas lluviosos): R0 = {R0_q:.6f} mm; F(R0) = {F(R0_q) * 100:.6f} %; diferencia con la biseccion: {abs(R0_q - R0_bis):.2e} mm')
# F es una funcion escalonada: cuanto cambia F si R0 se mueve +-0,01 mm / +-0,1 mm
for d in (0.01, 0.1, 0.5):
    chk.append(f'Sensibilidad: F con R0 {R0_usado - d:.3f} = {F(R0_usado - d) * 100:.3f} %; con R0 {R0_usado + d:.3f} = {F(R0_usado + d) * 100:.3f} %')
# referencia del analisis anterior (solo informativa)
chk.append(f'Referencia interna: con R0 = 5,285 mm (calibracion del analisis anterior), F_ALADIN en este dominio = {F(5.285) * 100:.3f} % (objetivo {F_obj * 100:.3f} %)')
chk.append(f'Mediana de los dias humedos de ALADIN con R0: {np.median(a[a >= R0_usado]):.2f} mm/dia; con 1 mm: {np.median(a[a >= 1.0]):.2f} mm/dia; CR2MET con 1 mm: ver paso 1.3')

json.dump(dict(periodo='1980-2014', celdas=502, F_objetivo_cr2met_pct=F_obj * 100, R0_biseccion_mm=R0_bis, R0_usado_mm=R0_usado,
               F_aladin_R0_usado_pct=F(R0_usado) * 100, F_aladin_1mm_pct=F(1.0) * 100, iteraciones=it, tolerancia_fraccion=TOL,
               definicion='dia humedo de ALADIN: pr >= R0; dia seco: pr < R0'),
          open(ROOT / 'salidas' / 'paso_2_3_R0.json', 'w'), indent=2, ensure_ascii=False)
with open(ROOT / 'salidas' / 'paso_2_3_biseccion.csv', 'w') as fh:
    fh.write('iteracion,R0_inf_mm,R0_sup_mm,R0_medio_mm,F_aladin_pct,F_menos_objetivo_pp\n')
    for h in hist:
        fh.write(','.join(f'{x:.8g}' for x in h) + '\n')
(ROOT / 'salidas' / 'paso_2_3_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))
print('primeras iteraciones:')
for h in hist[:6]:
    print(f'  it {h[0]:>2}: R0 = {h[3]:.4f} mm -> F = {h[4]:.3f} %')
