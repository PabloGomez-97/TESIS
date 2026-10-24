"""Paso 3.1 (reformulado) — Extraer las rachas secas COMPLETAS por celda (972 celdas) para cada serie y criterio de umbral, y cuantificar las censuradas excluidas.

Dia seco: pr < umbral (CR2MET 0,1 mm; ALADIN R0). Racha seca: secuencia maxima de dias secos consecutivos de una celda. Anio de la racha = anio del dia
en que EMPIEZA. Una racha es CENSURADA si toca el primer o el ultimo dia del periodo (no se sabe cuando empezo o cuando termina): se EXCLUYE. Los archivos
paso_3_1_rachas_completas_<serie>.npz son los datos que usan todos los pasos siguientes; las censuradas se guardan aparte y se cuantifican (cuantas son, cuanto duran,
cuantas superan el p99 de las completas).

Series y umbrales:
  cr2met_hist  : CR2MET interpolado 1980-2014, umbral 0,1 mm
  aladin_hist_i   : ALADIN hist., R0 global (paso 2.3)           aladin_hist_ii : ALADIN hist., 0,1 mm            aladin_hist_iii : ALADIN hist., R0 local por celda (paso 2.5)
  aladin_ssp585_i : ALADIN SSP5-8.5 2040-2074, R0 global           aladin_ssp585_ii : ALADIN SSP5-8.5, 0,1 mm
  aladin_ssp370_i : ALADIN SSP3-7.0 2040-2074, R0 global
Entradas: salidas/paso_1_1_..., paso_1_2_..., paso_2_3_R0.json, paso_2_5_R0_local_por_celda.csv
Salidas: salidas/paso_3_1_rachas_completas_<serie>.npz, salidas/paso_3_1_rachas_censuradas_<serie>.npz (celda, inicio [indice de dia], duracion, anio_inicio, fechas), salidas/paso_3_1_resumen.csv, salidas/paso_3_1_censuradas_por_region.csv, salidas/paso_3_1_chequeos.txt
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
S = ROOT / 'salidas'
R0 = json.load(open(S / 'paso_2_3_R0.json'))['R0_usado_mm']
R0_loc = pd.read_csv(S / 'paso_2_5_R0_local_por_celda.csv')['R0_local_mm'].values
nombres = list(np.load(S / 'paso_0_1_mascara_dominio.npz')['nombres'])
cr = xr.open_dataset(S / 'paso_1_1_cr2met_interpolado_dominio.nc')
rid = cr['region_id'].values
nc = rid.size
chk = [f'R0 global = {R0} mm; R0 local: {R0_loc.size} valores (min {R0_loc.min():.2f}, max {R0_loc.max():.2f}); celdas: {nc}']


def rachas(seco):
    """seco: bool [dias, celdas]. Devuelve (celda, inicio, duracion, censurada) de todas las rachas secas."""
    nd = seco.shape[0]
    cel, ini, dur, cen = [], [], [], []
    for j in range(seco.shape[1]):
        x = np.r_[False, seco[:, j], False].astype(np.int8)
        d = np.diff(x)
        s, e = np.where(d == 1)[0], np.where(d == -1)[0]     # inicio (indice del primer dia seco) y fin (indice siguiente al ultimo)
        cel.append(np.full(s.size, j, dtype=np.int16)); ini.append(s.astype(np.int32)); dur.append((e - s).astype(np.int16))
        cen.append((s == 0) | (e == nd))
    return np.concatenate(cel), np.concatenate(ini), np.concatenate(dur), np.concatenate(cen)


reg_filas = []


def procesar(nombre, pr, tiempo, umbral):
    seco = pr < (umbral[None, :] if np.ndim(umbral) else umbral)
    cel, ini, dur, cen = rachas(seco)
    anio = pd.DatetimeIndex(tiempo)[ini].year.values.astype(np.int16)
    fechas = np.asarray(tiempo, dtype='datetime64[D]')
    for tag, m in (('completas', ~cen), ('censuradas', cen)):
        np.savez_compressed(S / f'paso_3_1_rachas_{tag}_{nombre}.npz', celda=cel[m], inicio=ini[m], duracion=dur[m], anio_inicio=anio[m], fechas=fechas)
    # chequeos: (1) las duraciones de TODAS las rachas suman exactamente los dias secos; (2) completas + censuradas = todas; (3) scipy.ndimage.label en celdas al azar
    ok = int(dur.sum()) == int(seco.sum())
    rng = np.random.default_rng(3)
    ok2 = True
    for j in rng.choice(nc, 8, replace=False):
        lab, n = ndimage.label(seco[:, j])
        sizes = np.bincount(lab.ravel())[1:] if n else np.array([], int)
        mine = np.sort(dur[cel == j]); ok2 &= (n == mine.size) and np.array_equal(np.sort(sizes), mine)
    dc, dcen = dur[~cen], dur[cen]
    p99 = float(np.percentile(dc, 99))
    fila = dict(serie=nombre, umbral=('R0 local por celda' if np.ndim(umbral) else f'{umbral:.3f} mm'), dias_secos_pct=100 * seco.mean(),
                rachas_totales=int(dur.size), completas=int((~cen).sum()), censuradas=int(cen.sum()), censuradas_pct=100 * cen.mean(),
                media_completas=float(dc.mean()), mediana_completas=float(np.median(dc)), p99_completas=p99, max_completas=int(dc.max()),
                completas_ge20=int((dc >= 20).sum()), censuradas_ge20=int((dcen >= 20).sum()),
                censuradas_mayores_p99=int((dcen > p99).sum()), media_censuradas=float(dcen.mean()), max_censuradas=int(dcen.max()),
                suma_duraciones_igual_dias_secos=ok, coincide_con_scipy_label=bool(ok2))
    for r, nom in enumerate(nombres):
        mc = rid[cel] == r
        pr99 = float(np.percentile(dur[mc & ~cen], 99))
        reg_filas.append(dict(serie=nombre, region=nom, completas=int((mc & ~cen).sum()), censuradas=int((mc & cen).sum()), p99_completas_region=pr99,
                              censuradas_mayores_p99_region=int((dur[mc & cen] > pr99).sum()), censuradas_ge20=int((dur[mc & cen] >= 20).sum())))
    chk.append(f'{nombre}: umbral {fila["umbral"]}; dias secos {fila["dias_secos_pct"]:.2f} %; rachas {fila["rachas_totales"]} = completas {fila["completas"]} + censuradas {fila["censuradas"]} ({fila["censuradas_pct"]:.3f} %); '
               f'completas: media {fila["media_completas"]:.2f} d, mediana {fila["mediana_completas"]:.0f} d, p99 {p99:.0f} d, max {fila["max_completas"]} d; '
               f'censuradas: media {fila["media_censuradas"]:.1f} d, max {fila["max_censuradas"]} d, >= 20 d: {fila["censuradas_ge20"]}, mayores que el p99 de las completas: {fila["censuradas_mayores_p99"]}; '
               f'suma de duraciones = dias secos: {ok}; coincide con scipy.ndimage.label: {ok2}')
    return fila


filas = []
pc = cr['pr'].values.astype('float64'); tc = cr['time'].values
filas.append(procesar('cr2met_hist', pc, tc, 0.1))
ah = xr.open_dataset(S / 'paso_1_2_aladin_hist_1980_2014.nc'); ph, th = ah['pr'].values.astype('float64'), ah['time'].values
filas.append(procesar('aladin_hist_i', ph, th, R0))
filas.append(procesar('aladin_hist_ii', ph, th, 0.1))
filas.append(procesar('aladin_hist_iii', ph, th, R0_loc))
a5 = xr.open_dataset(S / 'paso_1_2_aladin_ssp585_2040_2074.nc'); p5, t5 = a5['pr'].values.astype('float64'), a5['time'].values
filas.append(procesar('aladin_ssp585_i', p5, t5, R0))
filas.append(procesar('aladin_ssp585_ii', p5, t5, 0.1))
a3 = xr.open_dataset(S / 'paso_1_2_aladin_ssp370_2040_2074.nc'); p3, t3 = a3['pr'].values.astype('float64'), a3['time'].values
filas.append(procesar('aladin_ssp370_i', p3, t3, R0))
pd.DataFrame(filas).to_csv(S / 'paso_3_1_resumen.csv', index=False)
pd.DataFrame(reg_filas).to_csv(S / 'paso_3_1_censuradas_por_region.csv', index=False)

# ejemplo concreto: las primeras rachas de una celda cercana a Santiago (todas, con marca de censurada, reconstruidas desde completas y censuradas)
j = int(np.argmin((cr['lat_celda'].values + 33.45) ** 2 + (cr['lon_celda'].values + 70.67) ** 2))
for tag, nom in (('CR2MET', 'cr2met_hist'), ('ALADIN (criterio i)', 'aladin_hist_i')):
    c1, c2 = np.load(S / f'paso_3_1_rachas_completas_{nom}.npz'), np.load(S / f'paso_3_1_rachas_censuradas_{nom}.npz')
    fe = pd.DatetimeIndex(c1['fechas'])
    m1, m2 = c1['celda'] == j, c2['celda'] == j
    t = sorted([(int(a), int(b), False) for a, b in zip(c1['inicio'][m1], c1['duracion'][m1])] + [(int(a), int(b), True) for a, b in zip(c2['inicio'][m2], c2['duracion'][m2])])[:6]
    chk.append(f'Ejemplo, celda {j} (cercana a Santiago), primeras 6 rachas de {tag} (inicio, duracion en dias, censurada): ' + str([(str(fe[a].date()), b, c) for a, b, c in t]))
(S / 'paso_3_1_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))
