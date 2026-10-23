"""Verificacion independiente de la propuesta «Chile central».

Recalcula, con otro codigo (scipy.ndimage.label por celda, mascara con shapely.intersects_xy), y compara con las salidas:
  1. conteos N_hist y N_fut de rachas >= D (inicio mar-nov, censuradas excluidas) por region -> outputs_pregunta5_aladin/results.json
  2. estadisticos del pool anual por region y producto (n, media, mediana, p90, p99, max, n >= 20) -> resultados_tesis/tabla3_estadisticos_ventanas.csv
Escribe resultados_tesis/verificacion_independiente.txt con las diferencias maximas.
"""
import warnings
warnings.filterwarnings('ignore')

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from scipy import ndimage

import unidades as U

ROOT = Path(__file__).resolve().parent
DS = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
SALIDA = ROOT / 'resultados_tesis' / 'verificacion_independiente.txt'
lineas = []


def log(txt):
    print(txt, flush=True)
    lineas.append(txt)


def cargar_aladin(patron, ini, fin):
    ds = xr.open_mfdataset(sorted(glob.glob(str(ROOT / patron))), use_cftime=True, chunks={'time': 365})
    da = ds['pr'].sel(time=slice(ini, fin))
    return da


def rachas(seco, meses):
    """seco: (t, celdas) bool. Devuelve lista de (duracion, mes_inicio) de rachas completas (no censuradas)."""
    n = seco.shape[0]
    durs, mes = [], []
    for k in range(seco.shape[1]):
        lab, m = ndimage.label(seco[:, k])
        if m == 0:
            continue
        for i, sl in enumerate(ndimage.find_objects(lab)):
            a, b = sl[0].start, sl[0].stop
            if a == 0 or b == n:
                continue
            durs.append(b - a)
            mes.append(meses[a])
    return np.array(durs), np.array(mes)


def main():
    ala_h = cargar_aladin('pr1/pr_CHP12_*_historical_*.nc', '1980-01-01', '2014-12-31')
    ala_f = cargar_aladin('pr1/pr_CHP12_*_ssp585_*.nc', '2040-01-01', '2074-12-31')
    lat2d, lon2d = ala_h['lat'].values, ala_h['lon'].values
    z = np.load(ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz')
    chile = z['chile_mask']
    jc, ic = np.where(chile)
    recs = U.registros_chile()
    mes_h = np.array([t.month for t in ala_h['time'].values])
    mes_f = np.array([t.month for t in ala_f['time'].values])

    pr_h = ala_h.values.astype(np.float64) * 86400.0
    pr_f = ala_f.values.astype(np.float64) * 86400.0
    cr = np.load(ROOT / '_test_regrillado' / 'cr2met_lineal_vs_area.npz')['lin'].astype(np.float64)
    pos = -np.ones(chile.shape, int)
    pos[jc, ic] = np.arange(jc.size)

    res = json.load(open(ROOT / 'outputs_pregunta5_aladin' / 'results.json', encoding='utf-8'))
    curvas = res['curves']
    nombres = {'Valparaiso': 'Valparaíso', 'Metropolitana': 'Metropolitana', 'OHiggins': "O'Higgins", 'Maule': 'Maule'}
    t3 = pd.read_csv(ROOT / 'resultados_tesis' / 'tabla3_estadisticos_ventanas.csv')
    dif_rr, dif_t3 = [], []
    for clave, spec in U.UNIDADES.items():
        geom = U.registro(spec['query'], recs).geometry
        m = shapely.intersects_xy(geom, lon2d, lat2d) & chile  # otra implementacion de la mascara
        jj, ii = np.where(m)
        log(f'--- {clave}: {int(m.sum())} celdas')
        # 1) conteos de los risk ratios (inicio mar-nov), por D
        dh, mh = rachas(pr_h[:, jj, ii] < 5.285, mes_h)
        df_, mf = rachas(pr_f[:, jj, ii] < 5.285, mes_f)
        sh, sf = (mh >= 3) & (mh <= 11), (mf >= 3) & (mf <= 11)
        for c in curvas[nombres[clave]]:
            D = c['Umbral (dias)']
            n1, n2 = int((dh[sh] >= D).sum()), int((df_[sf] >= D).sum())
            dif_rr.append(max(abs(n1 - c['N1']), abs(n2 - c['N2'])))
        log(f'    RR: max |dif| de conteos sobre D = {DS}: {max(dif_rr[-len(DS):])}')
        # 2) estadisticos del pool anual (Tabla 3)
        series = {'CR2MET hist': rachas(cr[:, pos[jj, ii]] < 1.0, mes_h)[0], 'ALADIN hist': dh, 'ALADIN fut': df_}
        for prod, d in series.items():
            fila = t3[(t3['ventana'] == clave) & (t3['producto'] == prod)].iloc[0]
            mine = dict(n_spells=d.size, media=d.mean(), mediana=np.median(d), p90=np.percentile(d, 90),
                        p99=np.percentile(d, 99), max=d.max(), n_ge20=int((d >= 20).sum()))
            dd = max(abs(float(mine[k]) - float(fila[k])) for k in mine)
            dif_t3.append(dd)
            log(f'    Tabla 3 {prod}: n={d.size} media={d.mean():.3f}  max |dif| = {dd:.4g}')
    log(f'MAXIMA DIFERENCIA en conteos de RR: {max(dif_rr)}')
    log(f'MAXIMA DIFERENCIA en Tabla 3 (n, media, mediana, p90, p99, max, n>=20): {max(dif_t3):.4g}')
    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text('\n'.join(lineas) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
