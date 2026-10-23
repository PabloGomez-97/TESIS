"""Cifras derivadas que usan los Docs v2 y que no estan literalmente en un CSV/JSON de salida.

Cada cifra sale de otras salidas (de esta carpeta o, en solo lectura, del repo) o de las caches del dominio completo; nada se
escribe a mano. Salida: resultados_tesis/cifras_derivadas.json

Contenido:
  unidades: celdas y latitud minima/maxima de las celdas de cada region; bins efectivos de las PDFs por region
  percentil_D20: porcentaje de las rachas (inicio mar-nov, ALADIN historico) con duracion >= 20 d, por region (results.json)
  tabla3_derivadas: cambio relativo (futuro/historico) de n_spells, media y p99 de ALADIN por region (tabla3_estadisticos_ventanas.csv)
  precipitacion: cociente de la precipitacion anual ALADIN/CR2MET por region (estacionalidad_marnov_regiones.csv)
  r0_local: estadisticos del mapa de R0 local (R0_local_map.npz del repo, dominio completo)
  celdas_excluidas: celdas de Chile sin >= 30 rachas en ambos productos (cache de mapas del repo)
  regrillado: diferencias maximas lineal vs area (mapas_lineal_vs_area.csv y r0_lineal_vs_area.json del repo)
"""
import warnings
warnings.filterwarnings('ignore')

import json
from pathlib import Path

import numpy as np
import pandas as pd

import unidades as U
import _generar_pregunta9_pdf_loglog_cuencas as f6

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / 'resultados_tesis' / 'cifras_derivadas.json'


def main():
    z = np.load(ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz')
    lat, lon, chile = z['lat'], z['lon'], z['chile_mask']
    masks = U.mascaras(lat, lon, chile)
    res = {}

    # 1) unidades
    res['unidades'] = {k: {'celdas': int(m.sum()), 'lat_min': float(lat[m].min()), 'lat_max': float(lat[m].max())}
                       for k, m in masks.items()}
    res['unidades_lat_total'] = {'min': min(v['lat_min'] for v in res['unidades'].values()),
                                 'max': max(v['lat_max'] for v in res['unidades'].values())}
    import pickle
    durs = pickle.load(open(ROOT / '_pregunta9_outputs' / 'pdf_durations_i_global.pkl', 'rb'))
    res['bins_efectivos'] = {k: int(len(f6.common_log_bins(v)) - 1) for k, v in durs.items()}
    res['max_cr2met'] = {k: int(np.max(v['CR2MET (hist)'])) for k, v in durs.items()}

    # 2) percentil que corresponde a D = 20 d en las rachas con inicio mar-nov (ALADIN historico)
    rr = json.load(open(ROOT / 'outputs_pregunta5_aladin' / 'results.json', encoding='utf-8'))
    perc = {}
    summ = rr['summary']
    for reg in sorted({r['Region'] for r in summ}):
        h = [r for r in summ if r['Region'] == reg and r['Periodo'] == '1980-2014'][0]
        perc[reg] = {'frac_ge20_pct': 100 * h['Spells >= 20 d'] / h['Dry spells totales'],
                     'percentil': 100 - 100 * h['Spells >= 20 d'] / h['Dry spells totales'],
                     'total_hist': h['Dry spells totales'], 'ge20_hist': h['Spells >= 20 d']}
    res['percentil_D20'] = perc

    # 3) Tabla 3: cambios relativos
    t3 = pd.read_csv(ROOT / 'resultados_tesis' / 'tabla3_estadisticos_ventanas.csv')
    der = {}
    for v in t3.ventana.unique():
        h = t3[(t3.ventana == v) & (t3.producto == 'ALADIN hist')].iloc[0]
        f = t3[(t3.ventana == v) & (t3.producto == 'ALADIN fut')].iloc[0]
        c = t3[(t3.ventana == v) & (t3.producto == 'CR2MET hist')].iloc[0]
        der[v] = {'n_spells_cambio_pct': 100 * (f.n_spells / h.n_spells - 1),
                  'media_cambio_pct': 100 * (f.media / h.media - 1), 'p99_cambio_pct': 100 * (f.p99 / h.p99 - 1),
                  'frac_ge20_hist': h.frac_ge20_pct, 'frac_ge20_fut': f.frac_ge20_pct,
                  'p99_cr2met': c.p99}
    res['tabla3_derivadas'] = der
    res['razon_p99_cr2met_mayor_menor'] = float(t3[t3.producto == 'CR2MET hist'].p99.max() / t3[t3.producto == 'CR2MET hist'].p99.min())

    # 4) precipitacion anual
    e = pd.read_csv(ROOT / 'resultados_tesis' / 'estacionalidad_marnov_regiones.csv')
    res['precipitacion'] = {r.region: {'razon': r.ALADIN_total_mm_anio / r.CR2MET_total_mm_anio} for r in e.itertuples()}
    res['precipitacion_razon_rango'] = {'min': min(v['razon'] for v in res['precipitacion'].values()),
                                        'max': max(v['razon'] for v in res['precipitacion'].values())}

    # 5) R0 local (dominio completo; mapa del repo)
    r0 = np.load(REPO / '_pregunta7_outputs' / 'R0_local_map.npz')
    ch0, v = r0['chile'], r0['r0'].astype(float)[r0['chile']]
    fin = np.isfinite(v)
    q3 = np.nanpercentile(v, 75)
    la = r0['lat'][ch0]
    res['r0_local'] = {'celdas': int(ch0.sum()), 'sin_solucion': int((~fin).sum()),
                       'mediana': float(np.nanmedian(v)), 'p10': float(np.nanpercentile(v, 10)),
                       'p90': float(np.nanpercentile(v, 90)),
                       'pct_menor_global_de_validas': float(100 * (v[fin] < 5.285).mean()),
                       'lat_cuartil_superior_p10_p90': [float(x) for x in np.percentile(la[fin & (v >= q3)], [10, 90])]}

    # 6) celdas excluidas de los mapas (dominio completo)
    ok = np.isfinite(z['mean_cr2']) & np.isfinite(z['mean_ala']) & chile
    ex = chile & ~ok
    res['celdas_excluidas'] = {'n': int(ex.sum()), 'lat_min': float(lat[ex].min()), 'lat_max': float(lat[ex].max()),
                               'comunes': int(ok.sum()), 'chile': int(chile.sum())}

    # 7) regrillado (dominio completo; salidas del repo, solo lectura)
    mp = pd.read_csv(REPO / '_test_regrillado' / 'mapas_lineal_vs_area.csv')
    dif = {}
    for met in ('media', 't99'):
        d = []
        for crit in ('i', 'ii', 'iii'):
            a = mp[(mp.cr2met == 'lineal') & (mp.criterio == crit) & (mp.metrica == met)].Delta.iloc[0]
            b = mp[(mp.cr2met == 'area') & (mp.criterio == crit) & (mp.metrica == met)].Delta.iloc[0]
            d.append(abs(a - b))
        dif[met] = float(max(d))
    rj = json.load(open(REPO / '_test_regrillado' / 'r0_lineal_vs_area.json'))
    reg = pd.read_csv(ROOT / '_test_regrillado' / 'regiones_cr2met_lineal_vs_area.csv')
    d_reg = {'media': 0.0, 'p90': 0.0, 'p99': 0.0}
    for k in reg.region.unique():
        a = reg[(reg.region == k) & (reg.cr2met == 'lineal')].iloc[0]
        b = reg[(reg.region == k) & (reg.cr2met == 'area')].iloc[0]
        for m in d_reg:
            d_reg[m] = max(d_reg[m], abs(float(a[m]) - float(b[m])))
    res['regrillado'] = {'dif_max_delta': dif,
                         'dif_fwet_pp': rj['area']['fwet_cr2_1mm_pct'] - rj['lineal']['fwet_cr2_1mm_pct'],
                         'dif_max_regiones': d_reg}
    nin = np.load(REPO / '_test_regrillado' / 'cr2met_lineal_vs_area.npz')['n_inside']
    res['regrillado']['mediana_centros'] = float(np.median(nin))

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
