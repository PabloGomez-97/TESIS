"""Fraccion de dias humedos por ventana y para el dominio: historico 1980-2014 (criterio i y 1 mm) y futuro 2040-2074.

R0 se calibra para igualar la fraccion media del dominio completo; este script mide cuanto se cumple esa igualdad dentro
de cada ventana (Loa, Maipo, Maule, Biobio) y cuanto cambia con R0 fijo en SSP5-8.5 y SSP3-7.0. Fraccion = promedio espacial (igual peso por celda) de la fraccion temporal
de dias con pr >= umbral. CR2MET: 1 mm, interpolado linealmente a la grilla ALADIN; ALADIN: R0 = 5,285 mm y 1 mm.
Mismas ventanas y mascara de Chile que _generar_pregunta9_pdf_loglog_cuencas.py.

Salida: resultados_tesis/frecuencia_humeda_ventanas.csv
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd

import _generar_pregunta9_pdf_loglog_cuencas as f6

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis' / 'frecuencia_humeda_ventanas.csv'


def main():
    ala = f6.open_aladin_period(f6.HIST_START, f6.HIST_END)
    cr2 = f6.open_cr2met_period(f6.HIST_START, f6.HIST_END).interp(lat=ala['lat'], lon=ala['lon'], method='linear')
    lat2d, lon2d = ala['lat'].values, ala['lon'].values
    chile = f6.build_chile_mask_on_aladin_grid(lat2d, lon2d, f6.load_chile_geometry())
    zonas = {'Dominio': chile}
    for nombre, spec in f6.BASIN_SPECS.items():
        zonas[nombre] = f6.build_basin_mask(lat2d, lon2d, chile, spec['bounds'])
    jj, ii = np.where(chile)
    a = ala.values[:, jj, ii]
    c = cr2.values[:, jj, ii]
    fut = {}
    for esc in ('ssp585', 'ssp370'):
        pr_f = f6.open_aladin_period(f6.FUT_START, f6.FUT_END, scenario=esc)
        fut[esc] = pr_f.values[:, jj, ii]
    en_chile = np.zeros(chile.shape, dtype=int)
    en_chile[jj, ii] = np.arange(jj.size)
    filas = []
    for nombre, m in zonas.items():
        cols = en_chile[m]
        f = lambda x, t: float((x[:, cols] >= t).mean(axis=0).mean() * 100)
        filas.append({
            'zona': nombre, 'celdas': int(m.sum()),
            'CR2MET_1mm_pct': f(c, 1.0), 'ALADIN_R0_pct': f(a, f6.TAU_ALADIN_DOMINIO), 'ALADIN_1mm_pct': f(a, 1.0),
            'ALADIN_R0_ssp585_2040_2074_pct': f(fut['ssp585'], f6.TAU_ALADIN_DOMINIO),
            'ALADIN_R0_ssp370_2040_2074_pct': f(fut['ssp370'], f6.TAU_ALADIN_DOMINIO),
        })
    df = pd.DataFrame(filas)
    df['exceso_R0_pp'] = df['ALADIN_R0_pct'] - df['CR2MET_1mm_pct']
    df['exceso_1mm_pp'] = df['ALADIN_1mm_pct'] - df['CR2MET_1mm_pct']
    for esc in ('ssp585', 'ssp370'):
        col = f'ALADIN_R0_{esc}_2040_2074_pct'
        df[f'cambio_rel_{esc}_pct'] = 100 * (df[col] - df['ALADIN_R0_pct']) / df['ALADIN_R0_pct']
    OUT.parent.mkdir(exist_ok=True)
    df.round(3).to_csv(OUT, index=False)
    print(df.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
