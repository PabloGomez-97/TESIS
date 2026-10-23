"""Sensibilidad al regrillado de CR2MET (lineal frente a promedio por area) en las 4 regiones de la propuesta «Chile central».

Los resultados de dominio completo (fwet, R0 calibrado, mapas y Delta de los criterios i, ii y iii) no dependen de las unidades y
siguen siendo los de _test_regrillado_conservativo.py del repo. Aqui solo se recalculan, desde la misma cache
(_test_regrillado/cr2met_lineal_vs_area.npz), los estadisticos del pool anual de CR2MET por region con ambos regrillados
(mismas reglas de rachas: NaN interrumpe, censuradas excluidas, umbral 1 mm).

Salida: _test_regrillado/regiones_cr2met_lineal_vs_area.csv
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd

import unidades as U
from _test_regrillado_conservativo import spells_per_cell

ROOT = Path(__file__).resolve().parent


def main():
    z = np.load(ROOT / '_test_regrillado' / 'cr2met_lineal_vs_area.npz')
    lin, area = z['lin'], z['area']
    m = np.load(ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz')
    chile = m['chile_mask']
    jj, ii = np.where(chile)
    masks = U.mascaras(m['lat'], m['lon'], chile)
    filas = []
    for nombre, cr in (('lineal', lin), ('area', area)):
        durs = spells_per_cell(cr, 1.0)
        for clave, mk in masks.items():
            d = np.concatenate([durs[k] for k in np.where(mk[jj, ii])[0]])
            filas.append({'region': clave, 'cr2met': nombre, 'n_spells': d.size, 'media': d.mean(),
                          'p90': np.percentile(d, 90), 'p99': np.percentile(d, 99), 'max': d.max()})
    df = pd.DataFrame(filas)
    out = ROOT / '_test_regrillado' / 'regiones_cr2met_lineal_vs_area.csv'
    df.to_csv(out, index=False)
    print(df.round(2).to_string(index=False))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
