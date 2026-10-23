"""Tabla 3: estadisticos del pool regional de duraciones de dry spells por ventana y producto (criterio i).

Lee las duraciones por ventana generadas por _generar_pregunta9_pdf_loglog_cuencas.py (cache
_pregunta9_outputs/pdf_durations_i_global.pkl; reglas §2.2: rachas censuradas excluidas, R0 = 5,285 mm en ALADIN,
1 mm en CR2MET). Calcula n_spells, media, mediana, p90, p99, maximo y numero/fraccion de rachas >= 20 d sin redondear, y el sesgo relativo
(ALADIN hist - CR2MET) / CR2MET de la media y del p99 del pool.

Tambien calcula, con las duraciones de la Figura S1 (_pregunta9_calibracion_outputs/spell_durations_sensibilidad_4regiones.pkl),
el sesgo relativo de la media y del p99 del pool de cada ventana para los tres criterios de umbral (ii: 1 mm; i: R0 global;
iii: R0 local), respecto de CR2MET a 1 mm.

Salidas: resultados_tesis/tabla3_estadisticos_ventanas.csv y resultados_tesis/sesgo_relativo_criterios_ventanas.csv
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT / '_pregunta9_outputs' / 'pdf_durations_i_global.pkl'
SALIDA = ROOT / 'resultados_tesis' / 'tabla3_estadisticos_ventanas.csv'
ENTRADA_S1 = ROOT / '_pregunta9_calibracion_outputs' / 'spell_durations_sensibilidad_4regiones.pkl'
SALIDA_S1 = ROOT / 'resultados_tesis' / 'sesgo_relativo_criterios_ventanas.csv'
CRITERIOS = [('ALADIN 1 mm', 'ii (1 mm)'), ('ALADIN R0 global', 'i (R0 global)'), ('ALADIN R0 local', 'iii (R0 local)')]
PRODUCTOS = [('CR2MET (hist)', 'CR2MET hist'), ('ALADIN (hist)', 'ALADIN hist'), ('ALADIN (futuro)', 'ALADIN fut')]


def main():
    with ENTRADA.open('rb') as f:
        durs = pickle.load(f)
    filas = []
    for ventana, por_producto in durs.items():
        for clave, etiqueta in PRODUCTOS:
            d = np.asarray(por_producto[clave], dtype=float)
            filas.append({
                'ventana': ventana, 'producto': etiqueta, 'n_spells': int(d.size),
                'media': d.mean(), 'mediana': np.median(d), 'p90': np.percentile(d, 90),
                'p99': np.percentile(d, 99), 'max': d.max(),
                'n_ge20': int((d >= 20).sum()), 'frac_ge20_pct': 100 * float((d >= 20).mean()),
            })
    df = pd.DataFrame(filas)
    ref = df[df['producto'] == 'CR2MET hist'].set_index('ventana')
    for col in ('media', 'p99'):
        df[f'{col}_sesgo_rel_pct'] = np.nan
        for v in ref.index:
            m = (df['ventana'] == v) & (df['producto'] == 'ALADIN hist')
            df.loc[m, f'{col}_sesgo_rel_pct'] = 100 * (df.loc[m, col] - ref.loc[v, col]) / ref.loc[v, col]
    SALIDA.parent.mkdir(exist_ok=True)
    df.round(3).to_csv(SALIDA, index=False)
    print(df.round(2).to_string(index=False))

    with ENTRADA_S1.open('rb') as f:
        s1 = pickle.load(f)
    filas = []
    for ventana, series in s1.items():
        ref = np.asarray(series['CR2MET 1 mm'], dtype=float)
        for clave, etiqueta in CRITERIOS:
            d = np.asarray(series[clave], dtype=float)
            filas.append({
                'ventana': ventana, 'criterio': etiqueta, 'n_spells_ALADIN': int(d.size), 'n_spells_CR2MET': int(ref.size),
                'media_ALADIN': d.mean(), 'media_CR2MET': ref.mean(),
                'media_sesgo_rel_pct': 100 * (d.mean() - ref.mean()) / ref.mean(),
                'p99_ALADIN': np.percentile(d, 99), 'p99_CR2MET': np.percentile(ref, 99),
                'p99_sesgo_rel_pct': 100 * (np.percentile(d, 99) - np.percentile(ref, 99)) / np.percentile(ref, 99),
            })
    rel = pd.DataFrame(filas)
    rel.round(3).to_csv(SALIDA_S1, index=False)
    print(rel.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
