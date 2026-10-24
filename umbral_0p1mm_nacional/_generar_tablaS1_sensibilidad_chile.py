"""Tabla S1: sesgo historico ALADIN - CR2MET en Chile continental para los tres criterios de umbral (§2.7).

Lee los resultados de _run_pregunta7_bootstrap.py (no recalcula nada).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESUMEN = ROOT / '_pregunta7_outputs' / 'resumen_pregunta7.csv'
BOOT = ROOT / 'outputs_pregunta7_bootstrap' / 'bootstrap_summary.csv'
OUT_CSV = ROOT / 'paper_figuras' / 'tablaS1_sensibilidad_umbral_chile.csv'
CRITERIOS = [
    ('A_opcion_B_integrado', 'Forma A', 'i — R0 global (3,668 mm)'),
    ('B_mismo_umbral', 'Forma B', 'ii — 0.1 mm en ambos'),
    ('C_umbral_local', 'Forma C', 'iii — R0 local por píxel'),
]
METRICAS = [('Duracion media', 'Duracion media', 'Duración media'), ('t99', 't99', 't99')]


def main():
    res = pd.read_csv(RESUMEN)
    boot = pd.read_csv(BOOT)
    rows = []
    for esc, forma, crit in CRITERIOS:
        for met_res, met_boot, met_label in METRICAS:
            r = res[(res['escenario'] == esc) & (res['metrica'] == met_res)].set_index('dataset')
            b = boot[boot['Escenario'].str.startswith(forma) & boot['Metrica'].str.startswith(met_boot)].iloc[0]
            rows.append({
                'Criterio': crit,
                'Métrica': met_label,
                'CR2MET (días)': r.loc['CR2MET', 'mean_espacial'],
                'ALADIN (días)': r.loc['ALADIN', 'mean_espacial'],
                'Δ medio (días)': r.loc['DELTA', 'mean_espacial'],
                'Δ mediano (días)': r.loc['DELTA', 'median_espacial'],
                '% celdas ALADIN < CR2MET (sig.)': b['% ALADIN < CR2MET (sig.)'],
                '% celdas ALADIN > CR2MET (sig.)': b['% ALADIN > CR2MET (sig.)'],
                'Celdas': int(b['Celdas validas']),
            })
    df = pd.DataFrame(rows)
    df.round(1).to_csv(OUT_CSV, index=False)
    print(df.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
