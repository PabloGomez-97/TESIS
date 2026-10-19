"""Chequeo de coherencia: resultados con las 4 ventanas antiguas (Loa, Maipo, Maule, Biobio) frente a las 4 macrozonas.

Para cada zona se calcula, con las mismas rachas (criterio i, todo el anio, censuradas excluidas):
  - sesgo historico relativo de ALADIN respecto de CR2MET (media y percentil 99),
  - cambio futuro relativo de ALADIN (2040-2074 SSP5-8.5 frente a 1980-2014).
Si ambas geografias cuentan la misma historia, los signos y el orden norte-sur deben coincidir.

Entradas: _pregunta9_outputs/resumen_pregunta9.csv (ventanas, Tabla 3 del paper) y resultados_tesis/tabla3_macrozonas.csv
Salida: resultados_tesis/comparacion_ventanas_macrozonas.csv
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EQUIV = {'Loa': 'Norte árido', 'Maipo': 'Centro mediterráneo', 'Maule': 'Centro mediterráneo', 'Biobio': 'Sur templado'}


def tabla(df, nombres, geografia):
    filas = []
    for z in nombres:
        d = df[df['zona'] == z].set_index('conjunto')
        filas.append(dict(
            geografia=geografia, zona=z,
            media_cr2met=d.loc['cr2met', 'media'], media_aladin=d.loc['aladin_hist', 'media'], media_futuro=d.loc['aladin_fut', 'media'],
            p99_cr2met=d.loc['cr2met', 'p99'], p99_aladin=d.loc['aladin_hist', 'p99'], p99_futuro=d.loc['aladin_fut', 'p99']))
    t = pd.DataFrame(filas)
    t['sesgo_media_pct'] = (t.media_aladin / t.media_cr2met - 1) * 100
    t['sesgo_p99_pct'] = (t.p99_aladin / t.p99_cr2met - 1) * 100
    t['cambio_futuro_media_pct'] = (t.media_futuro / t.media_aladin - 1) * 100
    t['cambio_futuro_p99_pct'] = (t.p99_futuro / t.p99_aladin - 1) * 100
    return t


def main():
    v = pd.read_csv(ROOT / '_pregunta9_outputs' / 'resumen_pregunta9.csv')
    v = v[v.criterio == 'i_global'].rename(columns={'cuenca': 'zona', 'dataset': 'conjunto', 'mean_dias': 'media', 'p99_dias': 'p99'})
    v['conjunto'] = v['conjunto'].map({'CR2MET (hist)': 'cr2met', 'ALADIN (hist)': 'aladin_hist', 'ALADIN (futuro)': 'aladin_fut'})
    m = pd.read_csv(ROOT / 'resultados_tesis' / 'tabla3_macrozonas.csv').rename(columns={'macrozona': 'zona'})
    m['conjunto'] = m['conjunto'].map({'cr2met_hist': 'cr2met', 'aladin_hist': 'aladin_hist', 'aladin_ssp585': 'aladin_fut'})
    out = pd.concat([tabla(v, ['Loa', 'Maipo', 'Maule', 'Biobio'], 'ventana'),
                     tabla(m, ['Norte árido', 'Norte semiárido', 'Centro mediterráneo', 'Sur templado'], 'macrozona')])
    out['equivalente'] = out.zona.map(EQUIV)
    out.to_csv(ROOT / 'resultados_tesis' / 'comparacion_ventanas_macrozonas.csv', index=False)
    cols = ['zona', 'media_cr2met', 'media_aladin', 'media_futuro', 'sesgo_media_pct', 'cambio_futuro_media_pct',
            'p99_cr2met', 'p99_aladin', 'p99_futuro', 'sesgo_p99_pct', 'cambio_futuro_p99_pct']
    print(out[cols].round(1).to_string(index=False))


if __name__ == '__main__':
    main()
