"""Porcentaje de celdas con diferencia significativa ALADIN - CR2MET (bootstrap de anios, IC95 por celda, criterio i, 1980-2014)
dentro de cada region de la propuesta y en todo el dominio. Usa los mapas ya calculados por _run_pregunta7_bootstrap.py
(../resultados_tesis/bootstrap_por_celda_A_opcion_B_integrado_{mean,t99}.npz; solo lectura). Celdas = validas (>= 30 rachas en ambos).
Salida: resultados_tesis/significancia_celdas_regiones.csv"""
from pathlib import Path
import numpy as np
import pandas as pd
import unidades as U

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'resultados_tesis'
z = np.load(ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz')
lat, lon, chile = z['lat'], z['lon'], z['chile_mask']
zonas = dict(U.mascaras(lat, lon, chile))
zonas['Todo el dominio'] = chile
filas = []
for met, nombre in (('mean', 'Duración media'), ('t99', 'Percentil 99')):
    b = np.load(BASE / f'bootstrap_por_celda_A_opcion_B_integrado_{met}.npz')
    valida, sig, mas_largo, mas_corto = b['valida'], b['significativo'], b['mas_largo'], b['mas_corto']
    for nom, m in zonas.items():
        sel = valida & m
        n = int(sel.sum())
        filas.append(dict(metrica=nombre, zona=nom, celdas_validas=n,
                          pct_significativas=100 * sig[sel].mean(),
                          pct_aladin_mas_corto_sig=100 * (sig & mas_corto)[sel].mean(),
                          pct_aladin_mas_largo_sig=100 * (sig & mas_largo)[sel].mean(),
                          n_mas_corto_sig=int((sig & mas_corto)[sel].sum())))
df = pd.DataFrame(filas)
df.to_csv(ROOT / 'resultados_tesis' / 'significancia_celdas_regiones.csv', index=False)
print(df.round(1).to_string(index=False))
