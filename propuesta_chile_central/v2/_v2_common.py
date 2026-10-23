"""Helpers y carga de datos para generar los Docs v2 (propuesta «Chile central»). Todas las cifras salen de las salidas."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent      # propuesta_chile_central/
REPO = ROOT.parent
KEYS = ['Valparaiso', 'Metropolitana', 'OHiggins', 'Maule']
LAB = {'Valparaiso': 'Valparaíso', 'Metropolitana': 'Metropolitana', 'OHiggins': "O'Higgins", 'Maule': 'Maule'}
LAB2KEY = {v: k for k, v in LAB.items()}


def fnum(x, nd=1):
    """Numero con coma decimal y signo menos tipografico."""
    s = f'{x:.{nd}f}'.replace('.', ',')
    return s.replace('-', '−')


def fint(n):
    """Entero con espacio como separador de miles (>= 1000)."""
    n = int(round(n))
    s = f'{abs(n):,}'.replace(',', ' ')
    return ('−' if n < 0 else '') + s


def ci(lo, hi, nd=2):
    return f'[{fnum(lo, nd)}; {fnum(hi, nd)}]'


def lista(items, conj='y'):
    items = list(items)
    if len(items) <= 1:
        return ''.join(items)
    return ', '.join(items[:-1]) + f' {conj} ' + items[-1]


# --- datos de esta carpeta
t3 = pd.read_csv(ROOT / 'resultados_tesis' / 'tabla3_estadisticos_ventanas.csv')
rel = pd.read_csv(ROOT / 'resultados_tesis' / 'sesgo_relativo_criterios_ventanas.csv')
frec = pd.read_csv(ROOT / 'resultados_tesis' / 'frecuencia_humeda_ventanas.csv').set_index('zona')
sesgo = pd.read_csv(ROOT / 'resultados_tesis' / 'sesgo_por_ventana.csv')
est = pd.read_csv(ROOT / 'resultados_tesis' / 'estacionalidad_marnov_regiones.csv').set_index('region')
cens = pd.read_csv(ROOT / '_test_regrillado' / 'rachas_censuradas_por_cuenca.csv')
mk = pd.read_csv(ROOT / 'resultados_tesis' / 'descomposicion_markov_rr_regiones.csv')
ctr = pd.read_csv(ROOT / 'resultados_tesis' / 'contraste_persistencia_regiones.csv').set_index('region')
res_s = pd.read_csv(ROOT / '_pregunta9_calibracion_outputs' / 'resumen_sensibilidad_umbral_4regiones.csv')
RR = {n: json.load(open(ROOT / d / 'results.json', encoding='utf-8'))
      for n, d in (('main', 'outputs_pregunta5_aladin'), ('mm1', 'outputs_pregunta5_aladin_1mm'),
                   ('s370', 'outputs_pregunta5_aladin_ssp370'))}
DER = json.load(open(ROOT / 'resultados_tesis' / 'cifras_derivadas.json', encoding='utf-8'))
# --- datos del repo (dominio completo, solo lectura)
cal = json.load(open(REPO / 'resultados_tesis' / 'calibracion_R0.json'))
tS1 = pd.read_csv(REPO / 'paper_figuras' / 'tablaS1_sensibilidad_umbral_chile.csv')
minr = pd.read_csv(REPO / 'resultados_tesis' / 'sensibilidad_minimo_rachas.csv')
boot = pd.read_csv(REPO / 'outputs_pregunta7_bootstrap' / 'bootstrap_summary.csv')
dom = sesgo[sesgo.zona == 'Todo el dominio']
r0json = json.load(open(REPO / '_test_regrillado' / 'r0_lineal_vs_area.json'))
mapas = pd.read_csv(REPO / '_test_regrillado' / 'mapas_lineal_vs_area.csv')


def t3row(key, prod):
    return t3[(t3.ventana == key) & (t3.producto == prod)].iloc[0]


def relrow(key, crit):
    return rel[(rel.ventana == key) & (rel.criterio == crit)].iloc[0]


def rr_row(which, key):
    lab = LAB[key]
    return [r for r in RR[which]['rr_table'] if r['Region'] == lab][0]


def curve(which, key):
    return RR[which]['curves'][LAB[key]]


def cpt(which, key, D):
    return [c for c in curve(which, key) if c['Umbral (dias)'] == D][0]


def sig_D(which, key):
    """Valores de D con IC95 completamente sobre 1 (sin replicas descartadas; limite inferior redondeado a 2 decimales > 1,00)."""
    return [c['Umbral (dias)'] for c in curve(which, key) if c['Replicas descartadas'] == 0 and round(c['IC95 inferior'], 2) > 1.0]


def cens_row(serie, key):
    return cens[(cens.serie == serie) & (cens.cuenca == key)].iloc[0]
