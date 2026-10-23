"""Seis barras: fraccion de dias humedos (izq.) y duracion media de rachas (der.) de CR2MET (1 mm), ALADIN (1 mm) y ALADIN (R0 calibrado).
Fracciones: ../resultados_tesis/calibracion_R0.json. Duraciones: resultados_tesis/duracion_media_celdas_comunes_i.csv (mismas 3 526 celdas
para las tres series; _calcular_duracion_media_celdas_comunes.py). Salida: paper_figuras/fig_barras_calibracion.png"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
cal = json.load(open(ROOT.parent / 'resultados_tesis' / 'calibracion_R0.json'))
d = pd.read_csv(ROOT / 'resultados_tesis' / 'duracion_media_celdas_comunes_i.csv')
fmt = lambda v, k=1: f'{v:.{k}f}'.replace('.', ',')
GRIS, ROJO, AZUL = '#8a8984', '#d4564e', '#3f77d1'
lab = ['CR2MET, 1 mm', 'ALADIN, 1 mm', 'ALADIN, R₀ = 5,285 mm']
fr = [cal['F_cr2met_1mm_pct'], cal['F_aladin_1mm_pct'], cal['F_aladin_R0_usado_pct']]
du = list(d['duracion_media_dias'])
n = int(d['celdas'][0])
nstr = f'{n:,}'.replace(',', ' ')
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
for ax, vals, tit, k, xm in ((axs[0], fr, 'Fracción de días húmedos (%)', 2, 38),
                             (axs[1], du, 'Duración media de rachas, país (días)', 1, 52)):
    ax.barh([2, 1, 0], vals, color=[GRIS, ROJO, AZUL], height=0.55)
    for y, x in zip([2, 1, 0], vals):
        ax.text(x + xm * 0.012, y, fmt(x, k), va='center', fontsize=10, fontweight='bold')
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(lab); ax.set_xlim(0, xm)
    ax.set_title(tit, loc='left', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
axs[1].set_yticklabels([''] * 3)
fig.text(0.01, 0.02,
         f'Con 1 mm, ALADIN tiene {fmt(cal["exceso_aladin_1mm_pp"])} puntos porcentuales más de días húmedos y sus rachas son {fmt(du[0] - du[1])} días más cortas.\n'
         f'Con R₀ la frecuencia coincide y la diferencia baja a {fmt(du[0] - du[2])} días. Duraciones: promedio sobre las mismas {nstr} celdas (≥ 30 rachas en CR2MET y en ALADIN con R₀).',
         fontsize=9, color='#444', va='bottom')
fig.tight_layout(rect=(0, 0.11, 1, 1))
out = ROOT / 'paper_figuras' / 'fig_barras_calibracion.png'
fig.savefig(out, dpi=200)
print(out)
