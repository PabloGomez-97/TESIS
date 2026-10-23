"""Cuanto acorta ALADIN las rachas respecto de CR2MET (1980-2014, criterio i), por region y para todo el pais.
Misma figura que la de macrozonas, con las 4 regiones de la propuesta. Barra llena = promedio de las diferencias por celda;
barra hueca = mediana. Entrada: resultados_tesis/sesgo_por_ventana.csv (_calcular_sesgo_ventanas.py).
Salida: paper_figuras/fig_acorta_regiones.png"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import unidades as U

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / 'resultados_tesis' / 'sesgo_por_ventana.csv')
orden = list(U.UNIDADES) + ['Todo el dominio']
etq = {k: v['etiqueta'] for k, v in U.UNIDADES.items()}
etq['Todo el dominio'] = 'Todo el país'
col = {k: v['color'] for k, v in U.UNIDADES.items()}
col['Todo el dominio'] = '#7f7f7a'

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
fig.suptitle('Cuánto acorta ALADIN las rachas respecto de CR2MET (1980–2014, criterio i)', fontsize=12)
for ax, met, tit in zip(axs, ['Duración media', 'Percentil 99'],
                        ['Duración media (días): ALADIN − CR2MET', 'Percentil 99 (días): ALADIN − CR2MET']):
    d = df[df.metrica == met].set_index('zona')
    for i, z in enumerate(orden):
        y = len(orden) - 1 - i
        m, md = d.loc[z, 'delta_promedio'], d.loc[z, 'delta_mediana']
        ax.barh(y + 0.19, m, height=0.34, color=col[z])
        ax.barh(y - 0.19, md, height=0.34, facecolor='white', edgecolor=col[z], linewidth=1.6)
        ax.text(m, y + 0.19, f'{m:.1f}'.replace('.', ',') + ' ', ha='right', va='center', fontsize=8, fontweight='bold')
        ax.text(md, y - 0.19, f'{md:.1f}'.replace('.', ',') + ' ', ha='right', va='center', fontsize=8)
    ax.axvline(0, color='k', lw=1)
    ax.set_title(tit, fontsize=10)
    ax.grid(axis='x', alpha=0.25)
    lo = min(d.delta_promedio.min(), d.delta_mediana.min())
    ax.set_xlim(lo * 1.22, 0)
ncel = df[df.metrica == 'Duración media'].set_index('zona')['celdas_validas']
axs[0].set_yticks(range(len(orden)))
axs[0].set_yticklabels([f'{etq[z]}\n({int(ncel[z]):,} celdas)'.replace(',', ' ') for z in reversed(orden)])
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='#555'), plt.Rectangle((0, 0), 1, 1, fc='white', ec='#555', lw=1.6)],
              labels=['Promedio de las diferencias', 'Mediana de las diferencias'], loc='lower center', ncol=2, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
out = ROOT / 'paper_figuras' / 'fig_acorta_regiones.png'
fig.savefig(out, dpi=200)
print(out)
