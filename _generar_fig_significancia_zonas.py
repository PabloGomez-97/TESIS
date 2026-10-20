"""Porcentaje de celdas con diferencia significativa ALADIN - CR2MET por zona (criterio i, 1980-2014).

Entrada: resultados_tesis/bootstrap_significancia_por_zona.csv. Salida: paper_figuras/fig_significancia_zonas.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZONAS = ['Norte árido', 'Norte semiárido', 'Centro mediterráneo', 'Sur templado', 'Las 14 regiones']


def main():
    df = pd.read_csv(ROOT / 'resultados_tesis' / 'bootstrap_significancia_por_zona.csv')
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, met in zip(ejes, ('Duración media', 'Percentil 99')):
        d = df[(df.metrica == met) & df.unidad.isin(ZONAS)].set_index('unidad').loc[ZONAS]
        y = np.arange(len(ZONAS))[::-1]
        ax.barh(y, d.pct_mas_corto, color='#E24B4A', label='ALADIN más corto (IC95 % bajo cero)')
        ax.barh(y, d.pct_sin_diferencia, left=d.pct_mas_corto, color='#D3D1C7', label='Sin diferencia clara')
        ax.barh(y, d.pct_mas_largo, left=d.pct_mas_corto + d.pct_sin_diferencia, color='#378ADD', label='ALADIN más largo')
        for yi, a, b in zip(y, d.pct_mas_corto, d.pct_mas_largo):
            ax.text(2, yi, f'{a:.1f} %'.replace('.', ','), va='center', fontsize=9, color='white', fontweight='bold')
            ax.text(98, yi, f'{b:.1f} %'.replace('.', ','), va='center', ha='right', fontsize=8, color='white')
        ax.set_xlim(0, 100); ax.set_title(met, fontsize=11); ax.set_xlabel('% de celdas')
        ax.set_yticks(y); ax.set_yticklabels([f'{z}\n({int(d.loc[z, "celdas"])} celdas)' for z in ZONAS], fontsize=9)
    ejes[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, -0.38), ncol=3)
    fig.suptitle('Celdas donde la diferencia ALADIN − CR2MET es real (bootstrap por años, criterio i)', fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / 'paper_figuras' / 'fig_significancia_zonas.png', dpi=170, facecolor='white', bbox_inches='tight')


if __name__ == '__main__':
    main()
