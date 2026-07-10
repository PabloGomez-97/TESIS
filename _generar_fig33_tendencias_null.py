"""Figura §3.3 — tendencias ALADIN histórico (resultado nulo)."""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper_figuras'
OUT.mkdir(exist_ok=True)

# Valores de pregunta8_tendencias_dryspells_aladin.ipynb (resumen espacial)
STATS = {
    'Duracion media': {
        'mean_slope': -0.027,
        'pct_sig': 0.035,
        'color': '#34495E',
    },
    't99 (p99)': {
        'mean_slope': 0.001,
        'pct_sig': 0.115,
        'color': '#8E44AD',
    },
}


def build_summary_figure():
    """Barras: pendiente media + % celdas significativas (resultado nulo)."""
    metrics = list(STATS.keys())
    slopes = [STATS[m]['mean_slope'] for m in metrics]
    pcts = [STATS[m]['pct_sig'] for m in metrics]
    colors = [STATS[m]['color'] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Panel A: pendiente media espacial
    ax = axes[0]
    bars = ax.bar(metrics, slopes, color=colors, edgecolor='white', width=0.55)
    ax.axhline(0, color='gray', linewidth=0.9, linestyle='--')
    ax.set_ylabel('Pendiente media espacial\n(dias/decada)', fontsize=9)
    ax.set_title('Tendencia lineal 1980-2014\n(ALADIN, R0 = 5.285 mm)', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, slopes):
        ax.annotate(
            f'{val:+.3f}', xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 6 if val >= 0 else -10), textcoords='offset points',
            ha='center', va='bottom' if val >= 0 else 'top', fontsize=9,
        )

    # Panel B: % significativo
    ax = axes[1]
    bars = ax.bar(metrics, pcts, color=colors, edgecolor='white', width=0.55)
    ax.axhline(0.2, color='crimson', linewidth=1.0, linestyle=':', label='0.2% (umbral citado)')
    ax.set_ylabel('% celdas con p < 0.05', fontsize=9)
    ax.set_title('Significancia espacial\n(bootstrap / regresion por pixel)', fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(0.25, max(pcts) * 1.8))
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(axis='x', labelsize=9)
    ax.legend(fontsize=8, loc='upper right')
    for bar, val in zip(bars, pcts):
        ax.annotate(
            f'{val:.3f}%', xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9,
        )

    fig.suptitle(
        'ALADIN historico: sin tendencia robusta en persistencia de dry spells',
        fontsize=11, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    path = OUT / 'fig33_tendencias_aladin_null_summary.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> {path}')
    return path


if __name__ == '__main__':
    print('Generando figura §3.3 (resumen tendencias)...')
    build_summary_figure()
    print('Listo.')
