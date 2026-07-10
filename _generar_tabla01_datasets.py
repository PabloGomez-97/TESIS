"""Genera Tabla 1: resumen de datasets para el paper."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'paper_figuras'
OUT_DIR.mkdir(exist_ok=True)

TABLE_ROWS = [
    {
        'Dataset': 'CR2MET v2.5',
        'Tipo': 'Producto gridded de referencia (observacional)',
        'Dominio': 'Chile continental (máscara Natural Earth)',
        'Grilla / resolución': 'Lat/lon regular 0.05° (~5 km); regrillado lineal a grilla ALADIN para comparaciones',
        'Periodo usado': '1980–2014 (histórico); 1980–2000 y 2001–2021 (análisis P1/P2, solo CR2MET)',
        'Variable(s)': 'Precipitación diaria (pr)',
        'Unidades / notas': 'mm/día; umbral día húmedo R = 1 mm',
    },
    {
        'Dataset': 'ALADIN CHP12 (histórico)',
        'Tipo': 'Modelo climático regional (RCM); CNRM-ALADIN64 v1',
        'Dominio': 'Chile continental; 3 587 celdas válidas sobre grilla curvilínea (y×x)',
        'Grilla / resolución': 'Grilla nativa curvilínea ~12 km (255×68 celdas dominio CHP12)',
        'Periodo usado': '1980–2014 (35 años; emparejado con futuro)',
        'Variable(s)': 'Precipitación diaria (pr)',
        'Unidades / notas': 'kg m⁻² s⁻¹ × 86 400 → mm/día; umbral calibrado R₀ = 5.285 mm (criterio i)',
    },
    {
        'Dataset': 'ALADIN CHP12 (SSP5-8.5)',
        'Tipo': 'Proyección RCM; forzado por CNRM-ESM2-1 (r1i1p1f2)',
        'Dominio': 'Mismo dominio y grilla que ALADIN histórico',
        'Grilla / resolución': 'Idéntica a ALADIN histórico',
        'Periodo usado': '2040–2074 (35 años; comparado con hist 1980–2014)',
        'Variable(s)': 'Precipitación diaria (pr)',
        'Unidades / notas': 'mm/día; mismo R₀ = 5.285 mm; spells con inicio mar–nov para RR',
    },
]

# Versión en inglés (para paper)
TABLE_ROWS_EN = [
    {
        'Dataset': 'CR2MET v2.5',
        'Type': 'Reference gridded product (observational)',
        'Domain': 'Mainland Chile (Natural Earth mask)',
        'Grid / resolution': 'Regular lat/lon 0.05° (~5 km); linearly regridded to ALADIN grid for comparisons',
        'Period used': '1980–2014 (historical); 1980–2000 and 2001–2021 (P1/P2 analysis, CR2MET only)',
        'Variable(s)': 'Daily precipitation (pr)',
        'Units / notes': 'mm day⁻¹; wet-day threshold R = 1 mm',
    },
    {
        'Dataset': 'ALADIN CHP12 (historical)',
        'Type': 'Regional climate model (RCM); CNRM-ALADIN64 v1',
        'Domain': 'Mainland Chile; 3,587 valid cells on curvilinear grid (y×x)',
        'Grid / resolution': 'Native curvilinear grid ~12 km (255×68 cells, CHP12 domain)',
        'Period used': '1980–2014 (35 years; paired with future)',
        'Variable(s)': 'Daily precipitation (pr)',
        'Units / notes': 'kg m⁻² s⁻¹ × 86 400 → mm day⁻¹; calibrated threshold R₀ = 5.285 mm (criterion i)',
    },
    {
        'Dataset': 'ALADIN CHP12 (SSP5-8.5)',
        'Type': 'RCM projection; driven by CNRM-ESM2-1 (r1i1p1f2)',
        'Domain': 'Same domain and grid as historical ALADIN',
        'Grid / resolution': 'Identical to historical ALADIN',
        'Period used': '2040–2074 (35 years; vs. hist 1980–2014)',
        'Variable(s)': 'Daily precipitation (pr)',
        'Units / notes': 'mm day⁻¹; same R₀ = 5.285 mm; spells starting Mar–Nov for risk ratios',
    },
]


def save_csv(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  -> {path}')


def _wrap(text, width=42):
    import textwrap
    return '\n'.join(textwrap.wrap(str(text), width=width)) if len(str(text)) > width else str(text)


def render_table_png(df, title, path, fontsize=7.2):
    fig, ax = plt.subplots(figsize=(16, 5.2))
    ax.axis('off')

    col_labels = list(df.columns)
    widths_by_col = [14, 16, 18, 24, 22, 14, 24]
    cell_text = [
        [_wrap(v, w) for v, w in zip(row, widths_by_col)]
        for row in df.values.tolist()
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='center',
        cellLoc='left',
        colLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 2.1)

    # Header style
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].get_text().set_color('white')
        table[(0, j)].get_text().set_weight('bold')

    # Alternate row colors
    for i in range(1, len(cell_text) + 1):
        color = '#F8F9FA' if i % 2 else '#FFFFFF'
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_edgecolor('#DEE2E6')

    # Column widths (relative)
    widths = [0.10, 0.12, 0.12, 0.18, 0.17, 0.10, 0.21]
    for j, w in enumerate(widths):
        for i in range(len(cell_text) + 1):
            table[(i, j)].set_width(w)
            table[(i, j)].get_text().set_wrap(True)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=14)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> {path}')


def main():
    print('Generando Tabla 1 — datasets...')
    df_es = pd.DataFrame(TABLE_ROWS)
    df_en = pd.DataFrame(TABLE_ROWS_EN)

    save_csv(df_es, OUT_DIR / 'tabla01_datasets.csv')
    save_csv(df_en, OUT_DIR / 'tabla01_datasets_en.csv')

    render_table_png(
        df_es,
        'Tabla 1. Resumen de datasets, resolución, periodos y variables',
        OUT_DIR / 'fig_tabla01_datasets.png',
    )
    render_table_png(
        df_en,
        'Table 1. Summary of datasets, resolution, periods, and variables',
        OUT_DIR / 'fig_tabla01_datasets_en.png',
    )

    # Copia para presentación
    pres = ROOT / 'presentacion_tesis' / 'figuras'
    pres.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(OUT_DIR / 'fig_tabla01_datasets.png', pres / 'fig_tabla01_datasets.png')
    print(f'  -> {pres / "fig_tabla01_datasets.png"}')
    print('Listo.')


if __name__ == '__main__':
    main()
