"""Figura S1: sensibilidad de las PDFs historicas por cuenca a la eleccion del umbral (§2.7).

Cuatro series 1980-2014 por cuenca: CR2MET 0.1 mm, ALADIN 0.1 mm (criterio ii), ALADIN R0 global
(criterio i) y ALADIN R0 local (criterio iii). Bins, mascaras y reglas de spells identicos a la
Figura 6 (_generar_pregunta9_pdf_loglog_cuencas.py).
"""
import warnings
warnings.filterwarnings('ignore')

import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import _generar_pregunta9_pdf_loglog_cuencas as f6

OUT = f6.ROOT / '_pregunta9_calibracion_outputs'
CACHE = OUT / 'spell_durations_sensibilidad_4cuencas.pkl'
RESUMEN = OUT / 'resumen_sensibilidad_umbral_4cuencas.csv'
FIG = f6.PAPER / 'figS1_sensibilidad_umbral_cuencas.png'

SERIES_STYLE = {
    'CR2MET 0.1 mm':       {'color': 'black',       'ls': '-',  'lw': 2.2, 'marker': 'o', 'ms': 3.5},
    'ALADIN 0.1 mm':       {'color': 'crimson',     'ls': '--', 'lw': 1.8, 'marker': 's', 'ms': 3.5},
    'ALADIN R0 global':  {'color': 'steelblue',   'ls': '-',  'lw': 1.8, 'marker': '^', 'ms': 3.5},
    'ALADIN R0 local':   {'color': 'forestgreen', 'ls': '-.', 'lw': 1.8, 'marker': 'D', 'ms': 3.0},
}
PANEL_LABELS = ['(a) Loa', '(b) Maipo', '(c) Maule', '(d) Biobío']


def compute_durations():
    if CACHE.exists():
        print(f'Cargando duraciones desde cache: {CACHE}')
        with CACHE.open('rb') as f:
            return pickle.load(f)

    print('1/3: Cargando datos historicos...')
    pr_ala = f6.open_aladin_period(f6.HIST_START, f6.HIST_END)
    pr_cr2 = f6.open_cr2met_period(f6.HIST_START, f6.HIST_END).interp(
        lat=pr_ala['lat'], lon=pr_ala['lon'], method='linear',
    )
    lat2d, lon2d = pr_ala['lat'].values, pr_ala['lon'].values
    chile_bool = f6.build_chile_mask_on_aladin_grid(lat2d, lon2d, f6.load_chile_geometry())
    coords = {'y': pr_ala['y'], 'x': pr_ala['x'], 'lat': pr_ala['lat'], 'lon': pr_ala['lon']}
    chile_mask = xr.DataArray(chile_bool, coords=coords, dims=['y', 'x'])
    basin_masks = {
        name: xr.DataArray(f6.build_basin_mask(lat2d, lon2d, chile_bool, spec['bounds']),
                           coords=coords, dims=['y', 'x'])
        for name, spec in f6.BASIN_SPECS.items()
    }

    # R0 local es pixel a pixel: basta calcularlo sobre la union de las cuencas.
    union = xr.concat(list(basin_masks.values()), dim='b').any('b')
    print(f'2/3: R0 local por biseccion ({int(union.sum())} pixeles)...')
    pr_cr2_m, pr_ala_m = pr_cr2.where(chile_mask), pr_ala.where(chile_mask)
    r0_local = f6.pixelwise_local_aladin_threshold(pr_cr2_m.where(union), pr_ala_m.where(union),
                                                   f6.TAU_CR2MET_REF, union)
    r0_local = r0_local.assign_coords(lat=pr_ala['lat'], lon=pr_ala['lon'])

    print('3/3: Extrayendo spells por cuenca...')
    durs = {}
    for name, bm in basin_masks.items():
        print(f'  {name}...')
        durs[name] = {
            'CR2MET 0.1 mm': f6.extract_basin_spell_records(pr_cr2_m, f6.TAU_CR2MET_REF, bm),
            'ALADIN 0.1 mm': f6.extract_basin_spell_records(pr_ala_m, f6.TAU_CR2MET_REF, bm),
            'ALADIN R0 global': f6.extract_basin_spell_records(pr_ala_m, 3.668, bm),
            'ALADIN R0 local': f6.extract_basin_spell_records(pr_ala_m, r0_local, bm),
        }
    OUT.mkdir(exist_ok=True)
    with CACHE.open('wb') as f:
        pickle.dump(durs, f)
    return durs


def summary_table(durs):
    rows = []
    for basin, series in durs.items():
        for label, d in series.items():
            d = np.asarray(d, dtype=float)
            rows.append({
                'cuenca': basin, 'serie': label, 'n_spells': int(d.size),
                'media_dias': d.mean(), 'mediana_dias': np.median(d),
                'p90_dias': np.percentile(d, 90), 'p99_dias': np.percentile(d, 99),
                'max_dias': d.max(),
            })
    return pd.DataFrame(rows)


def plot_panel(durs):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (basin, series), panel in zip(axes.ravel(), durs.items(), PANEL_LABELS):
        bins = f6.common_log_bins(series)
        for label, style in SERIES_STYLE.items():
            centers, heights, n = f6.pdf_from_durations(series[label], bins)
            if centers is None:
                continue
            ax.plot(centers, np.where(heights > 0, heights, np.nan),
                    label=f'{label} (n={n:,})', **style)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(panel, fontweight='bold', loc='left')
        ax.set_xlabel('Duración de racha seca (días)')
        ax.set_ylabel('f(t)  [integral = n_spells]')
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=8.5)
    fig.suptitle('Sensibilidad de las PDFs históricas (1980–2014) a la elección del umbral',
                 fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    durs = compute_durations()
    df = summary_table(durs)
    df.round(2).to_csv(RESUMEN, index=False)
    print(df.round(1).to_string(index=False))
    plot_panel(durs)
    print(f'-> {FIG}\n-> {RESUMEN}')


if __name__ == '__main__':
    main()
