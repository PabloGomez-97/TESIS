"""Figura 8: precipitacion anual media y duracion media de las rachas secas en ALADIN, pasado y futuro.

Fila superior: precipitacion anual media (mm/anio) en 1980-2014 y 2040-2074 (SSP5-8.5) y cambio relativo (%).
Fila inferior: duracion media de las rachas secas (dias, criterio i, R0 = 5,285 mm, censuradas excluidas) en ambos
periodos y su diferencia. Las estadisticas de duracion solo se calculan en celdas con al menos 30 rachas.

Entradas: pr1/ (ALADIN), resultados_tesis/rachas_aladin_hist.npz y rachas_aladin_ssp585.npz, resultados_tesis/grilla.npz
Salidas: paper_figuras/fig08_mapas_precip_duracion.png, resultados_tesis/fig08_resumen_por_zona.csv
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE
from _extraer_rachas_dominio import open_aladin, a_celdas, HIST, FUT

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'
MIN_RACHAS = 30


def duracion_media(nombre, ncel):
    z = np.load(OUT / f'rachas_{nombre}.npz')
    n = np.bincount(z['cell'], minlength=ncel)
    s = np.bincount(z['cell'], weights=z['duration'], minlength=ncel)
    return np.where(n >= MIN_RACHAS, s / np.maximum(n, 1), np.nan)


def main():
    g = np.load(OUT / 'grilla.npz')
    jj, ii, mask, idx = g['jj'], g['ii'], g['mask'], g['region_idx']
    nc = jj.size
    p_h = a_celdas(open_aladin(HIST, None), jj, ii).mean(axis=0) * 365.25
    p_f = a_celdas(open_aladin(FUT, 'ssp585'), jj, ii).mean(axis=0) * 365.25
    dp = (p_f / p_h - 1) * 100
    d_h, d_f = duracion_media('aladin_hist', nc), duracion_media('aladin_ssp585', nc)
    dd = d_f - d_h

    def mapa(v):
        m = np.full(mask.shape, np.nan)
        m[jj, ii] = v
        return m

    lat, lon = g['lat'], g['lon']
    paneles = [
        (p_h, 'Precipitación anual\n1980–2014 (mm/año)', 'Blues', (0, 4500)),
        (p_f, 'Precipitación anual\n2040–2074 SSP5-8.5 (mm/año)', 'Blues', (0, 4500)),
        (dp, 'Cambio relativo (%)', 'RdBu', (-40, 40)),
        (d_h, 'Duración media de rachas\n1980–2014 (días)', 'YlOrBr', (0, 200)),
        (d_f, 'Duración media de rachas\n2040–2074 SSP5-8.5 (días)', 'YlOrBr', (0, 200)),
        (dd, 'Diferencia futuro − pasado (días)', 'RdBu_r', (-40, 40)),
    ]
    fig, ejes = plt.subplots(2, 3, figsize=(11.5, 12.5))
    for ax, (v, t, cm, (a, b)) in zip(ejes.ravel(), paneles):
        im = ax.pcolormesh(lon, lat, np.ma.masked_invalid(mapa(v)), cmap=cm, vmin=a, vmax=b, shading='auto')
        ax.set_xlim(-76, -66); ax.set_ylim(-46.5, -17)
        ax.set_aspect(1.1); ax.set_title(t, fontsize=9)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, extend='both')
    fig.suptitle('ALADIN CHP12, criterio i: precipitación anual y duración media de las rachas secas', fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / 'paper_figuras' / 'fig08_mapas_precip_duracion.png', dpi=170, facecolor='white')

    reg = idx[jj, ii]
    filas = []
    for z in MACROZONAS + ['Todo el dominio (14 regiones)']:
        if z.startswith('Todo'):
            sel = reg >= 0
        else:
            sel = np.isin(reg, [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == z])
        ok = sel & np.isfinite(d_h) & np.isfinite(d_f)
        filas.append(dict(zona=z, celdas=int(sel.sum()), precip_hist_mm=float(p_h[sel].mean()), precip_fut_mm=float(p_f[sel].mean()),
                          cambio_precip_pct=float((p_f[sel].mean() / p_h[sel].mean() - 1) * 100),
                          celdas_duracion_validas=int(ok.sum()), dur_media_hist=float(d_h[ok].mean()),
                          dur_media_fut=float(d_f[ok].mean()), diferencia_dias=float(dd[ok].mean())))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / 'fig08_resumen_por_zona.csv', index=False)
    print(df.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
