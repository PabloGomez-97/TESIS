"""Estadisticos de duracion (Tabla 3) y PDFs empiricas por macrozona y por region.

Criterio i: CR2MET con R = 1 mm y ALADIN con R0 = 5,285 mm. Todas las rachas del anio (no solo mar-nov),
censuradas excluidas. Bins logaritmicos comunes a los tres conjuntos de cada macrozona (26 bordes entre
1 dia y la duracion maxima conjunta, redondeados a enteros y sin repetidos, desplazados -0,5).
f(t) = rachas por bin / ancho del bin (el area bajo la curva es el numero de rachas).

Salidas: resultados_tesis/tabla3_macrozonas.csv, tabla3_regiones.csv y paper_figuras/fig06_pdfs_macrozonas.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'
FIG = ROOT / 'paper_figuras'
CONJ = [('cr2met_hist', 'CR2MET 1980–2014', 'black', '-'),
        ('aladin_hist', 'ALADIN histórico', 'steelblue', '-'),
        ('aladin_ssp585', 'ALADIN SSP5-8.5 2040–2074', 'crimson', '--')]
PDF_BINS = 25


def cargar():
    g = np.load(OUT / 'grilla.npz')
    reg_de_celda = g['region_idx'][g['jj'], g['ii']].astype(np.int64)
    datos = {}
    for nombre, *_ in CONJ:
        z = np.load(OUT / f'rachas_{nombre}.npz')
        datos[nombre] = (reg_de_celda[z['cell']], z['duration'])
    return datos, reg_de_celda


def duraciones(datos, nombre, regiones_idx):
    reg, dur = datos[nombre]
    return dur[np.isin(reg, regiones_idx)]


def estadisticos(d):
    return dict(n_spells=int(d.size), media=float(d.mean()), mediana=float(np.median(d)),
                p90=float(np.percentile(d, 90)), p99=float(np.percentile(d, 99)), maximo=int(d.max()))


def bins_comunes(listas):
    hi = int(max(d.max() for d in listas))
    bordes = np.unique(np.round(np.logspace(0, np.log10(hi + 1), PDF_BINS + 1)))
    return bordes - 0.5


def pdf(d, bins):
    cuentas, e = np.histogram(d, bins=bins)
    lo, hi = e[:-1] + 0.5, e[1:] - 0.5
    return np.sqrt(lo * hi), cuentas / np.diff(e)


def main():
    datos, reg_de_celda = cargar()
    filas_m, filas_r = [], []
    unidades_m = {m: [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == m] for m in MACROZONAS}
    n_celdas = {m: int(np.isin(reg_de_celda, ks).sum()) for m, ks in unidades_m.items()}
    fig, ejes = plt.subplots(2, 2, figsize=(11, 8), sharey=False)
    for ax, m in zip(ejes.ravel(), MACROZONAS):
        ds = {n: duraciones(datos, n, unidades_m[m]) for n, *_ in CONJ}
        bins = bins_comunes(list(ds.values()))
        for nombre, etiqueta, color, ls in CONJ:
            filas_m.append(dict(macrozona=m, conjunto=nombre, celdas=n_celdas[m], **estadisticos(ds[nombre])))
            x, y = pdf(ds[nombre], bins)
            ok = y > 0
            ax.loglog(x[ok], y[ok], color=color, ls=ls, lw=1.8, label=etiqueta)
        ax.set_title(f'{m} ({n_celdas[m]} celdas)', fontsize=11)
        ax.set_xlabel('Duración de la racha t (días)')
        ax.set_ylabel('f(t): rachas por día de ancho de bin')
        ax.grid(True, which='both', alpha=0.25)
    ejes[0, 0].legend(fontsize=8, loc='lower left')
    fig.suptitle('PDFs empíricas de duración de dry spells por macrozona (criterio i, escala log–log)', fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / 'fig06_pdfs_macrozonas.png', dpi=170, facecolor='white')
    for k, n in enumerate(NOMBRES_CORTOS):
        for nombre, *_ in CONJ:
            filas_r.append(dict(region=n, macrozona=MACROZONA_DE[n], conjunto=nombre,
                                celdas=int((reg_de_celda == k).sum()),
                                **estadisticos(duraciones(datos, nombre, [k]))))
    pd.DataFrame(filas_m).to_csv(OUT / 'tabla3_macrozonas.csv', index=False)
    pd.DataFrame(filas_r).to_csv(OUT / 'tabla3_regiones.csv', index=False)
    print(pd.DataFrame(filas_m).round(1).to_string(index=False))


if __name__ == '__main__':
    main()
