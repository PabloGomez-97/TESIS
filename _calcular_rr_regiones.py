"""Risk ratios RR(D) = N_fut(t >= D) / N_hist(t >= D) por region, macrozona y dominio completo.

Reglas (iguales al paper salvo lo indicado):
- rachas de ALADIN con R0 = 5,285 mm/dia, censuradas excluidas, inicio entre marzo y noviembre;
- IC95 por bootstrap de anios, con remuestreos INDEPENDIENTES del historico y del futuro (1000 replicas);
- NUEVO: regla explicita para N_hist = 0. Se informa la fraccion de replicas con N_hist* = 0, y el RR solo se
  declara 'estimable' si N_hist >= N_MIN y ninguna replica tiene N_hist* = 0. Ademas se informa el exceso de
  conteos dN = N_fut - N_hist con su IC95, que no tiene denominador.
Uso: python _calcular_rr_regiones.py  (genera principal, ssp370 y 1 mm)
"""
import numpy as np
import pandas as pd
from pathlib import Path

from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'
DS = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
MES_MIN, MES_MAX = 3, 11
B = 1000
SEED = 42
N_MIN = 30
ALPHA = 5.0

ESCENARIOS = {
    'ssp585':     ('aladin_hist',     'aladin_ssp585',     (1980, 2014), (2040, 2074)),
    'ssp370':     ('aladin_hist',     'aladin_ssp370',     (1980, 2014), (2040, 2074)),
    'ssp585_1mm': ('aladin_hist_1mm', 'aladin_ssp585_1mm', (1980, 2014), (2040, 2074)),
    'ssp585_Rfrec': ('aladin_hist', 'aladin_ssp585_Rfrec', (1980, 2014), (2040, 2074)),
    'ssp370_Rfrec': ('aladin_hist', 'aladin_ssp370_Rfrec', (1980, 2014), (2040, 2074)),
    'ssp585_RfrecReg': ('aladin_hist', 'aladin_ssp585_RfrecReg', (1980, 2014), (2040, 2074)),
    'ssp370_RfrecReg': ('aladin_hist', 'aladin_ssp370_RfrecReg', (1980, 2014), (2040, 2074)),
}


def conteos_por_anio(nombre, reg_de_celda, y0, y1, D):
    """Devuelve matriz (14, n_anios): rachas con duracion >= D e inicio mar-nov, por region y anio."""
    z = np.load(OUT / f'rachas_{nombre}.npz')
    ok = (z['duration'] >= D) & (z['start_month'] >= MES_MIN) & (z['start_month'] <= MES_MAX)
    reg = reg_de_celda[z['cell'][ok]]
    ok2 = reg >= 0
    n_anios = y1 - y0 + 1
    clave = reg[ok2].astype(np.int64) * n_anios + (z['start_year'][ok].astype(np.int64)[ok2] - y0)
    return np.bincount(clave, minlength=len(NOMBRES_CORTOS) * n_anios).reshape(len(NOMBRES_CORTOS), n_anios)


def unidades(mat):
    """Agrega la matriz (14, anios) en regiones, macrozonas y dominio (14 regiones)."""
    out = {('region', n): mat[k] for k, n in enumerate(NOMBRES_CORTOS)}
    for m in MACROZONAS:
        filas = [k for k, n in enumerate(NOMBRES_CORTOS) if MACROZONA_DE[n] == m]
        out[('macrozona', m)] = mat[filas].sum(axis=0)
    out[('dominio', '14 regiones')] = mat.sum(axis=0)
    return out


def bootstrap(ch, cf, rng):
    """ch, cf: conteos por anio (hist, fut). Remuestreos independientes de anios."""
    ih = rng.integers(0, ch.size, size=(B, ch.size))
    jf = rng.integers(0, cf.size, size=(B, cf.size))
    nh, nf = ch[ih].sum(axis=1), cf[jf].sum(axis=1)
    return nh, nf


def correr(tag, sufijo=''):
    hist, fut, (h0, h1), (f0, f1) = ESCENARIOS[tag]
    g = np.load(OUT / 'grilla.npz')
    reg_de_celda = g['region_idx'][g['jj'], g['ii']].astype(np.int64)
    filas = []
    for D in DS:
        uh = unidades(conteos_por_anio(hist, reg_de_celda, h0, h1, D))
        uf = unidades(conteos_por_anio(fut, reg_de_celda, f0, f1, D))
        for k, (clave, ch) in enumerate(uh.items()):
            cf = uf[clave]
            rng = np.random.default_rng(SEED + 1000 * D + k)
            nh_b, nf_b = bootstrap(ch, cf, rng)
            Nh, Nf = int(ch.sum()), int(cf.sum())
            frac0 = float((nh_b == 0).mean())
            valido = nh_b > 0
            rr = Nf / Nh if Nh > 0 else np.nan
            if valido.any():
                lo, hi = np.percentile(nf_b[valido] / nh_b[valido], [ALPHA / 2, 100 - ALPHA / 2])
            else:
                lo = hi = np.nan
            dlo, dhi = np.percentile(nf_b - nh_b, [ALPHA / 2, 100 - ALPHA / 2])
            estimable = (Nh >= N_MIN) and frac0 == 0.0
            filas.append(dict(
                nivel=clave[0], unidad=clave[1], D=D, N_hist=Nh, N_fut=Nf, RR=rr,
                IC_low=lo if estimable else np.nan, IC_high=hi if estimable else np.nan,
                frac_replicas_Nhist0=frac0, estimable=estimable,
                dN=Nf - Nh, dN_low=dlo, dN_high=dhi,
                aumento_sig=bool(estimable and lo > 1), descenso_sig=bool(estimable and hi < 1),
            ))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / f'rr_{tag}{sufijo}.csv', index=False)
    return df


if __name__ == '__main__':
    for tag in ESCENARIOS:
        df = correr(tag)
        print(f'--- {tag}: RR(D=20) por unidad ---')
        d20 = df[df.D == 20][['nivel', 'unidad', 'N_hist', 'N_fut', 'RR', 'IC_low', 'IC_high', 'estimable']]
        print(d20.round(2).to_string(index=False))
