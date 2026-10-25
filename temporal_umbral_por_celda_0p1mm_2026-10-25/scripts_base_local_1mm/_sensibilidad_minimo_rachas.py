"""Sensibilidad del sesgo historico (criterio i) al minimo de rachas por celda para calcular media y t99.

El analisis principal exige >= 30 rachas completas por celda y producto (MIN_SPELLS_FOR_STATS en _run_pregunta7_bootstrap.py).
Con pocas rachas el t99 depende de las 2-3 rachas mas largas, y las celdas con pocas rachas estan en el norte arido.
Este script repite el resumen espacial exigiendo >= 30, 50, 100 y 200 rachas en AMBOS productos. Se decidio DESPUES de ver
el resultado con 30: se informa como sensibilidad, no como analisis principal.

Reutiliza las funciones de _run_pregunta7_bootstrap.py (misma mascara, mismo regrillado, mismas reglas de rachas) y la
significancia por celda ya calculada (resultados_tesis/bootstrap_por_celda_C_umbral_local_*.npz): el bootstrap de cada
celda no depende de que otras celdas se incluyan, por lo que los porcentajes de significancia son los de ese mismo analisis
restringidos a las celdas retenidas.

Verificacion interna: las medias y t99 por celda recalculadas deben coincidir con la cache de mapas
(_pregunta7_outputs/metrics_integrado_cache.npz).

Salida: resultados_tesis/sensibilidad_minimo_rachas.csv
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import _run_pregunta7_bootstrap as b
from umbral_local import campo_umbral

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis' / 'sensibilidad_minimo_rachas.csv'
CACHE = ROOT / '_pregunta7_outputs' / 'metrics_integrado_cache.npz'
BOOT = ROOT / 'resultados_tesis'
MINIMOS = [30, 50, 100, 200]


def conteos(groups):
    return np.array([sum(v.size for v in g.values()) for g in groups], dtype=np.int64)


def main():
    pr_aladin = b.open_aladin_historical()
    pr_cr2met = b.regrid_cr2met_to_aladin(b.open_cr2met_historical(), pr_aladin)
    pr_cr2met, pr_aladin = xr.align(b.normalize_daily_time(pr_cr2met), b.normalize_daily_time(pr_aladin), join='inner')
    mask_bool = b.build_chile_mask_on_aladin_grid(
        pr_aladin['lat'].values, pr_aladin['lon'].values, b.load_chile_geometry())
    mask = xr.DataArray(
        mask_bool, coords={'y': pr_aladin['y'], 'x': pr_aladin['x'], 'lat': pr_aladin['lat'], 'lon': pr_aladin['lon']},
        dims=['y', 'x'])
    flat = mask.stack(cell=('y', 'x')).values
    g_cr, _ = b.extract_spell_groups_by_year(pr_cr2met.where(mask), b.TAU_CR2MET_REF, mask)
    g_al, _ = b.extract_spell_groups_by_year(pr_aladin.where(mask), campo_umbral(), mask)
    n_cr, n_al = conteos(g_cr), conteos(g_al)
    ny, nx = mask_bool.shape
    n_min = np.minimum(n_cr, n_al).reshape(ny, nx)

    z = np.load(CACHE)
    medias = {m: (z[f'{m}_cr2'], z[f'{m}_ala']) for m in ('mean', 't99')}
    # verificacion: lo recalculado con >= 30 rachas coincide con la cache del analisis principal
    for m in ('mean', 't99'):
        rec_cr = b.full_period_metric(g_cr, flat, m).reshape(ny, nx)
        rec_al = b.full_period_metric(g_al, flat, m).reshape(ny, nx)
        assert np.array_equal(np.isfinite(rec_cr), np.isfinite(medias[m][0])), f'{m}: celdas validas CR2MET difieren'
        assert np.allclose(rec_cr, medias[m][0], equal_nan=True, atol=1e-3), f'{m}: CR2MET difiere de la cache'
        assert np.allclose(rec_al, medias[m][1], equal_nan=True, atol=1e-3), f'{m}: ALADIN difiere de la cache'

    boot = {m: np.load(BOOT / f'bootstrap_por_celda_C_umbral_local_{m}.npz') for m in ('mean', 't99')}
    lat = z['lat']
    comunes = z['chile_mask'] & np.isfinite(medias['mean'][0]) & np.isfinite(medias['mean'][1]) \
        & np.isfinite(medias['t99'][0]) & np.isfinite(medias['t99'][1])
    filas = []
    for k in MINIMOS:
        sel = comunes & (n_min >= k)
        fila = {'minimo_rachas': k, 'celdas': int(sel.sum()), 'celdas_excluidas_vs_30': int(comunes.sum() - sel.sum()),
                'lat_media_excluidas': float(lat[comunes & ~sel].mean()) if (comunes & ~sel).any() else np.nan}
        for m, nom in (('mean', 'media'), ('t99', 't99')):
            d = (medias[m][1] - medias[m][0])[sel]
            fila[f'delta_{nom}_promedio'] = float(d.mean())
            fila[f'delta_{nom}_mediana'] = float(np.median(d))
            v = boot[m]['valida'] & sel
            fila[f'pct_{nom}_ALADIN_menor_sig'] = float(100 * (boot[m]['mas_corto'] & v).sum() / v.sum())
            fila[f'pct_{nom}_ALADIN_mayor_sig'] = float(100 * (boot[m]['mas_largo'] & v).sum() / v.sum())
        filas.append(fila)
    df = pd.DataFrame(filas)
    OUT.parent.mkdir(exist_ok=True)
    df.round(3).to_csv(OUT, index=False)
    print(df.round(2).T.to_string())


if __name__ == '__main__':
    main()
