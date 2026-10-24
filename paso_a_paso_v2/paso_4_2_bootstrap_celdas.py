"""Paso 4.2 — Bootstrap por celda: ¿la diferencia ALADIN - CR2MET (duracion media y p99) es distinguible del azar?

Diseno (1980-2014, ALADIN criterio i):
  - Remuestreo de ANIOS: en cada replica se sortean 35 anios calendario con reemplazo, de forma INDEPENDIENTE para CR2MET y para ALADIN. El mismo sorteo se usa en todas las celdas
    (conserva la coherencia espacial de cada replica y permite, mas adelante, intervalos para promedios regionales).
  - En cada celda se juntan las duraciones de las rachas completas cuyo anio de inicio salio sorteado (con su multiplicidad) y se recalculan la media y el p99
    (numpy.percentile lineal sobre la lista expandida; implementado con pesos y comprobado contra la lista expandida).
  - Delta* = estadistico*(ALADIN) - estadistico*(CR2MET). IC95 = percentiles 2,5 y 97,5 de los Delta*; significativo si el IC no incluye 0.
  - 1 000 replicas, semilla 42. Sin minimo de rachas: se cuentan y se informan las replicas en que una celda queda sin rachas en algun producto (no se descartan en silencio).
Entradas: salidas/paso_3_1_rachas_completas_{cr2met_hist,aladin_hist_i}.npz, salidas/paso_4_1_estadisticos_por_celda.csv
Salidas: salidas/paso_4_2_bootstrap_por_celda.csv, salidas/paso_4_2_deltas_replicas.npz (Delta* por replica y celda), salidas/paso_4_2_resumen.csv, salidas/paso_4_2_chequeos.txt,
         figuras/paso_4_2_mapas_significancia.png
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
S = ROOT / 'salidas'
R, SEED, ANIOS = 1000, 42, np.arange(1980, 2015)
NY = ANIOS.size
est = pd.read_csv(S / 'paso_4_1_estadisticos_por_celda.csv')
nombres = list(np.load(S / 'paso_0_1_mascara_dominio.npz')['nombres'])
nc = len(est)
rid = np.array([nombres.index(r) for r in est['region']])
chk = []

rng = np.random.default_rng(SEED)
def sorteo():
    idx = rng.integers(0, NY, size=(R, NY))
    W = np.zeros((R, NY), dtype=np.int32)
    for r in range(R):
        W[r] = np.bincount(idx[r], minlength=NY)
    return W
W_c, W_a = sorteo(), sorteo()                      # independientes entre productos
chk.append(f'{R} replicas, semilla {SEED}; sorteos independientes para CR2MET y ALADIN (mismo sorteo en todas las celdas); anios 1980-2014 ({NY})')
chk.append(f'Control del sorteo: peso total por replica = {np.unique(W_c.sum(1))} y {np.unique(W_a.sum(1))}; anios no sorteados por replica (promedio): CR2MET {np.mean((W_c == 0).sum(1)):.1f}, ALADIN {np.mean((W_a == 0).sum(1)):.1f} (esperado ≈ {NY * (1 - 1 / NY) ** NY:.1f})')


def cargar(nom):
    z = np.load(S / f'paso_3_1_rachas_completas_{nom}.npz')
    cel, dur, anio = z['celda'], z['duracion'].astype(np.float64), z['anio_inicio'].astype(int) - 1980
    orden = np.lexsort((dur, cel))                  # por celda y, dentro, por duracion creciente
    cel, dur, anio = cel[orden], dur[orden], anio[orden]
    lim = np.r_[0, np.cumsum(np.bincount(cel, minlength=nc))]
    return [(dur[lim[j]:lim[j + 1]], anio[lim[j]:lim[j + 1]]) for j in range(nc)]


def stats_boot(d, yi, W):
    """media y p99 de la lista remuestreada (R replicas) para una celda. d ordenada de menor a mayor; yi = anio de cada racha."""
    ws = W[:, yi].astype(np.float64)                # (R, n)
    N = ws.sum(1)
    ok = N > 0
    media = np.where(ok, (ws * d).sum(1) / np.where(ok, N, 1), np.nan)
    cw = np.cumsum(ws, axis=1)
    pos = 0.99 * (N - 1)
    lo = np.floor(pos); hi = np.minimum(lo + 1, N - 1); fr = pos - lo
    x_lo = d[(cw > lo[:, None]).argmax(1)]; x_hi = d[(cw > hi[:, None]).argmax(1)]
    p99 = np.where(ok, x_lo + fr * (x_hi - x_lo), np.nan)
    return media, p99, ok


rc, ra = cargar('cr2met_hist'), cargar('aladin_hist_i')
# comprobacion de la implementacion con pesos frente a la lista expandida (celdas y replicas al azar)
rr = np.random.default_rng(5)
maxdif = 0.0
for j in rr.choice(nc, 6, replace=False):
    for W, datos in ((W_c, rc), (W_a, ra)):
        d, yi = datos[j]
        m, p, _ = stats_boot(d, yi, W[:3])
        for r in range(3):
            x = np.repeat(d, W[r, yi].astype(int))
            maxdif = max(maxdif, abs(m[r] - x.mean()), abs(p[r] - np.percentile(x, 99)))
chk.append(f'Comprobacion contra la lista expandida (6 celdas x 2 productos x 3 replicas): diferencia maxima en media y p99 = {maxdif:.2e} d')

t0 = time.time()
dM = np.full((R, nc), np.nan, dtype=np.float32); dP = np.full((R, nc), np.nan, dtype=np.float32)
sin_rachas = 0
for j in range(nc):
    mc, pc, okc = stats_boot(*rc[j], W_c)
    ma, pa, oka = stats_boot(*ra[j], W_a)
    sin_rachas += int((~(okc & oka)).sum())
    dM[:, j], dP[:, j] = ma - mc, pa - pc
chk.append(f'Bootstrap hecho en {time.time() - t0:.0f} s; replicas-celda sin rachas en algun producto: {sin_rachas} de {R * nc}; Delta* con NaN: media {int(np.isnan(dM).sum())}, p99 {int(np.isnan(dP).sum())}')
np.savez_compressed(S / 'paso_4_2_deltas_replicas.npz', delta_media=dM, delta_p99=dP, W_cr2met=W_c, W_aladin=W_a, semilla=SEED)

out = est[['celda', 'region', 'lat', 'lon', 'delta_media', 'delta_p99']].copy()
for tag, D, col in (('media', dM, 'delta_media'), ('p99', dP, 'delta_p99')):
    lo, hi = np.nanpercentile(D, 2.5, axis=0), np.nanpercentile(D, 97.5, axis=0)
    out[f'ic_inf_{tag}'], out[f'ic_sup_{tag}'] = lo, hi
    out[f'mediana_boot_{tag}'] = np.nanmedian(D, axis=0)
    out[f'cat_{tag}'] = np.where(hi < 0, 'ALADIN mas corto', np.where(lo > 0, 'ALADIN mas largo', 'sin diferencia clara'))
    chk.append(f'{tag}: el Delta observado cae dentro de su IC95 en {int(((out[col] >= lo) & (out[col] <= hi)).sum())} de {nc} celdas; mediana de (mediana de Delta* - Delta observado) = {np.median(out[f"mediana_boot_{tag}"] - out[col]):+.3f} d')
out.to_csv(S / 'paso_4_2_bootstrap_por_celda.csv', index=False)

filas = []
for etq, k in [('Dominio', np.ones(nc, bool))] + [(n, rid == r) for r, n in enumerate(nombres)]:
    f = dict(zona=etq, celdas=int(k.sum()))
    for tag in ('media', 'p99'):
        c = out.loc[k, f'cat_{tag}']
        f[f'{tag}_aladin_mas_corto_pct'] = 100 * (c == 'ALADIN mas corto').mean()
        f[f'{tag}_aladin_mas_largo_pct'] = 100 * (c == 'ALADIN mas largo').mean()
        f[f'{tag}_sin_diferencia_clara_pct'] = 100 * (c == 'sin diferencia clara').mean()
    filas.append(f)
res = pd.DataFrame(filas)
res.to_csv(S / 'paso_4_2_resumen.csv', index=False)
for f in filas:
    chk.append(f'{f["zona"]} ({f["celdas"]} celdas): MEDIA: ALADIN mas corto {f["media_aladin_mas_corto_pct"]:.1f} %, mas largo {f["media_aladin_mas_largo_pct"]:.1f} %, sin diferencia clara {f["media_sin_diferencia_clara_pct"]:.1f} % | '
               f'P99: mas corto {f["p99_aladin_mas_corto_pct"]:.1f} %, mas largo {f["p99_aladin_mas_largo_pct"]:.1f} %, sin diferencia clara {f["p99_sin_diferencia_clara_pct"]:.1f} %')
anch = (out['ic_sup_p99'] - out['ic_inf_p99'])
chk.append(f'Ancho del IC95 del Delta (dias), mediana entre celdas: media {np.median(out["ic_sup_media"] - out["ic_inf_media"]):.2f}; p99 {np.median(anch):.1f} (p10 {np.percentile(anch, 10):.1f}, p90 {np.percentile(anch, 90):.1f})')
(S / 'paso_4_2_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))

cm = ListedColormap(['#2b6cb0', '#d9d9d9', '#c0392b'])
fig, axs = plt.subplots(1, 2, figsize=(12, 8.5))
for ax, tag, tit in zip(axs, ('media', 'p99'), ('Duración media', 'p99')):
    v = out[f'cat_{tag}'].map({'ALADIN mas corto': 0, 'sin diferencia clara': 1, 'ALADIN mas largo': 2}).values
    ax.scatter(out['lon'], out['lat'], c=v, cmap=cm, vmin=0, vmax=2, s=24, marker='s')
    ax.set_aspect(1 / np.cos(np.deg2rad(36))); ax.set_xlabel('longitud'); ax.set_ylabel('latitud')
    p = [(out[f'cat_{tag}'] == c).mean() * 100 for c in ('ALADIN mas corto', 'sin diferencia clara', 'ALADIN mas largo')]
    ax.set_title(f'{tit}\nALADIN más corto {p[0]:.0f} % | sin diferencia clara {p[1]:.0f} % | ALADIN más largo {p[2]:.0f} %', fontsize=10)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in ('#2b6cb0', '#d9d9d9', '#c0392b')], labels=['ALADIN significativamente más corto', 'sin diferencia clara (IC95 incluye 0)', 'ALADIN significativamente más largo'], loc='lower center', ncol=3, fontsize=9, frameon=False)
fig.suptitle('¿Es real la diferencia ALADIN − CR2MET en cada celda? Bootstrap por años (1 000 réplicas, IC95), 1980–2014', fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 0.97))
fig.savefig(ROOT / 'figuras' / 'paso_4_2_mapas_significancia.png', dpi=140)
