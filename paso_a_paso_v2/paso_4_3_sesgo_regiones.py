"""Paso 4.3 — Resumen del sesgo historico de ALADIN (criterio i) por region y para el dominio, con intervalos del bootstrap por anios.

Dos medidas distintas del mismo defecto (nunca mezclarlas):
  A) Delta por celda promediado en la region (dias): promedio de (ALADIN - CR2MET) calculado celda por celda (pasos 4.1-4.2). Cada celda pesa igual.
  B) Sesgo relativo del pool regional (%): se juntan todas las rachas completas de las celdas de la region en una sola lista por producto y se compara
     (ALADIN - CR2MET) / CR2MET para la media, la mediana y el p99. Cada racha pesa igual.
IC95 de ambas medidas con los MISMOS remuestreos de anios del paso 4.2 (salidas/paso_4_2_deltas_replicas.npz; sorteos independientes entre productos, mismo sorteo en todas las celdas).
Para el pool se usa el histograma de duraciones por anio (35 x duracion maxima): con los pesos de una replica se obtiene la distribucion remuestreada y su percentil exacto (lineal).
Entradas: salidas/paso_3_1_rachas_completas_{cr2met_hist,aladin_hist_i}.npz, salidas/paso_4_1_estadisticos_por_celda.csv, salidas/paso_4_2_deltas_replicas.npz
Salidas: salidas/paso_4_3_sesgo_por_region.csv, salidas/paso_4_3_chequeos.txt, figuras/paso_4_3_sesgo_regiones.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
S = ROOT / 'salidas'
est = pd.read_csv(S / 'paso_4_1_estadisticos_por_celda.csv')
nombres = list(np.load(S / 'paso_0_1_mascara_dominio.npz')['nombres'])
rid = np.array([nombres.index(r) for r in est['region']])
nc = len(est)
rep = np.load(S / 'paso_4_2_deltas_replicas.npz')
dM, dP, W_c, W_a = rep['delta_media'], rep['delta_p99'], rep['W_cr2met'], rep['W_aladin']
R = dM.shape[0]
chk = [f'{R} replicas (las del paso 4.2, semilla {int(rep["semilla"])}); celdas: {nc}']
zonas = [('Dominio', np.ones(nc, bool))] + [(n, rid == r) for r, n in enumerate(nombres)]

# histogramas de duracion por anio para cada producto y zona
def cargar(nom):
    z = np.load(S / f'paso_3_1_rachas_completas_{nom}.npz')
    return z['celda'], z['duracion'].astype(int), z['anio_inicio'].astype(int) - 1980
cc, dc, yc = cargar('cr2met_hist'); ca, da, ya = cargar('aladin_hist_i')
DMAX = max(dc.max(), da.max()) + 1
def hist(cel, dur, anio, k):
    m = k[cel]
    H = np.zeros((35, DMAX), dtype=np.float64)
    np.add.at(H, (anio[m], dur[m]), 1)
    return H


def estadisticos(counts):
    """counts: (R, DMAX) numero de rachas de cada duracion en cada replica -> media, mediana, p99 (numpy.percentile lineal), R vectores."""
    d = np.arange(DMAX)
    n = counts.sum(1)
    media = (counts * d).sum(1) / n
    cw = np.cumsum(counts, axis=1)
    def pct(q):
        pos = q * (n - 1); lo = np.floor(pos); hi = np.minimum(lo + 1, n - 1); fr = pos - lo
        x_lo = (cw > lo[:, None]).argmax(1); x_hi = (cw > hi[:, None]).argmax(1)
        return x_lo + fr * (x_hi - x_lo)
    return media, pct(0.5), pct(0.99)


filas = []
maxdif = 0.0
for etq, k in zonas:
    Hc, Ha = hist(cc, dc, yc, k), hist(ca, da, ya, k)
    # observado (pesos = 1 para cada anio) y comprobacion contra la lista directa
    mo_c, me_c, p_c = estadisticos(Hc.sum(0, keepdims=True)); mo_a, me_a, p_a = estadisticos(Ha.sum(0, keepdims=True))
    lc, la = dc[k[cc]], da[k[ca]]
    maxdif = max(maxdif, abs(mo_c[0] - lc.mean()), abs(p_c[0] - np.percentile(lc, 99)), abs(me_c[0] - np.median(lc)),
                 abs(mo_a[0] - la.mean()), abs(p_a[0] - np.percentile(la, 99)), abs(me_a[0] - np.median(la)))
    # replicas
    bc, ba = estadisticos(W_c @ Hc), estadisticos(W_a @ Ha)
    f = dict(zona=etq, celdas=int(k.sum()), rachas_cr2met=int(lc.size), rachas_aladin=int(la.size))
    obs = {'media': (mo_c[0], mo_a[0]), 'mediana': (me_c[0], me_a[0]), 'p99': (p_c[0], p_a[0])}
    for j, tag in enumerate(('media', 'mediana', 'p99')):
        o_c, o_a = obs[tag]
        rel = 100 * (o_a - o_c) / o_c
        rel_b = 100 * (ba[j] - bc[j]) / bc[j]
        f.update({f'pool_{tag}_cr2met': o_c, f'pool_{tag}_aladin': o_a, f'pool_{tag}_sesgo_pct': rel,
                  f'pool_{tag}_sesgo_pct_ic_inf': np.percentile(rel_b, 2.5), f'pool_{tag}_sesgo_pct_ic_sup': np.percentile(rel_b, 97.5)})
    for tag, D, col in (('media', dM, 'delta_media'), ('p99', dP, 'delta_p99')):
        mm = est.loc[k, col].values
        b = D[:, k].mean(1)
        f.update({f'celda_delta_{tag}_promedio': mm.mean(), f'celda_delta_{tag}_ic_inf': np.percentile(b, 2.5), f'celda_delta_{tag}_ic_sup': np.percentile(b, 97.5),
                  f'celda_delta_{tag}_mediana': np.median(mm)})
    filas.append(f)
res = pd.DataFrame(filas)
res.to_csv(S / 'paso_4_3_sesgo_por_region.csv', index=False)
chk.append(f'Comprobacion del pool por histogramas contra la lista directa (media, mediana y p99, 8 zonas x 2 productos): diferencia maxima = {maxdif:.2e} d')
for f in filas:
    chk.append(f'{f["zona"]} ({f["celdas"]} celdas; rachas CR2MET {f["rachas_cr2met"]}, ALADIN {f["rachas_aladin"]}):')
    chk.append(f'   A) Delta por celda promediado (dias) [IC95]: media {f["celda_delta_media_promedio"]:+.2f} [{f["celda_delta_media_ic_inf"]:+.2f}; {f["celda_delta_media_ic_sup"]:+.2f}] (mediana entre celdas {f["celda_delta_media_mediana"]:+.2f}); '
               f'p99 {f["celda_delta_p99_promedio"]:+.2f} [{f["celda_delta_p99_ic_inf"]:+.2f}; {f["celda_delta_p99_ic_sup"]:+.2f}] (mediana {f["celda_delta_p99_mediana"]:+.2f})')
    chk.append(f'   B) Pool regional: media CR2MET {f["pool_media_cr2met"]:.2f} d, ALADIN {f["pool_media_aladin"]:.2f} d, sesgo {f["pool_media_sesgo_pct"]:+.1f} % [{f["pool_media_sesgo_pct_ic_inf"]:+.1f}; {f["pool_media_sesgo_pct_ic_sup"]:+.1f}]; '
               f'mediana {f["pool_mediana_cr2met"]:.0f} | {f["pool_mediana_aladin"]:.0f} d, sesgo {f["pool_mediana_sesgo_pct"]:+.1f} % [{f["pool_mediana_sesgo_pct_ic_inf"]:+.1f}; {f["pool_mediana_sesgo_pct_ic_sup"]:+.1f}]; '
               f'p99 {f["pool_p99_cr2met"]:.1f} | {f["pool_p99_aladin"]:.1f} d, sesgo {f["pool_p99_sesgo_pct"]:+.1f} % [{f["pool_p99_sesgo_pct_ic_inf"]:+.1f}; {f["pool_p99_sesgo_pct_ic_sup"]:+.1f}]')
(S / 'paso_4_3_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))

cols = ['#555555', '#7B68B5', '#2C7FB8', '#2AA198', '#253494', '#8C6D31', '#D95F02', '#B2182B']
etqs = [f['zona'].replace('Metropolitana', 'Metrop.') for f in filas]
fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
def barras(ax, vals, lo, hi, tit, un, fmt='{:+.1f}'):
    x = np.arange(len(vals))
    ax.bar(x, vals, color=cols, width=0.65, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], capsize=3, ecolor='k')
    ax.axhline(0, color='k', lw=0.8)
    for i, v in enumerate(vals):
        ax.text(i, hi[i] + (max(hi) - min(lo)) * 0.02 if v >= 0 else lo[i] - (max(hi) - min(lo)) * 0.07, fmt.format(v).replace('.', ','), ha='center', fontsize=8)
    ax.margins(y=0.14); ax.set_xticks(x); ax.set_xticklabels(etqs, rotation=30, ha='right', fontsize=9); ax.set_ylabel(un); ax.set_title(tit, fontsize=10, loc='left')
barras(axs[0, 0], res['celda_delta_media_promedio'], res['celda_delta_media_ic_inf'], res['celda_delta_media_ic_sup'], 'A. Δ de la duración media por celda, promediado (ALADIN − CR2MET)', 'días', '{:+.2f}')
barras(axs[0, 1], res['celda_delta_p99_promedio'], res['celda_delta_p99_ic_inf'], res['celda_delta_p99_ic_sup'], 'A. Δ del p99 por celda, promediado (ALADIN − CR2MET)', 'días', '{:+.1f}')
barras(axs[1, 0], res['pool_media_sesgo_pct'], res['pool_media_sesgo_pct_ic_inf'], res['pool_media_sesgo_pct_ic_sup'], 'B. Sesgo relativo de la duración media del pool regional', '% de CR2MET')
barras(axs[1, 1], res['pool_p99_sesgo_pct'], res['pool_p99_sesgo_pct_ic_inf'], res['pool_p99_sesgo_pct_ic_sup'], 'B. Sesgo relativo del p99 del pool regional', '% de CR2MET')
fig.suptitle('Sesgo histórico de ALADIN (criterio i) por región, 1980–2014; barras de error = IC95 (bootstrap por años, 1 000 réplicas)', fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / 'figuras' / 'paso_4_3_sesgo_regiones.png', dpi=140)
