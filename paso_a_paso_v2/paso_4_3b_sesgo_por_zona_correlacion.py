"""Paso 4.3b (complemento del 4.3: promedio por zona, IC95 y correlacion espacial) — Resumen del sesgo historico de ALADIN (criterio i) por region y para el dominio: Delta medio y mediano en dias, sesgo relativo en %, con intervalos IC95 de bootstrap.

Estadistico de una zona = promedio simple de los estadisticos de sus celdas (cada celda pesa igual). Intervalos: se reutilizan los sorteos de anios guardados en el paso 4.2 (mismos pesos W para
CR2MET y ALADIN, 1 000 replicas): en cada replica se recalculan media y p99 de cada celda y producto, se promedian en la zona y se obtienen el Delta de la zona y el sesgo relativo
(promedio ALADIN / promedio CR2MET - 1). IC95 = percentiles 2,5 y 97,5. Ademas, correlacion espacial (Pearson y Spearman) entre las 972 celdas de CR2MET y ALADIN (adicion propuesta, sin intervalo).
Entradas: salidas/paso_3_1_rachas_completas_*.npz, salidas/paso_4_1_estadisticos_por_celda.csv, salidas/paso_4_2_deltas_replicas.npz
Salidas: salidas/paso_4_3b_resumen_por_zona.csv, salidas/paso_4_3b_chequeos.txt, figuras/paso_4_3b_sesgo_por_region.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
S = ROOT / 'salidas'
est = pd.read_csv(S / 'paso_4_1_estadisticos_por_celda.csv')
z42 = np.load(S / 'paso_4_2_deltas_replicas.npz')
W_c, W_a = z42['W_cr2met'], z42['W_aladin']
R = W_c.shape[0]
nombres = ['Valparaíso', 'Metropolitana', "O'Higgins", 'Maule', 'Ñuble', 'Biobío', 'La Araucanía']
nc = len(est)
rid = np.array([nombres.index(r) for r in est['region']])
chk = [f'{R} replicas (los sorteos de anios guardados en el paso 4.2); {nc} celdas']


def cargar(nom):
    z = np.load(S / f'paso_3_1_rachas_completas_{nom}.npz')
    cel, dur, anio = z['celda'], z['duracion'].astype(np.float64), z['anio_inicio'].astype(int) - 1980
    o = np.lexsort((dur, cel)); cel, dur, anio = cel[o], dur[o], anio[o]
    lim = np.r_[0, np.cumsum(np.bincount(cel, minlength=nc))]
    return [(dur[lim[j]:lim[j + 1]], anio[lim[j]:lim[j + 1]]) for j in range(nc)]


def stats_boot(d, yi, W):
    ws = W[:, yi].astype(np.float64); N = ws.sum(1)
    media = (ws * d).sum(1) / N
    cw = np.cumsum(ws, axis=1); pos = 0.99 * (N - 1); lo = np.floor(pos); hi = np.minimum(lo + 1, N - 1); fr = pos - lo
    p99 = d[(cw > lo[:, None]).argmax(1)] + fr * (d[(cw > hi[:, None]).argmax(1)] - d[(cw > lo[:, None]).argmax(1)])
    return media, p99


rc, ra = cargar('cr2met_hist'), cargar('aladin_hist_i')
Mc, Pc, Ma, Pa = (np.zeros((R, nc)) for _ in range(4))
for j in range(nc):
    Mc[:, j], Pc[:, j] = stats_boot(*rc[j], W_c)
    Ma[:, j], Pa[:, j] = stats_boot(*ra[j], W_a)
# control: las diferencias recalculadas coinciden con las guardadas en el paso 4.2
chk.append(f'Control: diferencia maxima entre Delta* recalculado y el guardado en 4.2: media {np.abs((Ma - Mc) - z42["delta_media"]).max():.2e} d; p99 {np.abs((Pa - Pc) - z42["delta_p99"]).max():.2e} d')

filas = []
zonas = [('Dominio (972 celdas)', np.ones(nc, bool))] + [(n, rid == r) for r, n in enumerate(nombres)]
for etq, k in zonas:
    f = dict(zona=etq, celdas=int(k.sum()))
    for tag, ec, ea, Bc, Ba in (('media', est['media_cr2met'].values, est['media_aladin'].values, Mc, Ma), ('p99', est['p99_cr2met'].values, est['p99_aladin'].values, Pc, Pa)):
        d = ea[k] - ec[k]
        f[f'{tag}_cr2met'], f[f'{tag}_aladin'] = ec[k].mean(), ea[k].mean()
        f[f'delta_{tag}_promedio'], f[f'delta_{tag}_mediano'] = d.mean(), np.median(d)
        f[f'rel_{tag}_pct'] = 100 * (ea[k].mean() / ec[k].mean() - 1)
        dz = Ba[:, k].mean(1) - Bc[:, k].mean(1)                       # Delta de la zona en cada replica
        rel = 100 * (Ba[:, k].mean(1) / Bc[:, k].mean(1) - 1)
        f[f'delta_{tag}_ic_inf'], f[f'delta_{tag}_ic_sup'] = np.percentile(dz, 2.5), np.percentile(dz, 97.5)
        f[f'rel_{tag}_ic_inf'], f[f'rel_{tag}_ic_sup'] = np.percentile(rel, 2.5), np.percentile(rel, 97.5)
        f[f'delta_{tag}_incluye_0'] = bool(f[f'delta_{tag}_ic_inf'] <= 0 <= f[f'delta_{tag}_ic_sup'])
    filas.append(f)
df = pd.DataFrame(filas)
df.to_csv(S / 'paso_4_3b_resumen_por_zona.csv', index=False)
for f in filas:
    chk.append(f'{f["zona"]}: MEDIA CR2MET {f["media_cr2met"]:.2f}, ALADIN {f["media_aladin"]:.2f}; Delta {f["delta_media_promedio"]:+.2f} d [{f["delta_media_ic_inf"]:+.2f}; {f["delta_media_ic_sup"]:+.2f}] (mediano {f["delta_media_mediano"]:+.2f}), '
               f'relativo {f["rel_media_pct"]:+.1f} % [{f["rel_media_ic_inf"]:+.1f}; {f["rel_media_ic_sup"]:+.1f}] | P99 CR2MET {f["p99_cr2met"]:.1f}, ALADIN {f["p99_aladin"]:.1f}; Delta {f["delta_p99_promedio"]:+.2f} d [{f["delta_p99_ic_inf"]:+.2f}; {f["delta_p99_ic_sup"]:+.2f}] '
               f'(mediano {f["delta_p99_mediano"]:+.2f}), relativo {f["rel_p99_pct"]:+.1f} % [{f["rel_p99_ic_inf"]:+.1f}; {f["rel_p99_ic_sup"]:+.1f}]')
for tag in ('media', 'p99'):
    a, b = est[f'{tag}_cr2met'].values, est[f'{tag}_aladin'].values
    chk.append(f'Correlacion espacial entre las {nc} celdas, {tag}: Pearson {stats.pearsonr(a, b)[0]:.3f}; Spearman {stats.spearmanr(a, b)[0]:.3f}')
(S / 'paso_4_3b_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))

col = {'Dominio (972 celdas)': '#7f7f7a', 'Valparaíso': '#7B68B5', 'Metropolitana': '#2C7FB8', "O'Higgins": '#2AA198', 'Maule': '#253494', 'Ñuble': '#8C6D31', 'Biobío': '#D95F02', 'La Araucanía': '#B2182B'}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
orden = list(df['zona'])[1:] + [df['zona'][0]]
for ax, tag, tit in zip(axs, ('media', 'p99'), ('Duración media: ALADIN − CR2MET (días)', 'p99: ALADIN − CR2MET (días)')):
    for i, z in enumerate(orden):
        f = df[df.zona == z].iloc[0]; y = len(orden) - 1 - i
        m, lo, hi = f[f'delta_{tag}_promedio'], f[f'delta_{tag}_ic_inf'], f[f'delta_{tag}_ic_sup']
        ax.barh(y, m, height=0.6, color=col[z])
        ax.plot([lo, hi], [y, y], color='k', lw=1.4); ax.plot([lo, lo], [y - 0.15, y + 0.15], 'k', lw=1.2); ax.plot([hi, hi], [y - 0.15, y + 0.15], 'k', lw=1.2)
        ax.text(max(hi, 0) + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01 + 0.05 * abs(hi), y, f'{m:+.2f} d ({f[f"rel_{tag}_pct"]:+.0f} %)'.replace('.', ','), va='center', fontsize=8)
    ax.axvline(0, color='k', lw=1); ax.set_title(tit, fontsize=10); ax.grid(axis='x', alpha=0.25)
axs[0].set_yticks(range(len(orden))); axs[0].set_yticklabels([z.replace(' (972 celdas)', '\n(972 celdas)') for z in reversed(orden)], fontsize=9)
fig.suptitle('Sesgo histórico de ALADIN (criterio i), 1980–2014: promedio de las diferencias por celda, con IC95 del bootstrap por años; entre paréntesis, sesgo relativo', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ax in axs:
    x0, x1 = ax.get_xlim(); ax.set_xlim(x0, x1 + (x1 - x0) * 0.35)
fig.savefig(ROOT / 'figuras' / 'paso_4_3b_sesgo_por_region.png', dpi=140)
