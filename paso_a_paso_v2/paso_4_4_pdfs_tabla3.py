"""Paso 4.4 — PDFs de duracion por region (CR2MET frente a ALADIN criterio i, 1980-2014) y Tabla 3 (estadisticos del conjunto regional).

PDF empirica de la duracion t (dias): bins uniformes en log10(t); 26 bordes entre 1 dia y la duracion maxima observada entre CR2MET hist., ALADIN hist. (criterio i) y ALADIN SSP5-8.5
(criterio i) de la region, redondeados a enteros, con los bordes repetidos fusionados (los mismos bins se usaran en el paso 5.1 para el futuro). f(t) = numero de rachas del bin / ancho del bin
(dias), de modo que el area es el numero total de rachas. El binning solo afecta a la figura: los estadisticos de la tabla se calculan sobre las duraciones.
Bandas y IC95: remuestreo de anios del paso 4.2 (mismos sorteos; independientes entre productos). Tabla 3: n, media, mediana, p90, p99, maximo y % de rachas >= 20 d.
Entradas: salidas/paso_3_1_rachas_completas_{cr2met_hist,aladin_hist_i,aladin_ssp585_i}.npz, salidas/paso_4_2_deltas_replicas.npz
Salidas: salidas/paso_4_4_tabla3_historico.csv, salidas/paso_4_4_diferencias_ic.csv, salidas/paso_4_4_pdfs.npz, salidas/paso_4_4_chequeos.txt, figuras/paso_4_4_pdfs_regiones.png
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
W_c, W_a = rep['W_cr2met'], rep['W_aladin']
R = W_c.shape[0]
chk = [f'{R} replicas del paso 4.2']


def cargar(nom):
    z = np.load(S / f'paso_3_1_rachas_completas_{nom}.npz')
    return z['celda'], z['duracion'].astype(int), z['anio_inicio'].astype(int) - 1980


cc, dc, yc = cargar('cr2met_hist'); ca, da, ya = cargar('aladin_hist_i'); cf, df_, yf = cargar('aladin_ssp585_i')
DMAX = max(dc.max(), da.max(), df_.max()) + 1
zonas = [('Dominio', np.ones(nc, bool))] + [(n, rid == r) for r, n in enumerate(nombres)]


def hist_anio(cel, dur, anio, k):
    m = k[cel]
    H = np.zeros((35, DMAX)); np.add.at(H, (anio[m], dur[m]), 1)
    return H


def stats(counts):
    """counts (R, DMAX) -> dict de vectores: n, media, mediana, p90, p99, pct>=20 (numpy.percentile lineal)."""
    d = np.arange(DMAX); n = counts.sum(1); cw = np.cumsum(counts, axis=1)
    def pct(q):
        pos = q * (n - 1); lo = np.floor(pos); hi = np.minimum(lo + 1, n - 1); fr = pos - lo
        x_lo = (cw > lo[:, None]).argmax(1); x_hi = (cw > hi[:, None]).argmax(1)
        return x_lo + fr * (x_hi - x_lo)
    return dict(n=n, media=(counts * d).sum(1) / n, mediana=pct(0.5), p90=pct(0.9), p99=pct(0.99), pct20=100 * counts[:, 20:].sum(1) / n)


filas, difs, pdfs = [], [], {}
fig, axs = plt.subplots(2, 4, figsize=(18, 8.5), sharey=False)
maxdif = 0.0
for ax, (etq, k) in zip(axs.ravel(), zonas):
    Hc, Ha = hist_anio(cc, dc, yc, k), hist_anio(ca, da, ya, k)
    dmax_z = max(dc[k[cc]].max(), da[k[ca]].max(), df_[k[cf]].max())
    edges = np.unique(np.round(10 ** np.linspace(0, np.log10(dmax_z), 26)).astype(int))
    edges[-1] = dmax_z + 1
    ancho = np.diff(edges).astype(float)
    B = np.zeros((DMAX, edges.size - 1))
    for b in range(edges.size - 1):
        B[edges[b]:edges[b + 1], b] = 1
    obs_c, obs_a = stats(Hc.sum(0, keepdims=True)), stats(Ha.sum(0, keepdims=True))
    lc, la = dc[k[cc]], da[k[ca]]
    maxdif = max(maxdif, abs(obs_c['media'][0] - lc.mean()), abs(obs_c['p99'][0] - np.percentile(lc, 99)), abs(obs_c['p90'][0] - np.percentile(lc, 90)),
                 abs(obs_a['media'][0] - la.mean()), abs(obs_a['p99'][0] - np.percentile(la, 99)), abs(obs_a['pct20'][0] - 100 * (la >= 20).mean()))
    bc, ba = stats(W_c @ Hc), stats(W_a @ Ha)
    for nom, o, x in (('CR2MET (0,1 mm)', obs_c, lc), ('ALADIN criterio i', obs_a, la)):
        filas.append(dict(zona=etq, serie=nom, n=int(x.size), media=o['media'][0], mediana=o['mediana'][0], p90=o['p90'][0], p99=o['p99'][0], maximo=int(x.max()), pct_ge20=o['pct20'][0]))
    f = dict(zona=etq)
    for tag in ('media', 'p90', 'p99'):
        rel = 100 * (obs_a[tag][0] - obs_c[tag][0]) / obs_c[tag][0]; rb = 100 * (ba[tag] - bc[tag]) / bc[tag]
        f.update({f'{tag}_sesgo_pct': rel, f'{tag}_ic_inf': np.percentile(rb, 2.5), f'{tag}_ic_sup': np.percentile(rb, 97.5)})
    dpp = obs_a['pct20'][0] - obs_c['pct20'][0]; db = ba['pct20'] - bc['pct20']
    f.update(pct_ge20_dif_pp=dpp, pct_ge20_ic_inf=np.percentile(db, 2.5), pct_ge20_ic_sup=np.percentile(db, 97.5))
    difs.append(f)
    # PDFs y bandas
    xm = np.sqrt(edges[:-1] * (edges[1:] - 1).clip(min=edges[:-1]))
    dens = lambda H, W: (W @ H) @ B / ancho
    f_c, f_a = dens(Hc, np.ones((1, 35))).ravel(), dens(Ha, np.ones((1, 35))).ravel()
    Fc_b, Fa_b = dens(Hc, W_c), dens(Ha, W_a)
    pdfs[etq] = dict(edges=edges, f_cr2met=f_c, f_aladin=f_a, lo_cr2met=np.percentile(Fc_b, 2.5, axis=0), hi_cr2met=np.percentile(Fc_b, 97.5, axis=0),
                     lo_aladin=np.percentile(Fa_b, 2.5, axis=0), hi_aladin=np.percentile(Fa_b, 97.5, axis=0))
    p = pdfs[etq]
    ax.fill_between(xm, np.where(p['lo_cr2met'] > 0, p['lo_cr2met'], np.nan), p['hi_cr2met'], color='#555555', alpha=0.25, lw=0)
    ax.fill_between(xm, np.where(p['lo_aladin'] > 0, p['lo_aladin'], np.nan), p['hi_aladin'], color='#d4564e', alpha=0.25, lw=0)
    ax.plot(xm, np.where(f_c > 0, f_c, np.nan), '-o', color='#333333', ms=3, lw=1.3, label='CR2MET (0,1 mm)')
    ax.plot(xm, np.where(f_a > 0, f_a, np.nan), '-o', color='#d4564e', ms=3, lw=1.3, label='ALADIN (criterio i)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(f'{etq}  ({lc.size:,} | {la.size:,} rachas)'.replace(',', ' '), fontsize=10); ax.set_xlabel('duración t (días)'); ax.set_ylabel('f(t): rachas por día de ancho de bin')
    ax.grid(alpha=0.2, which='both')
axs[0, 0].legend(fontsize=8)
fig.suptitle('PDFs empíricas de la duración de las rachas secas, 1980–2014 (CR2MET frente a ALADIN criterio i); bandas = IC95 del bootstrap por años', fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / 'figuras' / 'paso_4_4_pdfs_regiones.png', dpi=130)

pd.DataFrame(filas).to_csv(S / 'paso_4_4_tabla3_historico.csv', index=False)
dd = pd.DataFrame(difs); dd.to_csv(S / 'paso_4_4_diferencias_ic.csv', index=False)
np.savez(S / 'paso_4_4_pdfs.npz', **{f'{z}__{key}': v for z, p in pdfs.items() for key, v in p.items()})
chk.append(f'Comprobacion de los estadisticos agrupados contra la lista directa: diferencia maxima = {maxdif:.2e}')
chk.append('Tabla 3 (historico): zona | serie | n | media | mediana | p90 | p99 | maximo | % >= 20 d')
for r in filas:
    chk.append(f'  {r["zona"]:<14} | {r["serie"]:<18} | {r["n"]:>7} | {r["media"]:6.2f} | {r["mediana"]:4.1f} | {r["p90"]:5.1f} | {r["p99"]:6.1f} | {r["maximo"]:>4} | {r["pct_ge20"]:5.2f}')
chk.append('Diferencias ALADIN - CR2MET con IC95 (sesgo relativo %, salvo la ultima columna en pp): zona | media | p90 | p99 | % >= 20 d (pp)')
for f in difs:
    chk.append(f'  {f["zona"]:<14} | {f["media_sesgo_pct"]:+5.1f} [{f["media_ic_inf"]:+5.1f}; {f["media_ic_sup"]:+5.1f}] | {f["p90_sesgo_pct"]:+5.1f} [{f["p90_ic_inf"]:+5.1f}; {f["p90_ic_sup"]:+5.1f}] | '
               f'{f["p99_sesgo_pct"]:+5.1f} [{f["p99_ic_inf"]:+5.1f}; {f["p99_ic_sup"]:+5.1f}] | {f["pct_ge20_dif_pp"]:+5.2f} [{f["pct_ge20_ic_inf"]:+5.2f}; {f["pct_ge20_ic_sup"]:+5.2f}]')
(S / 'paso_4_4_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))
