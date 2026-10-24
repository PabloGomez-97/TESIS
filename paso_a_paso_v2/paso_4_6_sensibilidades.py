"""Paso 4.6 — Sensibilidades historicas del sesgo de ALADIN (1980-2014): ¿cuanto depende lo visto de la definicion de dia humedo y del regrillado?

Alternativas (todas con CR2MET a 0,1 mm como referencia de dia humedo):
  i   principal: ALADIN con R0 global (paso 2.3); CR2MET interpolado linealmente (bilineal).
  ii  ALADIN con 0,1 mm (mismo umbral en ambos, sin calibrar).
  iii ALADIN con R0 local por celda (paso 2.5).
  iv  regrillado: CR2MET promediado por area (promedio de los puntos de 0,05 deg cuyo centro cae dentro de cada celda de ALADIN) en lugar de interpolado; el R0 de ALADIN se recalibra por biseccion/cuantil
      para igualar la frecuencia de esa serie en el dominio (misma regla del paso 2.3).
Para cada alternativa: media y p99 por celda con todas las rachas completas, Delta = ALADIN - CR2MET por celda, bootstrap por anios con los MISMOS sorteos del paso 4.2 (independientes entre productos),
% de celdas con diferencia clara, y Delta promedio por region con IC95.
Entradas: salidas/paso_0_1_mascara_dominio.npz, paso_1_1_cr2met_interpolado_dominio.nc, paso_1_2_aladin_hist_1980_2014.nc, paso_2_3_R0.json, paso_3_1_rachas_completas_*.npz, paso_4_2_deltas_replicas.npz, pr/CR2MET_*.nc, pr1/ (limites de celda)
Salidas: salidas/paso_4_6_sensibilidades.csv, salidas/paso_4_6_regrillado.txt, salidas/paso_4_6_rachas_completas_{cr2met_area,aladin_area_i}.npz, salidas/paso_4_6_chequeos.txt, figuras/paso_4_6_sensibilidades.png
"""
from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MPath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
S = ROOT / 'salidas'
nombres = ['Valparaíso', 'Metropolitana', "O'Higgins", 'Maule', 'Ñuble', 'Biobío', 'La Araucanía']
m0 = np.load(S / 'paso_0_1_mascara_dominio.npz')
dom = m0['dominio']; jj, ii = np.where(dom)
cr = xr.open_dataset(S / 'paso_1_1_cr2met_interpolado_dominio.nc'); al = xr.open_dataset(S / 'paso_1_2_aladin_hist_1980_2014.nc')
rid = cr['region_id'].values; nc = rid.size
t = pd.DatetimeIndex(cr['time'].values)
R0 = json.load(open(S / 'paso_2_3_R0.json'))['R0_usado_mm']
rep = np.load(S / 'paso_4_2_deltas_replicas.npz'); W_c, W_a = rep['W_cr2met'], rep['W_aladin']; R = W_c.shape[0]
chk, reg = [], []


# ---------------- iv. regrillado: CR2MET promediado por area -----------------
f1 = sorted(glob.glob(str(ROOT / 'pr1' / 'pr_CHP12_*_historical_*19790101-20141231*.nc')))[0]
g = xr.open_dataset(f1)
blat, blon = g['bounds_lat'].values[jj, ii], g['bounds_lon'].values[jj, ii]          # (972, 4) esquinas de cada celda
ds = xr.open_mfdataset(sorted(glob.glob(str(ROOT / 'pr' / 'CR2MET_pr_v2.5_day_*.nc'))), chunks={'time': 366}, combine='by_coords', data_vars='minimal')
box = dict(lat=slice(blat.min() - 0.2, blat.max() + 0.2), lon=slice(blon.min() - 0.2, blon.max() + 0.2))
pr = ds['pr'].sel(time=slice('1980-01-01', '2014-12-31')).sel(**box).load()
LA, LO = np.meshgrid(pr['lat'].values, pr['lon'].values, indexing='ij')
pts = np.column_stack([LO.ravel(), LA.ravel()])
P = pr.values.reshape(pr.shape[0], -1).astype('float32')
area = np.full((P.shape[0], nc), np.nan, dtype='float32'); npts = np.zeros(nc, int); nsin = 0
for j in range(nc):
    poly = MPath(np.column_stack([blon[j], blat[j]]))
    idx = np.where(poly.contains_points(pts))[0]
    npts[j] = idx.size
    if idx.size == 0:
        nsin += 1; continue
    area[:, j] = np.nanmean(P[:, idx], axis=1) if np.isfinite(P[:, idx]).any() else np.nan
chk.append(f'Regrillado por area: puntos CR2MET (0,05 deg) dentro de cada celda de ALADIN: min {npts.min()}, mediana {int(np.median(npts))}, max {npts.max()}; celdas sin puntos: {nsin}; NaN en la serie por area: {int(np.isnan(area).sum())}')
assert nsin == 0 and not np.isnan(area).any()
bil = cr['pr'].values
fh_b, fh_a = (bil >= 0.1).mean() * 100, (area >= 0.1).mean() * 100
chk.append(f'Frecuencia de dias >= 0,1 mm en el dominio: bilineal {fh_b:.3f} %; promedio por area {fh_a:.3f} % (diferencia {fh_a - fh_b:+.3f} pp); lluvia anual media: bilineal {bil.mean() * 365.25:.0f} mm, area {area.mean() * 365.25:.0f} mm; correlacion diaria (promedio del dominio) {np.corrcoef(bil.mean(1), area.mean(1))[0, 1]:.4f}')
a_ = al['pr'].values.astype('float64').ravel()
k = int(round((area >= 0.1).mean() * a_.size)); R0_area = float(np.partition(a_, a_.size - k)[a_.size - k])
chk.append(f'R0 recalibrado con CR2MET por area: {R0_area:.3f} mm (R0 del criterio i: {R0} mm); F ALADIN con ese R0: {(a_ >= R0_area).mean() * 100:.4f} % (objetivo {fh_a:.4f} %)')
reg.append('\n'.join(chk))


# ---------------- utilidades de rachas y bootstrap -----------------
def rachas(seco):
    nd = seco.shape[0]; cel, ini, dur, cen = [], [], [], []
    for j in range(seco.shape[1]):
        d = np.diff(np.r_[False, seco[:, j], False].astype(np.int8))
        s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        cel.append(np.full(s.size, j, np.int16)); ini.append(s); dur.append(e - s); cen.append((s == 0) | (e == nd))
    return np.concatenate(cel), np.concatenate(ini), np.concatenate(dur), np.concatenate(cen)


def guardar_completas(nombre, pr_, umbral):
    cel, ini, dur, cen = rachas(pr_ < (umbral[None, :] if np.ndim(umbral) else umbral))
    np.savez_compressed(S / f'paso_4_6_rachas_completas_{nombre}.npz', celda=cel[~cen], inicio=ini[~cen], duracion=dur[~cen].astype(np.int16), anio_inicio=(t.year.values[ini[~cen]]).astype(np.int16))
    chk.append(f'{nombre}: rachas {dur.size} (censuradas {int(cen.sum())}); dias secos {100 * (pr_ < umbral).mean():.2f} %')


guardar_completas('cr2met_area', area.astype('float64'), 0.1)
guardar_completas('aladin_area_i', al['pr'].values.astype('float64'), R0_area)


def cargar(path):
    z = np.load(path)
    cel, dur, anio = z['celda'], z['duracion'].astype(np.float64), z['anio_inicio'].astype(int) - 1980
    o = np.lexsort((dur, cel)); cel, dur, anio = cel[o], dur[o], anio[o]
    lim = np.r_[0, np.cumsum(np.bincount(cel, minlength=nc))]
    return [(dur[lim[j]:lim[j + 1]], anio[lim[j]:lim[j + 1]]) for j in range(nc)]


def stats_boot(d, yi, W):
    ws = W[:, yi].astype(np.float64); N = ws.sum(1)
    media = (ws * d).sum(1) / N
    cw = np.cumsum(ws, axis=1); pos = 0.99 * (N - 1); lo = np.floor(pos); hi = np.minimum(lo + 1, N - 1); fr = pos - lo
    xl = d[(cw > lo[:, None]).argmax(1)]; xh = d[(cw > hi[:, None]).argmax(1)]
    return media, xl + fr * (xh - xl)


def observados(datos):
    m, p = np.zeros(nc), np.zeros(nc)
    for j, (d, y) in enumerate(datos):
        m[j], p[j] = d.mean(), np.percentile(d, 99)
    return m, p


alternativas = {
    'i (principal)': (S / 'paso_3_1_rachas_completas_cr2met_hist.npz', S / 'paso_3_1_rachas_completas_aladin_hist_i.npz'),
    'ii (0,1 mm en ambos)': (S / 'paso_3_1_rachas_completas_cr2met_hist.npz', S / 'paso_3_1_rachas_completas_aladin_hist_ii.npz'),
    'iii (R₀ local)': (S / 'paso_3_1_rachas_completas_cr2met_hist.npz', S / 'paso_3_1_rachas_completas_aladin_hist_iii.npz'),
    'iv (regrillado por área)': (S / 'paso_4_6_rachas_completas_cr2met_area.npz', S / 'paso_4_6_rachas_completas_aladin_area_i.npz'),
}
zonas = [('Dominio', np.ones(nc, bool))] + [(n, rid == r) for r, n in enumerate(nombres)]
filas = []
for alt, (pc, pa) in alternativas.items():
    rc, ra = cargar(pc), cargar(pa)
    mc, pcc = observados(rc); ma, paa = observados(ra)
    Dm, Dp = np.zeros((R, nc)), np.zeros((R, nc))
    for j in range(nc):
        a1, b1 = stats_boot(*rc[j], W_c); a2, b2 = stats_boot(*ra[j], W_a)
        Dm[:, j], Dp[:, j] = a2 - a1, b2 - b1
    dm, dp = ma - mc, paa - pcc
    chk.append(f'{alt}: Delta observado dentro del IC95 por celda: media {int(((dm >= np.percentile(Dm, 2.5, 0)) & (dm <= np.percentile(Dm, 97.5, 0))).sum())}/{nc}, p99 {int(((dp >= np.percentile(Dp, 2.5, 0)) & (dp <= np.percentile(Dp, 97.5, 0))).sum())}/{nc}; correlacion espacial: media {np.corrcoef(mc, ma)[0, 1]:.3f}, p99 {np.corrcoef(pcc, paa)[0, 1]:.3f}')
    loM, hiM, loP, hiP = (np.percentile(D, q, 0) for D in (Dm, Dp) for q in (2.5, 97.5))
    for zn, kz in zonas:
        f = dict(alternativa=alt, zona=zn, celdas=int(kz.sum()))
        for tag, d, D, lo, hi, base in (('media', dm, Dm, loM, hiM, mc), ('p99', dp, Dp, loP, hiP, pcc)):
            dz = D[:, kz].mean(1)
            f[f'delta_{tag}'] = d[kz].mean(); f[f'delta_{tag}_ic_inf'], f[f'delta_{tag}_ic_sup'] = np.percentile(dz, 2.5), np.percentile(dz, 97.5)
            f[f'rel_{tag}_pct'] = 100 * d[kz].mean() / base[kz].mean()
            f[f'{tag}_mas_corto_pct'] = 100 * (hi[kz] < 0).mean(); f[f'{tag}_mas_largo_pct'] = 100 * (lo[kz] > 0).mean()
        filas.append(f)
df = pd.DataFrame(filas); df.to_csv(S / 'paso_4_6_sensibilidades.csv', index=False)
for alt in alternativas:
    f = df[(df.alternativa == alt) & (df.zona == 'Dominio')].iloc[0]
    chk.append(f'DOMINIO {alt}: Delta media {f["delta_media"]:+.2f} d [{f["delta_media_ic_inf"]:+.2f}; {f["delta_media_ic_sup"]:+.2f}] ({f["rel_media_pct"]:+.1f} %); celdas ALADIN mas corto {f["media_mas_corto_pct"]:.0f} % / mas largo {f["media_mas_largo_pct"]:.0f} % | '
               f'Delta p99 {f["delta_p99"]:+.2f} d [{f["delta_p99_ic_inf"]:+.2f}; {f["delta_p99_ic_sup"]:+.2f}] ({f["rel_p99_pct"]:+.1f} %); celdas mas corto {f["p99_mas_corto_pct"]:.0f} % / mas largo {f["p99_mas_largo_pct"]:.0f} %')
chk.append('Delta de la media por region [IC95] (dias), por alternativa:')
for zn, _ in zonas[1:]:
    chk.append(f'  {zn:<14} ' + ' | '.join(f'{alt.split(" ")[0]} {df[(df.alternativa == alt) & (df.zona == zn)].iloc[0]["delta_media"]:+.2f} [{df[(df.alternativa == alt) & (df.zona == zn)].iloc[0]["delta_media_ic_inf"]:+.2f};{df[(df.alternativa == alt) & (df.zona == zn)].iloc[0]["delta_media_ic_sup"]:+.2f}]' for alt in alternativas))
chk.append('Delta del p99 por region [IC95] (dias), por alternativa:')
for zn, _ in zonas[1:]:
    chk.append(f'  {zn:<14} ' + ' | '.join(f'{alt.split(" ")[0]} {df[(df.alternativa == alt) & (df.zona == zn)].iloc[0]["delta_p99"]:+.2f} [{df[(df.alternativa == alt) & (df.zona == zn)].iloc[0]["delta_p99_ic_inf"]:+.2f};{df[(df.alternativa == alt) & (df.zona == zn)].iloc[0]["delta_p99_ic_sup"]:+.2f}]' for alt in alternativas))
(S / 'paso_4_6_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
(S / 'paso_4_6_regrillado.txt').write_text(reg[0], encoding='utf-8')
print('\n'.join(chk))

col = {'i (principal)': '#3f77d1', 'ii (0,1 mm en ambos)': '#d4564e', 'iii (R₀ local)': '#2AA198', 'iv (regrillado por área)': '#8C6D31'}
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
orden = [z for z, _ in zonas[1:]] + ['Dominio']
for ax, tag, tit in zip(axs, ('media', 'p99'), ('Δ duración media (días)', 'Δ p99 (días)')):
    for ia, alt in enumerate(alternativas):
        for iz, zn in enumerate(orden):
            f = df[(df.alternativa == alt) & (df.zona == zn)].iloc[0]; y = len(orden) - 1 - iz + (ia - 1.5) * 0.17
            ax.plot([f[f'delta_{tag}_ic_inf'], f[f'delta_{tag}_ic_sup']], [y, y], color=col[alt], lw=1.2)
            ax.plot(f[f'delta_{tag}'], y, 'o', color=col[alt], ms=4.5, label=alt if iz == 0 else None)
    ax.axvline(0, color='k', lw=1); ax.set_title(tit + ': ALADIN − CR2MET, promedio por celda con IC95', fontsize=10); ax.grid(axis='x', alpha=0.25)
axs[0].set_yticks(range(len(orden))); axs[0].set_yticklabels(list(reversed(orden))); axs[0].legend(fontsize=8, loc='lower left')
fig.suptitle('Sensibilidad del sesgo histórico a la definición de día húmedo y al regrillado (1980–2014)', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(ROOT / 'figuras' / 'paso_4_6_sensibilidades.png', dpi=140)
