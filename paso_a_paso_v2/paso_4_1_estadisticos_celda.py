"""Paso 4.1 — Duracion media y p99 de las rachas secas completas por celda (CR2MET frente a ALADIN criterio i, 1980-2014) y mapas de la diferencia.

Estadistico por celda con todas sus rachas completas; Delta = ALADIN - CR2MET en las celdas con estadistico en AMBAS series (celdas comunes). No hay minimo de rachas: se informa el numero de rachas por celda.
media = promedio simple; p99 = numpy.percentile lineal (ver paso 3.3).
Entradas: salidas/paso_3_1_rachas_completas_{cr2met_hist,aladin_hist_i}.npz, salidas/paso_1_1_cr2met_interpolado_dominio.nc (coordenadas y regiones), salidas/paso_0_1_mascara_dominio.npz
Salidas: salidas/paso_4_1_estadisticos_por_celda.csv, salidas/paso_4_1_resumen.csv, salidas/paso_4_1_chequeos.txt, figuras/paso_4_1_mapas_media_p99.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
S = ROOT / 'salidas'
nombres = list(np.load(S / 'paso_0_1_mascara_dominio.npz')['nombres'])
cr = xr.open_dataset(S / 'paso_1_1_cr2met_interpolado_dominio.nc')
lat, lon, rid = cr['lat_celda'].values, cr['lon_celda'].values, cr['region_id'].values
nc = lat.size
chk = []


def por_celda(nom):
    z = np.load(S / f'paso_3_1_rachas_completas_{nom}.npz')
    cel, dur = z['celda'], z['duracion'].astype(float)
    n = np.bincount(cel, minlength=nc)
    media, p99 = np.full(nc, np.nan), np.full(nc, np.nan)
    orden = np.argsort(cel, kind='stable')
    cs, ds = cel[orden], dur[orden]
    lim = np.r_[0, np.cumsum(np.bincount(cel, minlength=nc))]
    for j in range(nc):
        if n[j] > 0:
            d = ds[lim[j]:lim[j + 1]]
            media[j], p99[j] = d.mean(), np.percentile(d, 99)
    return n, media, p99


n_c, m_c, p_c = por_celda('cr2met_hist')
n_a, m_a, p_a = por_celda('aladin_hist_i')
val_c, val_a = np.isfinite(m_c), np.isfinite(m_a)
com = val_c & val_a
chk.append(f'Celdas con al menos una racha completa: CR2MET {val_c.sum()} de {nc}; ALADIN {val_a.sum()} de {nc}; validas en ambas (celdas comunes): {com.sum()}')
chk.append(f'Rachas completas por celda: CR2MET min {n_c.min()}, mediana {int(np.median(n_c))}, max {n_c.max()}; ALADIN min {n_a.min()}, mediana {int(np.median(n_a))}, max {n_a.max()}')
dm, dp = m_a - m_c, p_a - p_c
df = pd.DataFrame(dict(celda=np.arange(nc), region=[nombres[i] for i in rid], lat=lat, lon=lon, rachas_cr2met=n_c, rachas_aladin=n_a, media_cr2met=m_c, media_aladin=m_a,
                       delta_media=dm, p99_cr2met=p_c, p99_aladin=p_a, delta_p99=dp, comun=com))
df.to_csv(S / 'paso_4_1_estadisticos_por_celda.csv', index=False)
# control con el ejemplo a mano del paso 3.3 (celda 845)
chk.append(f'Control con el paso 3.3 (celda 845): CR2MET media {m_c[845]:.3f}, p99 {p_c[845]:.3f}; ALADIN media {m_a[845]:.3f}, p99 {p_a[845]:.3f} (a mano: 11,720 / 84,610 y 9,464 / 83,560)')
# resumen del dominio (celdas comunes) y por region
filas = []
for etq, k in [('Dominio (celdas comunes)', np.ones(nc, bool))] + [(n, rid == r) for r, n in enumerate(nombres)]:
    s = com & k
    filas.append(dict(zona=etq, celdas=int(s.sum()), media_cr2met=np.mean(m_c[s]), media_aladin=np.mean(m_a[s]), delta_media_promedio=np.mean(dm[s]), delta_media_mediana=np.median(dm[s]),
                      p99_cr2met=np.mean(p_c[s]), p99_aladin=np.mean(p_a[s]), delta_p99_promedio=np.mean(dp[s]), delta_p99_mediana=np.median(dp[s]),
                      celdas_aladin_mas_corto_media=int((dm[s] < 0).sum()), celdas_aladin_mas_corto_p99=int((dp[s] < 0).sum())))
res = pd.DataFrame(filas)
res.to_csv(S / 'paso_4_1_resumen.csv', index=False)
for f in filas:
    chk.append(f'{f["zona"]} ({f["celdas"]} celdas): media CR2MET {f["media_cr2met"]:.2f} d, ALADIN {f["media_aladin"]:.2f} d, Delta promedio {f["delta_media_promedio"]:+.2f} (mediana {f["delta_media_mediana"]:+.2f}); '
               f'p99 CR2MET {f["p99_cr2met"]:.1f} d, ALADIN {f["p99_aladin"]:.1f} d, Delta promedio {f["delta_p99_promedio"]:+.2f} (mediana {f["delta_p99_mediana"]:+.2f}); celdas con ALADIN mas corto: media {f["celdas_aladin_mas_corto_media"]}, p99 {f["celdas_aladin_mas_corto_p99"]}')
(S / 'paso_4_1_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))

fig, axs = plt.subplots(2, 3, figsize=(15, 13))
for fila, (nom, a, c, d, un) in enumerate((('Duración media', m_a, m_c, dm, 'días'), ('p99', p_a, p_c, dp, 'días'))):
    vmax = np.nanmax([np.nanmax(a), np.nanmax(c)]); vmin = np.nanmin([np.nanmin(a), np.nanmin(c)])
    for col, (v, tit) in enumerate(((c, 'CR2MET (0,1 mm)'), (a, 'ALADIN (criterio i)'))):
        ax = axs[fila, col]
        sc = ax.scatter(lon, lat, c=v, s=22, marker='s', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'{nom}: {tit}', fontsize=10); fig.colorbar(sc, ax=ax, shrink=0.6, label=un)
    ax = axs[fila, 2]
    lim = np.nanpercentile(np.abs(d), 98)
    sc = ax.scatter(lon, lat, c=d, s=22, marker='s', cmap='RdBu', vmin=-lim, vmax=lim)
    ax.set_title(f'Δ {nom} = ALADIN − CR2MET', fontsize=10); fig.colorbar(sc, ax=ax, shrink=0.6, label=un, extend='both')
    for ax in axs[fila]:
        ax.set_aspect(1 / np.cos(np.deg2rad(36))); ax.set_xlabel('longitud'); ax.set_ylabel('latitud')
fig.suptitle(f'Rachas secas completas por celda, 1980–2014 ({com.sum()} celdas comunes a ambos productos)', fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / 'figuras' / 'paso_4_1_mapas_media_p99.png', dpi=130)
