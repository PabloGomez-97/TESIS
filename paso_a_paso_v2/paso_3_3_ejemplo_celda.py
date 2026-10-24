"""Paso 3.3 — Ejemplo a mano con UNA celda (la 845, cercana a Santiago): de la lista de rachas completas a la media, la mediana y el p99.

Reglas que usaran los pasos siguientes (las mismas del analisis por celda): se usan solo rachas completas; media = promedio simple de las duraciones; p99 = percentil 99 con interpolacion lineal (numpy.percentile, metodo 'linear'), calculado aqui tambien a mano.
Entradas: salidas/paso_3_1_rachas_completas_{cr2met_hist,aladin_hist_i}.npz. Salidas: salidas/paso_3_3_ejemplo_celda.csv, salidas/paso_3_3_chequeos.txt, figuras/paso_3_3_ejemplo_celda.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
S = ROOT / 'salidas'
CELDA = 845
series = {'CR2MET (0,1 mm)': 'cr2met_hist', 'ALADIN criterio i (R₀ = 5,742 mm)': 'aladin_hist_i'}
chk, filas, dd = [f'Celda {CELDA}'], [], {}


def p99_a_mano(d):
    x = np.sort(d.astype(float)); n = x.size
    pos = 0.99 * (n - 1)                  # posicion (empezando en 0) del percentil 99 en la lista ordenada
    lo = int(np.floor(pos)); fr = pos - lo
    v = x[lo] + fr * (x[min(lo + 1, n - 1)] - x[lo])
    return v, pos, x[lo], x[min(lo + 1, n - 1)]


for etq, nom in series.items():
    z = np.load(S / f'paso_3_1_rachas_completas_{nom}.npz')
    d = z['duracion'][z['celda'] == CELDA].astype(int)
    dd[etq] = d
    n = d.size
    v, pos, a, b = p99_a_mano(d)
    fila = dict(serie=etq, rachas_completas=n, media=d.mean(), mediana=float(np.median(d)), p99=v, maximo=int(d.max()),
                suma_duraciones=int(d.sum()))
    filas.append(fila)
    chk.append(f'{etq}: {n} rachas completas; suma de duraciones {d.sum()} d; media = {d.sum()} / {n} = {d.mean():.3f} d; mediana {np.median(d):.1f} d; maximo {d.max()} d')
    chk.append(f'   p99 a mano: lista ordenada de {n} duraciones; posicion 0,99 x ({n} - 1) = {pos:.3f}; entre el valor en la posicion {int(np.floor(pos))} ({a:.0f} d) y el siguiente ({b:.0f} d) -> {v:.3f} d; '
               f'numpy.percentile da {np.percentile(d, 99):.3f} d; coinciden: {abs(v - np.percentile(d, 99)) < 1e-9}')
    chk.append(f'   rachas que duran mas de {v:.0f} d: {int((d > v).sum())} ({100 * (d > v).mean():.2f} % de las {n}); las 8 mas largas: {sorted(d)[-8:]}')
    c = np.bincount(d)
    chk.append('   cuantas rachas hay de 1, 2, 3, 4, 5, 6-10, 11-20, 21-50, >50 dias: ' + str([int(c[1]), int(c[2]), int(c[3]), int(c[4]), int(c[5]), int(c[6:11].sum()), int(c[11:21].sum()), int(c[21:51].sum()), int(c[51:].sum())]))
dm, dp = filas[1]['media'] - filas[0]['media'], filas[1]['p99'] - filas[0]['p99']
chk.append(f'Diferencia ALADIN - CR2MET en esta celda: media {dm:+.2f} d; p99 {dp:+.2f} d (esto es lo que el paso 4.1 calculara para cada una de las 972 celdas y dibujara en un mapa)')
pd.DataFrame(filas).to_csv(S / 'paso_3_3_ejemplo_celda.csv', index=False)
(S / 'paso_3_3_chequeos.txt').write_text('\n'.join(chk), encoding='utf-8')
print('\n'.join(chk))

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, (etq, d), col in zip(axs, dd.items(), ('#555555', '#d4564e')):
    c = np.bincount(d)[1:]
    ax.bar(np.arange(1, c.size + 1), c, width=0.9, color=col)
    ax.set_yscale('log'); ax.set_xlim(0, 105)
    f = [r for r in filas if r['serie'] == etq][0]
    for val, lab, ls in ((f['mediana'], f'mediana = {f["mediana"]:.0f} d', ':'), (f['media'], f'media = {f["media"]:.2f} d'.replace('.', ','), '--'), (f['p99'], f'p99 = {f["p99"]:.1f} d'.replace('.', ','), '-')):
        ax.axvline(val, color='#3f77d1', ls=ls, lw=1.6, label=lab)
    ax.set_title(f'{etq}\n{d.size} rachas completas en 35 años', fontsize=10)
    ax.set_xlabel('duración de la racha (días)'); ax.legend(fontsize=9)
axs[0].set_ylabel('número de rachas (escala log)')
fig.suptitle(f'Celda {CELDA} (cercana a Santiago): de la lista de rachas a la media, la mediana y el p99', fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / 'figuras' / 'paso_3_3_ejemplo_celda.png', dpi=150)
