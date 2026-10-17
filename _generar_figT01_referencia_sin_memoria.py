"""Figura del Capítulo 2 (§2.1.3): PDF observada de rachas secas frente a referencias sin memoria.

CR2MET 1980-2014, umbral 1 mm, mismas reglas que §3.2 (se excluyen las rachas censuradas).
Para cada píxel de la ventana se calculan dos referencias y se suman ponderadas por su número de rachas,
porque mezclar píxeles con distinta frecuencia de días secos ya produce una cola más pesada que una
geométrica única:
  - días independientes: P(t) = (1 - q) q^(t-1), con q = fracción de días secos del píxel;
  - Markov de primer orden: igual, con q = P(seco mañana | seco hoy) del píxel.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from _run_pregunta7_bootstrap import load_chile_geometry, build_chile_mask_on_aladin_grid

ROOT = Path(__file__).resolve().parent
CACHE_CR2 = ROOT / '_test_regrillado' / 'cr2met_lineal_vs_area.npz'
OUT_PNG = ROOT / 'paper_figuras' / 'figT01_referencia_sin_memoria.png'
OUT_CSV = ROOT / 'paper_figuras' / 'figT01_referencia_sin_memoria.csv'
THRESHOLD = 1.0
N_EDGES = 26
BASINS = {
    'Loa': (-69.8, -68.2, -24.0, -21.5),
    'Maipo': (-71.8, -69.8, -34.5, -33.2),
    'Maule': (-72.8, -71.0, -36.5, -34.8),
    'Biobío': (-73.8, -71.5, -38.5, -36.5),
}


def pixel_spells(dry):
    dx = np.diff(np.r_[False, dry, False].astype(np.int8))
    s, e = np.where(dx == 1)[0], np.where(dx == -1)[0]
    keep = (s > 0) & (e < dry.size)
    return (e - s)[keep]


def log_bins(tmax):
    edges = np.unique(np.round(np.logspace(0, np.log10(tmax + 1), N_EDGES)).astype(int))
    return edges - 0.5


def binned_density(counts_by_t, edges):
    t = np.arange(counts_by_t.size)
    out = np.empty(edges.size - 1)
    for i in range(edges.size - 1):
        sel = (t > edges[i]) & (t < edges[i + 1])
        out[i] = counts_by_t[sel].sum() / (edges[i + 1] - edges[i])
    return out


def main():
    ds = xr.open_mfdataset(sorted(str(p) for p in (ROOT / 'pr1').glob('pr_CHP12_*_historical_*.nc')),
                           use_cftime=True)
    lat2d, lon2d = ds['lat'].values, ds['lon'].values
    mask = build_chile_mask_on_aladin_grid(lat2d, lon2d, load_chile_geometry())
    jj, ii = np.where(mask)
    pr = np.load(CACHE_CR2, mmap_mode='r')['lin']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.2))
    rows = []
    for ax, (name, (lo0, lo1, la0, la1)) in zip(axes.ravel(), BASINS.items()):
        sel = np.where((lon2d[jj, ii] >= lo0) & (lon2d[jj, ii] <= lo1)
                       & (lat2d[jj, ii] >= la0) & (lat2d[jj, ii] <= la1))[0]
        durations, pix = [], []
        for k in sel:
            dry = np.asarray(pr[:, k]) < THRESHOLD
            d = pixel_spells(dry)
            if d.size == 0:
                continue
            q_ind = dry.mean()
            q_mk = (dry[1:] & dry[:-1]).sum() / dry[:-1].sum()
            durations.append(d)
            pix.append((d.size, q_ind, q_mk))
        durations = np.concatenate(durations)
        tmax = int(durations.max())
        t = np.arange(tmax + 1)
        obs = np.bincount(durations, minlength=tmax + 1).astype(float)
        ref_ind = np.zeros(tmax + 1)
        ref_mk = np.zeros(tmax + 1)
        for n, q_i, q_m in pix:
            ref_ind[1:] += n * (1 - q_i) * q_i ** (t[1:] - 1)
            ref_mk[1:] += n * (1 - q_m) * q_m ** (t[1:] - 1)

        edges = log_bins(tmax)
        centers = np.sqrt(np.clip(edges[:-1], 0.5, None) * edges[1:])
        f_obs = binned_density(obs, edges)
        f_ind = binned_density(ref_ind, edges)
        f_mk = binned_density(ref_mk, edges)
        ok = f_obs > 0
        ax.loglog(centers[ok], f_obs[ok], 'k-o', ms=3.5, lw=1.6, label='CR2MET observado')
        ax.loglog(centers, np.where(f_ind > 1e-3, f_ind, np.nan), color='0.55', ls='--', lw=1.6,
                  label='Días independientes')
        ax.loglog(centers, np.where(f_mk > 1e-3, f_mk, np.nan), color='#2C7FB8', ls=':', lw=2.0,
                  label='Markov de primer orden')
        ax.set_title(f'{name} ({len(pix)} celdas)', fontsize=11)
        ax.set_xlabel('Duración t (días)')
        ax.set_ylabel('f(t) (rachas por día de duración)')
        ax.set_ylim(bottom=1e-2)
        ax.grid(True, which='major', alpha=0.3)

        for d_ref in (20, 60):
            n_obs = int((durations >= d_ref).sum())
            rows.append({
                'cuenca': name, 'D': d_ref, 'N_obs(t>=D)': n_obs,
                'N_indep(t>=D)': float(ref_ind[d_ref:].sum()),
                'N_markov1(t>=D)': float(ref_mk[d_ref:].sum()),
                'obs/markov1': n_obs / ref_mk[d_ref:].sum(),
            })
    axes[0, 0].legend(fontsize=9, loc='lower left')
    fig.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(df.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
