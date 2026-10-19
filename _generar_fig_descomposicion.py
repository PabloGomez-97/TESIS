"""Figura 10: el aumento de rachas largas frente al cambio en la frecuencia de dias humedos.

Panel superior: cambio relativo de la fraccion de dias humedos (R0 fijo) entre 2040-2074 (SSP5-8.5) y 1980-2014.
Panel inferior: RR(20) observado, RR esperado si solo cambia la frecuencia (Markov) y RR con umbral futuro que
conserva la frecuencia de dias humedos de cada region.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from geografia_tesis import NOMBRES_CORTOS, MACROZONA_DE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_tesis'
FIG = ROOT / 'paper_figuras'
COLOR = {'Norte árido': '#1D9E75', 'Norte semiárido': '#D85A30', 'Centro mediterráneo': '#534AB7', 'Sur templado': '#D4537E'}


def main():
    fr = pd.read_csv(OUT / 'umbral_frecuencia_regiones.csv')
    fr = fr[fr.escenario == 'ssp585'].set_index('region').loc[NOMBRES_CORTOS]
    cambio = (fr.F_fut_con_R0_pct / fr.F_hist_pct - 1) * 100
    rr = pd.read_csv(OUT / 'rr_ssp585.csv'); rr = rr[(rr.nivel == 'region') & (rr.D == 20)].set_index('unidad').loc[NOMBRES_CORTOS]
    rf = pd.read_csv(OUT / 'rr_ssp585_RfrecReg.csv'); rf = rf[(rf.nivel == 'region') & (rf.D == 20)].set_index('unidad').loc[NOMBRES_CORTOS]
    mk = pd.read_csv(OUT / 'descomposicion_markov_ssp585.csv'); mk = mk[(mk.nivel == 'region') & (mk.D == 20)].set_index('unidad').loc[NOMBRES_CORTOS]
    x = np.arange(len(NOMBRES_CORTOS))
    cols = [COLOR[MACROZONA_DE[n]] for n in NOMBRES_CORTOS]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 8), sharex=True, gridspec_kw={'height_ratios': [0.9, 1.4]})
    a1.bar(x, cambio.values, color=cols)
    a1.axhline(0, color='gray', lw=0.8)
    a1.set_ylabel('Cambio en la fracción de días\nhúmedos con R₀ fijo (%)')
    a1.set_title('Menos días húmedos desde Atacama hacia el sur, y más rachas largas', fontsize=10)
    a1.grid(alpha=0.25, axis='y')
    a2.axhline(1, color='gray', ls=':')
    for xi in x:
        a2.plot([xi - 0.18] * 2, [rr.IC_low.iloc[xi], rr.IC_high.iloc[xi]], color='black', lw=1.6)
        a2.plot([xi + 0.18] * 2, [rf.IC_low.iloc[xi], rf.IC_high.iloc[xi]], color='#0C447C', lw=1.6)
    a2.plot(x - 0.18, rr.RR.values, 'o', color='black', ms=7, label='Observado (R₀ fijo) con IC95 %')
    a2.plot(x, mk.RR_solo_frec.values, 's', color='#BA7517', ms=7, label='Esperado si solo cambia la frecuencia (Markov)')
    a2.plot(x + 0.18, rf.RR.values, '^', color='#0C447C', ms=7, label='Prueba de control: futuro con la misma frecuencia de días húmedos, con IC95 %')
    a2.set_yscale('log'); a2.set_yticks([0.6, 0.8, 1, 1.5, 2, 3, 5, 8]); a2.set_yticklabels(['0,6', '0,8', '1', '1,5', '2', '3', '5', '8'])
    a2.set_ylabel('RR(20)')
    a2.legend(fontsize=8, loc='upper left'); a2.grid(alpha=0.25)
    a2.set_xticks(x); a2.set_xticklabels(NOMBRES_CORTOS, rotation=45, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'fig10_frecuencia_vs_agrupacion.png', dpi=170, facecolor='white')


if __name__ == '__main__':
    main()
