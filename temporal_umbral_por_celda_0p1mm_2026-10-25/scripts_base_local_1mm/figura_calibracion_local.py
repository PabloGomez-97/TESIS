"""Figura de calibración apropiada para un campo de umbrales; conserva la figura global de referencia."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
p=R/'paper_figuras';p.mkdir(exist_ok=True)
a=R.parent/'referencia_global/paper_figuras/fig03_calibracion_R0_ftau.png'
(p/'fig03_referencia_umbral_global.png').write_bytes(a.read_bytes())
d=pd.read_csv(R/'resultados_tesis/calibracion_local_por_celda.csv')
fig,axes=plt.subplots(1,3,figsize=(14,4.5))
axes[0].hist(d.umbral_local_mm,bins=40,color='#c16436');axes[0].axvline(5.285,ls='--',color='#436e9b',label='Global = 5,285');axes[0].set(xlabel='Umbral local (mm/día)',ylabel='Celdas');axes[0].legend()
for ax,col,title in ((axes[1],'aladin_global_humeda_pct','Umbral global'),(axes[2],'aladin_local_humeda_pct','Umbral local')):
 ax.scatter(d.cr2met_humeda_pct,d[col],s=3,alpha=.3,color='#436e9b' if col=='aladin_global_humeda_pct' else '#c16436');ax.plot([0,100],[0,100],':',color='black');ax.set(xlim=(0,100),ylim=(0,100),xlabel='CR2MET: días húmedos (%)',ylabel='ALADIN: días húmedos (%)',title=title);ax.set_aspect('equal')
fig.suptitle('Calibración histórica por celda · 1980–2014 · CR2MET a 1 mm/día');fig.tight_layout();fig.savefig(p/'fig03_calibracion_R0_ftau.png',dpi=200);plt.close(fig)
