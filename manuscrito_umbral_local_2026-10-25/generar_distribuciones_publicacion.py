from pathlib import Path
import pickle,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
source=BASE.parent/'temporal_umbral_por_celda_2026-10-24/ejecucion/_pregunta9_outputs/pdf_durations_i_global.pkl'
data=pickle.loads(source.read_bytes())
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
checks=[]
for ax,(name,series),letter in zip(axes.ravel(),data.items(),'abcd'):
 hi=int(max(np.max(d) for d in series.values() if len(d)))
 edges=np.unique(np.round(np.logspace(0,np.log10(hi+1),26)))-.5
 centers=np.sqrt((edges[:-1]+.5)*(edges[1:]-.5))
 for (label,values),color,ls in zip(series.items(),['black','steelblue','crimson'],['-','-','--']):
  counts,_=np.histogram(values,bins=edges);heights=counts/np.diff(edges)
  assert counts.sum()==len(values)
  ax.plot(centers,np.where(heights>0,heights,np.nan),color=color,ls=ls,lw=1.5,label=f'{label} (n={len(values):,})')
  checks.append({'ventana':name,'serie':label,'rachas':len(values),'integral':float(np.sum(heights*np.diff(edges))),'bins':len(edges)-1})
 ax.set(xscale='log',yscale='log',xlabel='Duración de la racha (días)',ylabel='Rachas por unidad de duración')
 ax.set_title(f'({letter}) Ventana {name}',fontsize=12)
 ax.legend(fontsize=8);ax.grid(alpha=.2,which='both')
fig.suptitle('CR2MET a 1 mm · ALADIN con umbral histórico por celda\nHistórico 1980–2014 y futuro 2040–2074 bajo SSP5-8.5',fontsize=12)
fig.savefig(BASE/'figuras/Figura_05.png',dpi=240);plt.close(fig)
(BASE/'qa/control_distribuciones_redibujadas.json').write_text(json.dumps({'fuente':str(source),'controles':checks},ensure_ascii=False,indent=2))
