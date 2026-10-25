from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
DATA=BASE.parent/'temporal_umbral_por_celda_2026-10-24/ejecucion'
def generate(subdir,name,title):
 source=DATA/subdir/'results.json'
 data=json.loads(source.read_text())
 fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
 for ax,(region,rr),color in zip(axes.ravel(),data['curves'].items(),['#b22222','#d57500','#268428','#3476a7']):
  x=np.array([r['Umbral (dias)'] for r in rr]);y=np.array([r['RR'] if r['RR'] is not None else np.nan for r in rr],float)
  low=np.array([r['IC95 inferior'] if r['IC95 inferior'] is not None else np.nan for r in rr],float)
  high=np.array([r['IC95 superior'] if r['IC95 superior'] is not None else np.nan for r in rr],float)
  disc=np.array([r['Replicas descartadas'] for r in rr]);valid=np.isfinite(y)&(y>0)
  band=valid&(disc==0)&np.isfinite(low)&np.isfinite(high)&(low>0)
  ax.fill_between(x,low,high,where=band,interpolate=False,color=color,alpha=.19)
  ax.plot(x,np.where(valid,y,np.nan),color=color,lw=1.6)
  ax.scatter(x[valid&(disc==0)],y[valid&(disc==0)],color=color,s=24,zorder=3)
  ax.scatter(x[valid&(disc>0)],y[valid&(disc>0)],edgecolors=color,facecolors='white',s=34,zorder=3)
  ax.axhline(1,color='black',ls=':',lw=1.2);ax.axvline(20,color='.5',ls='--',lw=.8)
  ax.set_yscale('log');ax.set_xlim(3,62);ax.set_xticks([5,10,20,30,40,50,60])
  ax.set_title(region,fontsize=12,color=color);ax.set_xlabel('Duración mínima D (días)');ax.set_ylabel('RR de conteos (escala logarítmica)')
  ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
 fig.suptitle(title+'\n1980–2014 frente a 2040–2074 · inicio marzo–noviembre',fontsize=12)
 fig.savefig(BASE/'figuras'/name,dpi=240);plt.close(fig)
 return {'archivo':name,'fuente':str(source),'nota':'Redibujado sin alterar estimaciones; banda solo si no hay descartes; eje logarítmico sin recortar RR.'}
if __name__=='__main__':
 index=[generate('outputs_pregunta5_aladin','Figura_06.png','ALADIN SSP5-8.5 · umbral histórico por celda · CR2MET a 1 mm'),generate('outputs_pregunta5_aladin_1mm','Figura_S3.png','ALADIN SSP5-8.5 · umbral fijo de 1 mm en ALADIN'),generate('outputs_pregunta5_aladin_ssp370','Figura_S4.png','ALADIN SSP3-7.0 · umbral histórico por celda · CR2MET a 1 mm')]
 (BASE/'figuras/PROCEDENCIA_RR_REDIBUJADOS.json').write_text(json.dumps(index,ensure_ascii=False,indent=2))
