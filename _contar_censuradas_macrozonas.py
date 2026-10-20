import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
from _extraer_rachas_dominio import open_aladin, open_cr2met, a_celdas, HIST, FUT, R0
from geografia_tesis import NOMBRES_CORTOS, MACROZONAS, MACROZONA_DE
g=np.load('resultados_tesis/grilla.npz'); jj,ii=g['jj'],g['ii']
reg=g['region_idx'][jj,ii].astype(int)
zona=np.full(reg.shape,-1)
for k,n in enumerate(NOMBRES_CORTOS): zona[reg==k]=MACROZONAS.index(MACROZONA_DE[n])
ref=open_aladin(HIST,None)
conj={'cr2met_hist':(open_cr2met(HIST,ref['lat'],ref['lon']),1.0),'aladin_hist':(ref,R0),'aladin_ssp585':(open_aladin(FUT,'ssp585'),R0)}
rows=[]
for nom,(da,thr) in conj.items():
    pr=a_celdas(da,jj,ii); n=pr.shape[0]
    dry=(pr<thr)
    pad=np.zeros((pr.shape[1],n+2),dtype=np.int8); pad[:,1:-1]=dry.T
    dx=np.diff(pad.ravel()); s=np.flatnonzero(dx==1)+1; e=np.flatnonzero(dx==-1)+1
    col=s//(n+2); fs=s%(n+2)-1; fe=e%(n+2)-1
    cens=(fs==0)|(fe==n)
    for z,m in enumerate(MACROZONAS):
        sel=zona[col]==z
        rows.append(dict(conjunto=nom,macrozona=m,rachas_totales=int(sel.sum()),censuradas=int((cens&sel).sum()),pct=round(100*(cens&sel).sum()/sel.sum(),2)))
    sel=zona[col]>=0
    rows.append(dict(conjunto=nom,macrozona='14 regiones',rachas_totales=int(sel.sum()),censuradas=int((cens&sel).sum()),pct=round(100*(cens&sel).sum()/sel.sum(),2)))
df=pd.DataFrame(rows); df.to_csv('resultados_tesis/censuradas_por_macrozona.csv',index=False); print(df.to_string(index=False))
