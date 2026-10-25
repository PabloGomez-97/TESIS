"""Frecuencias anuales y estacionales, usando exactamente las máscaras del RR."""
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import _run_pregunta5_aladin as rr
import _referencia_markov_rr_regiones as mk
from umbral_local import campo_umbral
ROOT=Path(__file__).resolve().parent

def main():
    a=rr.open_aladin_period(rr.HIST_START,rr.HIST_END)
    c=xr.open_mfdataset(str(ROOT/'pr/CR2MET_pr_v2.5_day_*.nc'),chunks={'time':365})['pr'].sel(time=slice(rr.HIST_START,rr.HIST_END)).interp(lat=a.lat,lon=a.lon)
    jj,ii,reg=mk.celdas_de_regiones(a.lat.values,a.lon.values)
    av=mk.a_celdas(a,jj,ii); cv=mk.a_celdas(c,jj,ii); t=campo_umbral().values[jj,ii]
    rows=[]
    for season in ('anual','mar-nov'):
        h=np.ones(av.shape[0],bool) if season=='anual' else np.isin(a.time.dt.month.values,np.arange(3,12))
        for k,name in enumerate(rr.REGION_SPECS):
            s=reg==k
            f=lambda x,thr:100*(x[h][:,s]>=thr).mean()
            ref=f(cv,1); fg=f(av,5.285);fl=f(av,t[s])
            rows.append(dict(region=name,estacion=season,celdas=int(s.sum()),CR2MET_pct=ref,global_pct=fg,local_pct=fl,exceso_global_pp=fg-ref,exceso_local_pp=fl-ref))
    pd.DataFrame(rows).to_csv(ROOT/'resultados_tesis/frecuencia_humeda_regiones_comparada.csv',index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
if __name__=='__main__':main()
