"""Regenera umbrales desde NetCDF y registra ajuste, NaN y ampliación del intervalo."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import json
import _run_pregunta7_bootstrap as b
ROOT=Path(__file__).resolve().parent

def main():
    a=b.open_aladin_historical()
    c=b.regrid_cr2met_to_aladin(b.open_cr2met_historical(),a)
    c,a=xr.align(b.normalize_daily_time(c),b.normalize_daily_time(a),join='inner')
    m=b.build_chile_mask_on_aladin_grid(a.lat.values,a.lon.values,b.load_chile_geometry())
    # El RR usa geometría administrativa; verificar y cubrir también sus celdas de borde.
    import cartopy.io.shapereader as sr
    from shapely.prepared import prep
    from shapely.geometry import Point
    covered=m.copy()
    records=sr.Reader(sr.natural_earth(resolution='10m',category='cultural',name='admin_1_states_provinces')).records()
    names=('coquimbo','higgins','araucan','los lagos')
    import unicodedata
    norm=lambda s: ''.join(x for x in unicodedata.normalize('NFKD',s.lower()) if not unicodedata.combining(x))
    for rec in records:
        if rec.attributes.get('admin')!='Chile' or not any(n in norm(rec.attributes.get('name','')) for n in names):continue
        geom=rec.geometry; pg=prep(geom)
        rm=np.array([pg.contains(Point(float(x),float(y))) or geom.touches(Point(float(x),float(y))) for x,y in zip(a.lon.values.ravel(),a.lat.values.ravel())]).reshape(m.shape)
        covered |= rm
    jj,ii=np.where(covered)
    av=a.values[:,jj,ii]; cv=c.values[:,jj,ii]
    ft=(cv>=1).mean(0)
    rr=[]; expanded=[]
    for k in range(jj.size):
        expanded.append(ft[k] < b.wet_fraction_1d(av[:,k],15)-b.BISECTION_TOL)
        rr.append(b.find_threshold_for_target_1d(av[:,k],ft[k]))
    rr=np.asarray(rr,dtype=np.float32)
    assert np.isfinite(rr).all(), 'Quedan celdas sin solución; no continuar silenciosamente.'
    local=(av>=rr).mean(0); glob=(av>=5.285).mean(0)
    full=np.full(m.shape,np.nan,np.float32);full[jj,ii]=rr
    (ROOT/'_pregunta7_outputs').mkdir(exist_ok=True)
    np.savez(ROOT/'_pregunta7_outputs/R0_local_map.npz',r0=full,chile=m,lat=a.lat.values,lon=a.lon.values,y=a.y.values,x=a.x.values)
    df=pd.DataFrame(dict(j=jj,i=ii,lat=a.lat.values[jj,ii],lon=a.lon.values[jj,ii],en_dominio=m[jj,ii],umbral_local_mm=rr,
        amplio_limite_15mm=expanded,cr2met_humeda_pct=100*ft,aladin_global_humeda_pct=100*glob,aladin_local_humeda_pct=100*local,
        error_local_pp=100*(local-ft),error_global_pp=100*(glob-ft),n_validos_cr2=np.isfinite(cv).sum(0),n_validos_aladin=np.isfinite(av).sum(0)))
    (ROOT/'resultados_tesis').mkdir(exist_ok=True)
    df.to_csv(ROOT/'resultados_tesis/calibracion_local_por_celda.csv',index=False)
    s=dict(periodo='1980-2014',referencia_cr2met_mm=1.0,umbral_futuro='mismo campo histórico; sin recalibrar',
        tolerancia_fraccion=b.BISECTION_TOL,celdas_dominio=int(m.sum()),celdas_cubiertas=int(covered.sum()),
        celdas_administrativas_fuera_mascara=int((covered&~m).sum()),ampliaron_busqueda=int(sum(expanded)),
        umbral_min=float(rr.min()),umbral_p10=float(np.percentile(rr,10)),umbral_mediana=float(np.median(rr)),
        umbral_p90=float(np.percentile(rr,90)),umbral_max=float(rr.max()),
        max_error_absoluto_pp=float(100*np.abs(local-ft).max()),error_medio_pp=float(100*(local-ft).mean()),
        celdas_fuera_tolerancia=int((np.abs(local-ft)>b.BISECTION_TOL+1e-12).sum()),
        dias_historicos=int(av.shape[0]),NaN_cr2met=int((~np.isfinite(cv)).sum()),NaN_aladin=int((~np.isfinite(av)).sum()))
    (ROOT/'resultados_tesis/calibracion_local.json').write_text(json.dumps(s,indent=2,ensure_ascii=False))
    print(json.dumps(s,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
