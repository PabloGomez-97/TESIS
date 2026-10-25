"""Comparación trazable: umbral local a 1 mm frente a umbral local a 0,1 mm."""
from pathlib import Path
import sys,json,pickle,difflib,hashlib
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;ROOT=Path(json.loads((R/'configuracion.json').read_text())['repo_origen'])
A=R/'referencia_local_1mm';B=R/'ejecucion';G=R/'referencia_global_0p1mm';O=R/'comparativa';O.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'reproducibilidad'))
from ejecutar import OUTPUT_DIRS,manifest,compare_file,equal_content

def outputs(root):
    dirs=set(OUTPUT_DIRS)|{'outputs_pregunta5_aladin_0.1mm'}
    return {p.relative_to(root).as_posix():p for d in dirs if (root/d).exists() for p in (root/d).rglob('*') if p.is_file() and p.suffix in {'.csv','.json','.npz','.png','.pkl'}}
def rd(root,rel):return pd.read_csv(root/rel)
def js(root,rel):return json.loads((root/rel).read_text())
def out(n,d):d.to_csv(O/(n+'.csv'),index=False);return d
checks=[]
def ck(name,condition):checks.append(dict(control=name,passed=bool(condition)))
def pair(x,y,keys):
    z=x.merge(y,on=keys,suffixes=('_1mm','_0p1mm'),validate='one_to_one')
    for c in x.select_dtypes(include='number').columns:
        if c not in keys and c in y:z[c+'_diferencia']=z[c+'_0p1mm']-z[c+'_1mm']
    return z

def pfile(rel,keys):return pair(rd(A,rel),rd(B,rel),keys)
def fmt(x):
    if isinstance(x,(bool,np.bool_)):return 'sí' if x else 'no'
    if isinstance(x,(int,np.integer)):return str(x)
    if isinstance(x,(float,np.floating)):
        if not np.isfinite(x):return '—'
        if abs(x)<.0005:x=0
        return f'{x:.3f}'
    return str(x)
def md(d):
    if d.empty:return 'No hay casos.'
    return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))

# Calibración: misma celda en ambas referencias.
ca=rd(A,'resultados_tesis/calibracion_local_por_celda.csv');cb=rd(B,'resultados_tesis/calibracion_local_por_celda.csv')
cal=out('01_calibracion_por_celda',pair(ca,cb,['j','i','lat','lon','en_dominio']))
cal1=js(A,'resultados_tesis/calibracion_local.json');cal01=js(B,'resultados_tesis/calibracion_local.json')
ck('Mismas 3587 celdas de calibración',len(cal)==3587 and cal01['celdas_dominio']==3587)
ck('Frecuencia húmeda CR2MET no disminuye al bajar referencia',np.all(cal.cr2met_humeda_pct_0p1mm>=cal.cr2met_humeda_pct_1mm-1e-9))
ck('Umbral local no aumenta al bajar referencia',np.all(cal.umbral_local_mm_0p1mm<=cal.umbral_local_mm_1mm+1e-7))
ck('Todas las celdas locales cumplen tolerancia',cal01['celdas_fuera_tolerancia']==0)
oldfield=np.load(G/'_pregunta7_outputs/R0_local_map.npz');newfield=np.load(B/'_pregunta7_outputs/R0_local_map.npz');oldvalid=oldfield['chile']&np.isfinite(oldfield['r0'])
ck('Campo local 0,1 previo preservado en sus celdas válidas',np.array_equal(oldfield['r0'][oldvalid],newfield['r0'][oldvalid]))
fa=rd(A,'resultados_tesis/frecuencia_humeda_ventanas.csv').rename(columns={'CR2MET_1mm_pct':'CR2MET_pct','ALADIN_1mm_pct':'ALADIN_igual_pct','exceso_1mm_pp':'exceso_igual_pp'})
fb=rd(B,'resultados_tesis/frecuencia_humeda_ventanas.csv').rename(columns={'CR2MET_0p1mm_pct':'CR2MET_pct','ALADIN_0p1mm_pct':'ALADIN_igual_pct','exceso_0p1mm_pp':'exceso_igual_pp'})
freq=out('02_frecuencia_ventanas',pair(fa,fb,['zona']))
freg=out('03_frecuencia_regiones',pfile('resultados_tesis/frecuencia_humeda_regiones_comparada.csv',['region','estacion']))
# Mapas, métricas y descomposición: cambia tanto ALADIN como CR2MET.
za=np.load(A/'_pregunta7_outputs/metrics_integrado_cache.npz');zb=np.load(B/'_pregunta7_outputs/metrics_integrado_cache.npz');m=za['chile_mask']
ck('Máscara espacial preservada',np.array_equal(m,zb['chile_mask']))
srows=[];crows=[]
for metric in ('mean','t99'):
    c1,a1=za[metric+'_cr2'],za[metric+'_ala'];c0,a0=zb[metric+'_cr2'],zb[metric+'_ala']
    v1=m&np.isfinite(c1)&np.isfinite(a1);v0=m&np.isfinite(c0)&np.isfinite(a0);common=v1&v0
    b1=np.load(A/f'resultados_tesis/bootstrap_por_celda_C_umbral_local_{metric}.npz');b0=np.load(B/f'resultados_tesis/bootstrap_por_celda_C_umbral_local_{metric}.npz')
    for sample,q1,q0 in (('muestra propia',v1,v0),('mismas celdas',common,common)):
        srows.append(dict(metrica=metric,muestra=sample,celdas_1mm=int(q1.sum()),celdas_0p1mm=int(q0.sum()),
            CR2MET_1mm=float(c1[q1].mean()),CR2MET_0p1mm=float(c0[q0].mean()),ALADIN_1mm=float(a1[q1].mean()),ALADIN_0p1mm=float(a0[q0].mean()),
            sesgo_1mm=float((a1-c1)[q1].mean()),sesgo_0p1mm=float((a0-c0)[q0].mean()),
            MAE_1mm=float(np.abs(a1-c1)[q1].mean()),MAE_0p1mm=float(np.abs(a0-c0)[q0].mean()),
            error_rel_absoluto_1mm_pct=float(100*(np.abs(a1-c1)/c1)[q1].mean()),error_rel_absoluto_0p1mm_pct=float(100*(np.abs(a0-c0)/c0)[q0].mean()),
            menor_sig_1mm_pct=float(100*b1['mas_corto'][q1].mean()),menor_sig_0p1mm_pct=float(100*b0['mas_corto'][q0].mean()),
            mayor_sig_1mm_pct=float(100*b1['mas_largo'][q1].mean()),mayor_sig_0p1mm_pct=float(100*b0['mas_largo'][q0].mean()),
            cambio_ALADIN_comunes=float((a0-a1)[common].mean()),cambio_CR2MET_comunes=float((c0-c1)[common].mean()),
            cambio_sesgo_comunes=float(((a0-c0)-(a1-c1))[common].mean())))
    ck('Identidad cambio sesgo = cambio ALADIN − cambio CR2MET '+metric,
        np.allclose(((a0-c0)-(a1-c1))[common],((a0-a1)-(c0-c1))[common],atol=1e-3,rtol=0))
    for j,i in zip(*np.where(m)):
        crows.append(dict(metrica=metric,j=j,i=i,lat=za['lat'][j,i],lon=za['lon'][j,i],valida_1mm=bool(v1[j,i]),valida_0p1mm=bool(v0[j,i]),
            CR2MET_1mm=float(c1[j,i]),CR2MET_0p1mm=float(c0[j,i]),ALADIN_1mm=float(a1[j,i]),ALADIN_0p1mm=float(a0[j,i]),
            sesgo_1mm=float(a1[j,i]-c1[j,i]),sesgo_0p1mm=float(a0[j,i]-c0[j,i]),
            significativo_1mm=bool(b1['significativo'][j,i]),significativo_0p1mm=bool(b0['significativo'][j,i])))
spatial=out('04_resumen_espacial',pd.DataFrame(srows));out('05_metricas_por_celda',pd.DataFrame(crows))
bias=out('06_sesgo_ventanas',pfile('resultados_tesis/sesgo_por_ventana.csv',['metrica','zona']))
tab=out('07_tabla3_completa',pfile('resultados_tesis/tabla3_estadisticos_ventanas.csv',['ventana','producto']))
frows=[]
for root,label in ((A,'local 1 mm'),(B,'local 0,1 mm')):
    t=rd(root,'resultados_tesis/tabla3_estadisticos_ventanas.csv')
    for v in t.ventana.unique():
        s=t[t.ventana==v].set_index('producto');h=s.loc['ALADIN hist'];f=s.loc['ALADIN fut']
        frows.append(dict(ventana=v,referencia=label,N_hist=int(h.n_spells),N_fut=int(f.n_spells),cambio_N_pct=100*(f.n_spells/h.n_spells-1),
            media_hist=h.media,media_fut=f.media,cambio_media_dias=f.media-h.media,cambio_media_pct=100*(f.media/h.media-1),
            p99_hist=h.p99,p99_fut=f.p99,cambio_p99_dias=f.p99-h.p99,cambio_p99_pct=100*(f.p99/h.p99-1)))
future=out('08_cambios_futuros_ventanas',pd.DataFrame(frows))
# Todos los RR: dos escenarios y la sensibilidad con igual umbral físico en ambos productos.
def decision(lo,hi):
    if not np.isfinite(lo) or not np.isfinite(hi):return 'no definido'
    return 'aumento' if lo>1 else ('disminución' if hi<1 else 'IC incluye 1')
rrrows=[];curverows=[]
for sa,sb,scenario in (('','','SSP5-8.5'),('_ssp370','_ssp370','SSP3-7.0'),('_1mm','_0.1mm','igual umbral escalar')):
    ra=js(A,f'outputs_pregunta5_aladin{sa}/results.json');rb=js(B,f'outputs_pregunta5_aladin{sb}/results.json')
    for x,y in zip(ra['rr_table'],rb['rr_table']):
        assert x['Region']==y['Region'];r=dict(escenario=scenario,region=x['Region'])
        for label,d in (('1mm',x),('0p1mm',y)):
            for key,name in [('N_hist','N_hist'),('N_fut','N_fut'),('RR observado','RR'),('IC95 inferior','IC_inf'),('IC95 superior','IC_sup'),('Replicas descartadas (de 1000)','descartadas')]:r[name+'_'+label]=d[key]
            r['decision_'+label]=decision(d['IC95 inferior'],d['IC95 superior'])
        r['cambio_RR']=r['RR_0p1mm']-r['RR_1mm'];rrrows.append(r)
    for region in ra['curves']:
        for x,y in zip(ra['curves'][region],rb['curves'][region]):
            r=dict(escenario=scenario,region=region,D=x['Umbral (dias)'])
            for label,d in (('1mm',x),('0p1mm',y)):
                for key in ('RR','IC95 inferior','IC95 superior','N1','N2','Total1','Total2','Replicas descartadas'):r[key+'_'+label]=d[key]
                r['decision_'+label]=decision(d['IC95 inferior'],d['IC95 superior'])
            r['cambia_decision']=r['decision_1mm']!=r['decision_0p1mm'];curverows.append(r)
rr=out('09_RR_D20',pd.DataFrame(rrrows));curves=out('10_RR_todas_duraciones',pd.DataFrame(curverows))
pers=out('11_persistencia',pfile('resultados_tesis/contraste_persistencia_regiones.csv',['region','meses']))
markov=out('12_markov',pfile('resultados_tesis/descomposicion_markov_rr_regiones.csv',['region','D']))
cs=[]
for root in (A,B):
    c=rd(root,'_test_regrillado/rachas_censuradas_por_cuenca.csv')
    c['serie']=c.serie.str.replace('CR2MET 1980-2014 (1 mm)','CR2MET hist',regex=False).str.replace('CR2MET 1980-2014 (0.1 mm)','CR2MET hist',regex=False);cs.append(c)
cens=out('13_censuradas',pair(*cs,['serie','cuenca']))
mins=out('14_minimo_rachas',pfile('resultados_tesis/sensibilidad_minimo_rachas.csv',['minimo_rachas']))
regrid=out('15_regrillado',pfile('_test_regrillado/mapas_lineal_vs_area.csv',['cr2met','criterio','metrica']))
# Comparación adicional 2×2: separar la referencia de la escala de calibración.
configurations=[(R/'referencia_global_1mm','1 mm','global'),(A,'1 mm','local'),(G,'0,1 mm','global'),(B,'0,1 mm','local')]
factor_hist=[];factor_rr=[]
for metric in ('mean','t99'):
    maps=[np.load(root/'_pregunta7_outputs/metrics_integrado_cache.npz') for root,_,_ in configurations]
    valids=[z['chile_mask']&np.isfinite(z[metric+'_cr2'])&np.isfinite(z[metric+'_ala']) for z in maps]
    same=np.logical_and.reduce(valids)
    for (root,reference,method),z,own in zip(configurations,maps,valids):
        c,a=z[metric+'_cr2'],z[metric+'_ala']
        for label,v in (('muestra propia',own),('mismas celdas de las cuatro versiones',same)):
            factor_hist.append(dict(referencia=reference,calibracion=method,metrica=metric,muestra=label,celdas=int(v.sum()),CR2MET=float(c[v].mean()),ALADIN=float(a[v].mean()),sesgo=float((a-c)[v].mean()),MAE=float(np.abs(a-c)[v].mean())))
for root,reference,method in configurations:
    for suffix,scenario in (('','SSP5-8.5'),('_ssp370','SSP3-7.0')):
        result=js(root,f'outputs_pregunta5_aladin{suffix}/results.json')
        for region,items in result['curves'].items():
            for item in items:
                factor_rr.append(dict(referencia=reference,calibracion=method,escenario=scenario,region=region,D=item['Umbral (dias)'],N_hist=item['N1'],N_fut=item['N2'],RR=item['RR'],IC_inf=item['IC95 inferior'],IC_sup=item['IC95 superior'],descartadas=item['Replicas descartadas'],decision=decision(item['IC95 inferior'],item['IC95 superior'])))
factor_hist=out('17_comparacion_cuatro_versiones_historico',pd.DataFrame(factor_hist))
factor_rr=out('18_comparacion_cuatro_versiones_RR',pd.DataFrame(factor_rr))

# Controles respecto del global 0,1 previo: no comparar contra 1 mm como si fueran el mismo experimento.
for metric in ('mean','t99'):
    for criterion in ('A_opcion_B_integrado','B_mismo_umbral'):
        x=np.load(G/f'resultados_tesis/bootstrap_por_celda_{criterion}_{metric}.npz');y=np.load(B/f'resultados_tesis/bootstrap_por_celda_{criterion}_{metric}.npz')
        ck('Control histórico global 0,1 previo idéntico '+criterion+'/'+metric,all(np.array_equal(x[k],y[k],equal_nan=True) for k in x.files))
    x=np.load(G/f'resultados_tesis/bootstrap_por_celda_C_umbral_local_{metric}.npz');y=np.load(B/f'resultados_tesis/bootstrap_por_celda_C_umbral_local_{metric}.npz');v=x['valida']
    ck('Local 0,1 previo coincide en celdas previamente válidas '+metric,all(np.array_equal(x[k][v],y[k][v],equal_nan=True) for k in x.files))
ck('RR escalar 0,1 idéntico al cálculo previo',equal_content(js(G,'outputs_pregunta5_aladin_0.1mm/results.json'),js(B,'outputs_pregunta5_aladin_0.1mm/results.json')))
ck('Acumulados y estacionalidad sin cambios',rd(A,'resultados_tesis/estacionalidad_marnov_regiones.csv').equals(rd(B,'resultados_tesis/estacionalidad_marnov_regiones.csv')))
oldcache=np.load(A/'_test_regrillado/cr2met_lineal_vs_area.npz');newcache=np.load(B/'_test_regrillado/cr2met_lineal_vs_area.npz')
ck('Caché float64 conserva los valores a la precisión float32 previa',all(np.array_equal(oldcache[k],newcache[k].astype(oldcache[k].dtype),equal_nan=True) for k in oldcache.files))
precision={}
for version in ('lin','area'):
    oldvalues=oldcache[version];newvalues=newcache[version]
    changed_days=np.zeros(oldvalues.shape[1],dtype=int)
    for start in range(0,oldvalues.shape[0],365):
        changed_days+=((oldvalues[start:start+365]>=0.1)!=(newvalues[start:start+365]>=0.1)).sum(0)
    precision[version]=dict(dias_celda_reclasificados=int(changed_days.sum()),celdas_afectadas=int((changed_days>0).sum()))
    del oldvalues,newvalues
precision['nota']='Clasificación float64 frente a caché float32 previa a 0,1 mm; se repitieron regrillado, censura y T01.'
(R/'precision_auxiliar.json').write_text(json.dumps(precision,indent=2,ensure_ascii=False))
cons=js(R,'controles_consistencia.json');integ=js(R,'integridad.json')
ck('Controles internos de la cadena',cons['passed']);ck('Todos los análisis previos y NetCDF preservados',integ['passed'])
# Inventario canónico: sensibilidad escalar 1 mm y 0,1 mm se empareja por función.
def canonical(root):
    return {k.replace('outputs_pregunta5_aladin_1mm/','outputs_pregunta5_aladin_igual/').replace('outputs_pregunta5_aladin_0.1mm/','outputs_pregunta5_aladin_igual/').replace('figS_rr_curves_aladin_1mm.png','figS_rr_curves_aladin_igual.png').replace('figS_rr_curves_aladin_0.1mm.png','figS_rr_curves_aladin_igual.png'):p for k,p in outputs(root).items()}
xa,xb=canonical(A),canonical(B);inv=[]
for k in sorted(set(xa)|set(xb)):
    if k not in xa:result=dict(equal=False,method='nuevo')
    elif k not in xb:result=dict(equal=False,method='faltante')
    else:result=compare_file(xa[k],xb[k])
    inv.append(dict(archivo=k,igual=result['equal'],metodo=result['method']))
inv=out('16_inventario_salidas',pd.DataFrame(inv))
(R/'controles_comparativa.json').write_text(json.dumps(dict(passed=all(x['passed'] for x in checks),checks=checks),indent=2,ensure_ascii=False))
(R/'scripts_ejecutados_sha256.json').write_text(json.dumps(manifest({p.name:p for p in B.glob('*.py')}),indent=2))
diffs=[]
for p in (R/'scripts_base_local_1mm').glob('*.py'):
    diffs.extend(difflib.unified_diff(p.read_text().splitlines(True),(B/p.name).read_text().splitlines(True),fromfile='local_1mm/'+p.name,tofile='local_0p1mm/'+p.name))
(R/'cambios_codigo.diff').write_text(''.join(diffs))
# Figuras comparativas: etiquetas identifican ambos experimentos como locales.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});colors=['#436e9b','#c16436']
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
common_bins=np.linspace(0,max(cal.umbral_local_mm_1mm.max(),cal.umbral_local_mm_0p1mm.max()),41)
axes[0].hist(cal.umbral_local_mm_1mm,bins=common_bins,histtype='step',lw=2,color=colors[0],label='Referencia 1 mm');axes[0].hist(cal.umbral_local_mm_0p1mm,bins=common_bins,histtype='step',lw=2,color=colors[1],label='Referencia 0,1 mm');axes[0].set(xlabel='Umbral local de ALADIN (mm/día)',ylabel='Celdas');axes[0].legend()
axes[1].scatter(cal.umbral_local_mm_1mm,cal.umbral_local_mm_0p1mm,s=4,alpha=.35,color=colors[1]);hi=max(cal.umbral_local_mm_1mm.max(),cal.umbral_local_mm_0p1mm.max());axes[1].plot([0,hi],[0,hi],':',color='black');axes[1].set(xlabel='Umbral local con referencia 1 mm',ylabel='Umbral local con referencia 0,1 mm');axes[1].set_aspect('equal')
fig.suptitle('Calibración histórica por celda: dos referencias de CR2MET');fig.tight_layout();fig.savefig(O/'comparacion_umbrales.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
for ax,(region,d) in zip(axes.ravel(),curves[curves.escenario=='SSP5-8.5'].groupby('region',sort=False)):
    for label,color,title in zip(('1mm','0p1mm'),colors,('Local / CR2MET 1 mm','Local / CR2MET 0,1 mm')):
        x=d.D.to_numpy();y=d['RR_'+label].to_numpy(float);lo=d['IC95 inferior_'+label].to_numpy(float);hi=d['IC95 superior_'+label].to_numpy(float)
        ax.plot(x,y,'o-',ms=3,label=title,color=color);good=(d['Replicas descartadas_'+label].to_numpy()==0)&np.isfinite(lo)&np.isfinite(hi)&(lo>0);ax.fill_between(x,lo,hi,where=good,color=color,alpha=.14)
    ax.set_yscale('log');ax.axhline(1,color='k',ls=':');ax.axvline(20,color='0.6',ls='--',lw=.8);ax.set(title=region,ylabel='RR de conteos');ax.legend(fontsize=8);ax.grid(alpha=.15)
for ax in axes[-1]:ax.set_xlabel('Duración mínima D (días)')
fig.suptitle('SSP5-8.5 · IC95 por años, solo sin réplicas descartadas');fig.tight_layout();fig.savefig(O/'comparacion_RR.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for ax,metric in zip(axes,('media','p99')):
    d=tab[tab.producto=='ALADIN hist'];x=np.arange(len(d));w=.35
    ax.bar(x-w/2,d[metric+'_sesgo_rel_pct_1mm'],w,color=colors[0],label='Local / CR2MET 1 mm');ax.bar(x+w/2,d[metric+'_sesgo_rel_pct_0p1mm'],w,color=colors[1],label='Local / CR2MET 0,1 mm');ax.axhline(0,color='k',lw=.7);ax.set_xticks(x,d.ventana);ax.set(title='Media' if metric=='media' else 'p99',ylabel='Sesgo relativo del pool histórico (%)');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(O/'comparacion_sesgos.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for ax,metric in zip(axes,('mean','t99')):
    d=spatial[(spatial.metrica==metric)&(spatial.muestra=='mismas celdas')].iloc[0]
    vals=[d.CR2MET_1mm,d.ALADIN_1mm,d.CR2MET_0p1mm,d.ALADIN_0p1mm]
    ax.bar(np.arange(4),vals,color=[colors[0],colors[0],colors[1],colors[1]],alpha=.8);ax.set_xticks(np.arange(4),['CR2MET\n1 mm','ALADIN\nref. 1 mm','CR2MET\n0,1 mm','ALADIN\nref. 0,1 mm']);ax.set(title='Media por celda' if metric=='mean' else 'p99 por celda',ylabel='Promedio espacial (días)')
fig.suptitle('Comparación sobre las mismas celdas: cambian referencia y modelo');fig.tight_layout();fig.savefig(O/'comparacion_duraciones.png',dpi=180);plt.close(fig)
# Mapa comparativo de umbrales, con escala común para ambos campos.
import cartopy.crs as ccrs
import cartopy.feature as cfeature
z1=np.load(A/'_pregunta7_outputs/R0_local_map.npz');z0=np.load(B/'_pregunta7_outputs/R0_local_map.npz')
t1=np.where(z1['chile'],z1['r0'],np.nan);t0=np.where(z0['chile'],z0['r0'],np.nan);delta=t0-t1
fig,axes=plt.subplots(1,3,figsize=(11,10),subplot_kw={'projection':ccrs.PlateCarree()})
vmax=float(np.ceil(max(np.nanmax(t1),np.nanmax(t0))));dmin=float(np.floor(np.nanmin(delta)))
for ax,data,title,is_delta in zip(axes,(t1,t0,delta),('Referencia CR2MET 1 mm','Referencia CR2MET 0,1 mm','Diferencia: 0,1 mm − 1 mm'),(False,False,True)):
    ax.set_extent([-76,-66,-46.5,-17],crs=ccrs.PlateCarree());ax.add_feature(cfeature.COASTLINE.with_scale('10m'),linewidth=.5);ax.add_feature(cfeature.BORDERS.with_scale('10m'),linewidth=.4,linestyle=':')
    im=ax.pcolormesh(z1['lon'],z1['lat'],data,shading='auto',cmap='PuBu_r' if is_delta else 'YlOrRd',vmin=dmin if is_delta else 0,vmax=0 if is_delta else vmax,transform=ccrs.PlateCarree())
    ax.set_title(title,fontsize=10);gl=ax.gridlines(draw_labels=True,linewidth=.3,alpha=.3);gl.top_labels=gl.right_labels=False
    fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.025,pad=.04).set_label('Δ umbral (mm/día)' if is_delta else 'Umbral de ALADIN (mm/día)')
fig.suptitle('Campos históricos locales: misma escala en los dos primeros mapas',y=.97);fig.savefig(O/'comparacion_mapas_umbrales.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Informe: datos completos en CSV, tablas principales legibles.
P=[]
def section(n,title,text):P.append(f'## {n}. {title}\n\n{text}\n')
def compact_rr(d):
    return pd.DataFrame([{'región':x.region,'RR / 1 mm':f'{x.RR_1mm:.3f} [{x.IC_inf_1mm:.3f}; {x.IC_sup_1mm:.3f}]','resultado / 1 mm':x.decision_1mm,'RR / 0,1 mm':f'{x.RR_0p1mm:.3f} [{x.IC_inf_0p1mm:.3f}; {x.IC_sup_0p1mm:.3f}]','resultado / 0,1 mm':x.decision_0p1mm} for x in d.itertuples()])
main=rr[rr.escenario=='SSP5-8.5'];ms=spatial[(spatial.muestra=='mismas celdas')&(spatial.metrica=='mean')].iloc[0]
P.append('# Comparativa completa: umbral local por celda con CR2MET a 1 y 0,1 mm/día\n\nFecha: 25 de octubre de 2026. Todos los análisis anteriores se conservan.\n')
section(1,'Resultado principal y alcance',
f"En las mismas {int(ms.celdas_1mm)} celdas, el sesgo de la duración media pasa de {ms.sesgo_1mm:.3f} a {ms.sesgo_0p1mm:.3f} días y el error absoluto medio pasa de {ms.MAE_1mm:.3f} a {ms.MAE_0p1mm:.3f} días. Para rachas ≥20 días bajo SSP5-8.5, el número de regiones con aumento significativo pasa de {int((main.decision_1mm=='aumento').sum())} con la referencia de 1 mm a {int((main.decision_0p1mm=='aumento').sum())} con la de 0,1 mm.\n\n"
'**La comparación principal es local contra local.** Se repitió toda la cadena científica anterior desde NetCDF, sustituyendo la referencia CR2MET de 1 por 0,1 mm/día y recalibrando ALADIN por celda en el histórico. No se dividieron los umbrales anteriores por diez. El campo histórico de cada experimento se mantuvo fijo en el futuro.\n\n'
'**Esta comparación cambia la definición de racha seca.** Al bajar el umbral de CR2MET también cambia la referencia con la que se evalúa ALADIN. Un MAE menor no demostraría por sí solo que una definición sea mejor ni que el modelo haya mejorado físicamente. Se separan los cambios de CR2MET, los de ALADIN y los de sus diferencias.')
section(2,'Diseño preservado y versiones conservadas',md(pd.DataFrame([
['Referencia anterior','CR2MET ≥1 mm/día como día húmedo; ALADIN con campo local correspondiente.'],
['Referencia nueva','CR2MET ≥0,1 mm/día como día húmedo; ALADIN con campo local recalibrado.'],
['Períodos','1980–2014 / 2040–2074; 35 años por período.'],
['Datos y unidades','Mismos NetCDF; ALADIN ×86 400 a mm/día.'],
['Regrillado','Interpolar precipitación linealmente antes de clasificar días.'],
['Territorio','Mismas 3 587 celdas de Chile en el dominio 17,7°S–45,9°S.'],
['Ventanas','Loa, Maipo, Maule y Biobío, como cajas geográficas.'],
['Regiones',"Coquimbo, O’Higgins, La Araucanía y Los Lagos, con las máscaras administrativas previas."],
['Rachas','Censuradas de extremos excluidas; NaN interrumpe; se conserva la duración completa.'],
['Bootstrap','1 000 réplicas, años independientes, mismas semillas.'],
['RR','N_fut/N_hist de rachas ≥D con inicio mar–nov; se informan descartes por denominador cero.'],
['Mínimo','≥30 rachas completas por celda y producto, con sensibilidad a 50/100/200.']
],columns=['Elemento','Definición']))+
'\n\nSe conservan la raíz del paper, la prueba global anterior de 0,1 mm y la prueba local anterior de 1 mm. La carpeta actual congela los resultados locales a 1 mm en `referencia_local_1mm/`, y los controles globales a 0,1 mm en `referencia_global_0p1mm/`. Los hashes antes/después comprueban su integridad. No se editaron los Docs del paper.\n\n'
'Los identificadores A (global), B (igual umbral en ambos) y C (local) se conservan en las tablas de sensibilidad y bootstrap. El análisis principal usa C. Algunos nombres heredados contienen `i_global` o `integrado`, pero en las salidas principales de esta carpeta contienen el análisis local. La sensibilidad escalar usa ahora 0,1 mm en ALADIN, no 1 mm.')
calstats=pd.DataFrame([{'referencia':'1 mm',**{k:cal1[k] for k in ('celdas_dominio','ampliaron_busqueda','umbral_min','umbral_p10','umbral_mediana','umbral_p90','umbral_max','max_error_absoluto_pp')}},{'referencia':'0,1 mm',**{k:cal01[k] for k in ('celdas_dominio','ampliaron_busqueda','umbral_min','umbral_p10','umbral_mediana','umbral_p90','umbral_max','max_error_absoluto_pp')}}])
section(3,'Calibración por celda',md(calstats)+
'\n\nLos umbrales están en mm/día. La tolerancia es 0,0001 en fracción (0,01 puntos porcentuales). Se mantiene el algoritmo anterior: bisección en [0,15] y ampliación al máximo histórico finito de la celda, desplazado al siguiente valor representable, solo si el intervalo inicial no contiene la solución. No se sustituyen celdas sin solución por un umbral global. El campo se guarda en float32 sin redondear a tres decimales.\n\n'
f"En la referencia nueva quedan {cal01['celdas_fuera_tolerancia']} celdas fuera de tolerancia. La cobertura adicional fuera de la máscara nacional necesaria para los RR es de {cal01['celdas_administrativas_fuera_mascara']} celdas. En las series de calibración se registraron {cal01['NaN_cr2met']} valores no finitos de CR2MET y {cal01['NaN_aladin']} de ALADIN.\n\n"
'![Umbrales](comparativa/comparacion_umbrales.png)\n\n![Mapa comparativo](comparativa/comparacion_mapas_umbrales.png)\n\nLos dos primeros mapas usan la misma escala; el tercero muestra el umbral con referencia 0,1 mm menos el umbral con referencia 1 mm.\n\nFuente: `01_calibracion_por_celda.csv`, con las 3 587 correspondencias y sus errores de frecuencia. `calibracion_local.json` describe el ajuste aplicado; `calibracion_R0.json` y la figura global conservada describen únicamente la sensibilidad escalar auxiliar a 0,1 mm.')
section(4,'Frecuencia húmeda histórica y futura en ventanas',md(freq[['zona','CR2MET_pct_1mm','CR2MET_pct_0p1mm','ALADIN_R0_pct_1mm','ALADIN_R0_pct_0p1mm','exceso_R0_pp_1mm','exceso_R0_pp_0p1mm']])+
'\n\nAmbos experimentos igualan su propia referencia histórica anual. La referencia de 0,1 mm clasifica como húmedos días que con 1 mm eran secos; ello también cambia el umbral local de ALADIN. La igualdad anual no exige igualdad estacional.\n\n'
'Frecuencia húmeda futura, manteniendo fijo cada campo histórico:\n\n'+md(freq[['zona','ALADIN_R0_ssp585_2040_2074_pct_1mm','ALADIN_R0_ssp585_2040_2074_pct_0p1mm','cambio_rel_ssp585_pct_1mm','cambio_rel_ssp585_pct_0p1mm','cambio_rel_ssp370_pct_1mm','cambio_rel_ssp370_pct_0p1mm']])+
'\n\nFuente: `02_frecuencia_ventanas.csv`. Los cambios relativos son respecto del histórico de cada experimento; no son puntos porcentuales.')
section(5,'Frecuencia anual y estacional en regiones administrativas',md(freg[['region','estacion','CR2MET_pct_1mm','CR2MET_pct_0p1mm','local_pct_1mm','local_pct_0p1mm','exceso_local_pp_1mm','exceso_local_pp_0p1mm']])+
'\n\nSe usan las mismas máscaras de los RR. Marzo–noviembre no es el objetivo de calibración: controlar la frecuencia anual no garantiza igualar la frecuencia de esos meses. Fuente: `03_frecuencia_regiones.csv`.')
section(6,'Duraciones históricas por celda: referencia, modelo y errores',
'Las filas de «muestra propia» conservan la selección válida de cada configuración; «mismas celdas» restringe ambas a la intersección y elimina el efecto de cambiar la cobertura. En el informe anterior, global 1 mm frente a local 1 mm, se usó una intersección de 3 526 celdas. La comparación local 1 mm frente a local 0,1 mm tiene su propia intersección. Por eso el MAE de la versión local de 1 mm puede aparecer aquí con otro promedio: cambia la muestra de celdas, no sus resultados. La sección de las cuatro versiones incluye una muestra común adicional.\n\n'+md(spatial[['metrica','muestra','celdas_1mm','celdas_0p1mm','CR2MET_1mm','CR2MET_0p1mm','ALADIN_1mm','ALADIN_0p1mm','sesgo_1mm','sesgo_0p1mm','MAE_1mm','MAE_0p1mm']])+
'\n\n`mean` es la duración media por celda y `t99` su percentil 99. Todos los valores de duración y error están en días. El sesgo firmado es ALADIN−CR2MET; el MAE promedia |ALADIN−CR2MET| por celda. Un cambio de MAE entre estas dos referencias no evalúa exactamente la misma variable física.\n\n'
'![Duraciones](comparativa/comparacion_duraciones.png)\n\nDescomposición sobre las mismas celdas:\n\n'+md(spatial[spatial.muestra=='mismas celdas'][['metrica','cambio_ALADIN_comunes','cambio_CR2MET_comunes','cambio_sesgo_comunes','error_rel_absoluto_1mm_pct','error_rel_absoluto_0p1mm_pct']])+
'\n\nLa identidad es Δsesgo = ΔALADIN − ΔCR2MET. El error relativo absoluto divide el error por la métrica de CR2MET de la misma celda antes de promediar; ayuda a contextualizar la escala, pero no elimina el cambio de definición. Fuentes: `04_resumen_espacial.csv` y `05_metricas_por_celda.csv`.')
section(7,'Significancia histórica y resultados en ventanas',md(spatial[['metrica','muestra','menor_sig_1mm_pct','menor_sig_0p1mm_pct','mayor_sig_1mm_pct','mayor_sig_0p1mm_pct']])+
'\n\nLos porcentajes proceden de los bootstrap por celda de cada experimento y se restringen a la muestra indicada. No se aplica corrección por comparaciones múltiples.\n\n'
'Promedio espacial de diferencias por ventana:\n\n'+md(bias[['metrica','zona','celdas_validas_1mm','celdas_validas_0p1mm','delta_promedio_1mm','delta_promedio_0p1mm']])+
'\n\nSesgos relativos de los pools de rachas:\n\n'+md(tab[tab.producto=='ALADIN hist'][['ventana','media_1mm','media_0p1mm','media_sesgo_rel_pct_1mm','media_sesgo_rel_pct_0p1mm','p99_1mm','p99_0p1mm','p99_sesgo_rel_pct_1mm','p99_sesgo_rel_pct_0p1mm']])+
'\n\n![Sesgos](comparativa/comparacion_sesgos.png)\n\nEl promedio espacial y la estadística del pool no son intercambiables: el pool pesa cada racha por igual, por lo que las celdas con más rachas aportan más eventos. Las curvas llamadas PDFs en la cadena conservan conteos por ancho de intervalo; su integral es el número de rachas y no uno. Por eso un cambio de altura puede reflejar tanto cambios de conteo como de forma distributiva. Fuentes: `06_sesgo_ventanas.csv`, `07_tabla3_completa.csv`. Esta última conserva además CR2MET, mediana, p90, máximos y conteos/fracciones ≥20 días.')
section(8,'Cambios futuros de las distribuciones en ventanas',md(future)+
'\n\nLos cambios H→F se calculan dentro de cada definición, con su campo histórico fijo. Son resultados descriptivos: la cadena no calcula IC95 de los cambios de media o p99 de la Tabla 3. Comparar dos estimaciones puntuales no prueba que sean estadísticamente diferentes. Fuente: `08_cambios_futuros_ventanas.csv`.')
section(9,'RR principal de rachas de al menos 20 días',compact_rr(main).pipe(md)+
'\n\nConteos que sostienen esos cocientes:\n\n'+md(main[['region','N_hist_1mm','N_fut_1mm','N_hist_0p1mm','N_fut_0p1mm','descartadas_1mm','descartadas_0p1mm']])+
'\n\nUn aumento significativo exige IC95 inferior >1. Que un resultado sea significativo en una definición y no en otra no constituye un contraste estadístico de la diferencia entre los dos RR. Un cociente alto puede estar asociado a una base histórica pequeña; por eso se conservan sus conteos y réplicas descartadas. Fuente: `09_RR_D20.csv`.')
section(10,'Curvas RR y sensibilidad al escenario',
'![RR](comparativa/comparacion_RR.png)\n\nSSP3-7.0, con el mismo campo histórico local de cada definición:\n\n'+md(compact_rr(rr[rr.escenario=='SSP3-7.0']))+
'\n\nSensibilidad con igual umbral escalar en ambos productos: 1 mm en la referencia anterior y 0,1 mm en la nueva. **Estas filas son sensibilidades sin ajuste local**, conservadas para repetir el trabajo anterior:\n\n'+md(compact_rr(rr[rr.escenario=='igual umbral escalar']))+
'\n\nCambios de decisión a lo largo de las duraciones:\n\n'+md(curves[curves.cambia_decision][['escenario','region','D','RR_1mm','RR_0p1mm','decision_1mm','decision_0p1mm']])+
'\n\n`10_RR_todas_duraciones.csv` contiene los 120 puntos de cada versión emparejados: dos escenarios locales y una sensibilidad escalar, cuatro regiones y diez duraciones. No se interpreta la falta de significancia como ausencia de cambio. Las bandas se muestran solo donde no se descartaron réplicas.')
section(11,'Frecuencia seca y persistencia de primer orden',md(pers[['region','frec_seca_delta_1mm','frec_seca_delta_0p1mm','frec_seca_delta_IC95_inf_0p1mm','frec_seca_delta_IC95_sup_0p1mm','rho_delta_1mm','rho_delta_0p1mm','rho_delta_IC95_inf_0p1mm','rho_delta_IC95_sup_0p1mm','rho_IC_incluye_0_1mm','rho_IC_incluye_0_0p1mm']])+
'\n\nLa frecuencia seca se expresa como fracción: multiplicar por 100 para puntos porcentuales. ρ = P(seco hoy|seco ayer) − P(seco hoy|húmedo ayer). La calibración de frecuencia anual no impone su valor ni su cambio. Los intervalos reflejan remuestreo de años, sin recalibrar el campo en cada réplica. Fuente: `11_persistencia.csv`.')
section(12,'Diagnóstico Markov',md(markov[markov.D==20][['region','RR_obs_1mm','RR_solo_frec_1mm','solo_frec_dentro_IC95_obs_1mm','RR_obs_0p1mm','RR_solo_frec_0p1mm','cobertura_markov_hist_0p1mm','solo_frec_dentro_IC95_obs_0p1mm']])+
'\n\nCasos con al menos 100 rachas históricas:\n\n'+md(pd.DataFrame([{'referencia':label,'casos':int((markov['N_hist_obs_'+label]>=100).sum()),'compatibles':int(((markov['N_hist_obs_'+label]>=100)&markov['solo_frec_dentro_IC95_obs_'+label]).sum())} for label in ('1mm','0p1mm')]))+
'\n\nLa cobertura histórica es N_hist Markov/N_hist observado; un valor alejado de uno limita la representación del nivel histórico. El diagnóstico expresa compatibilidad con una referencia de primer orden, no atribución causal exclusiva. La dispersión de simulaciones no incluye incertidumbre de estimación de parámetros. Fuente: `12_markov.csv`.')
section(13,'Censura, mínimo de rachas y regrillado',
'### Rachas censuradas\n\n'+md(cens[['serie','cuenca','rachas_completas_1mm','rachas_completas_0p1mm','censuradas_1mm','censuradas_0p1mm','pct_censuradas_1mm','pct_censuradas_0p1mm']])+
'\n\nSe mantiene la exclusión de rachas que tocan extremos del período; su número puede cambiar al reclasificar días.\n\n### Mínimo de rachas\n\n'+md(mins[['minimo_rachas','celdas_1mm','celdas_0p1mm','delta_media_promedio_1mm','delta_media_promedio_0p1mm','delta_t99_promedio_1mm','delta_t99_promedio_0p1mm']])+
'\n\nEsta sensibilidad cambia la selección de celdas y restringe la significancia ya calculada; no recalcula un bootstrap distinto por cada mínimo.\n\n### Regrillado\n\n'+md(regrid[regrid.criterio=='i'][['cr2met','metrica','CR2MET_1mm','CR2MET_0p1mm','ALADIN_1mm','ALADIN_0p1mm','Delta_1mm','Delta_0p1mm','pix_1mm','pix_0p1mm']])+
'\n\nEl criterio `i` identifica el análisis local principal en esta prueba; se conservan las alternativas en el CSV. El método alternativo promedia celdas cuyos centros caen dentro de la celda ALADIN; no es conservación estricta por áreas de solape. Se recalibran los campos locales frente a cada referencia de regrillado. Fuentes: `13_censuradas.csv`, `14_minimo_rachas.csv`, `15_regrillado.csv`.\n\n'+
f"**Corrección de precisión auxiliar:** el primer control detectó una racha de CR2MET adicional en Biobío en el diagnóstico de censura (357 068 frente a 357 067), debido al redondeo de la caché interpolada a float32 cerca de 0,1 mm. Se conservó esa primera salida en `revision_precision_float32/` y se regeneró la caché en float64, manteniendo pesos, fechas y máscara. En la interpolación lineal, ese redondeo cambia la clasificación de {precision['lin']['dias_celda_reclasificados']} días-celda en {precision['lin']['celdas_afectadas']} celdas. En el promedio alternativo por centros son {precision['area']['dias_celda_reclasificados']} días-celda en {precision['area']['celdas_afectadas']} celdas. Se repitieron regrillado, censura y T01, y se verificaron el campo local y las métricas lineales frente al principal. La calibración, bootstrap, PDFs, Tabla 3, RR y persistencia principales ya usaban la precisión canónica y no se modificaron por esta corrección. Véanse `NOTA_PRECISION.md` y `precision_auxiliar.json`.")
summary=[]
for row in main.itertuples():summary.append(f'- **{row.region}, SSP5-8.5 a D=20:** RR {row.RR_1mm:.3f} → {row.RR_0p1mm:.3f}; {row.decision_1mm} → {row.decision_0p1mm}.')
for root,label in ((A,'1 mm'),(B,'0,1 mm')):
    q=rd(root,'resultados_tesis/contraste_persistencia_regiones.csv');summary.append(f"- **Persistencia, referencia {label}:** IC95 de Δρ incluye cero en {int(q.rho_IC_incluye_0.sum())} de {len(q)} regiones.")
section(14,'Comparación adicional: referencia y escala de calibración',
'Esta tabla distingue las dos decisiones: referencia de CR2MET (1 o 0,1 mm) y calibración de ALADIN (global o por celda). La comparación dentro de una misma referencia mantiene fijo el significado de la racha observada. La comparación entre referencias cambia su definición.\n\n'+md(factor_hist[factor_hist.muestra=='mismas celdas de las cuatro versiones'])+
'\n\nRR a D=20 bajo SSP5-8.5:\n\n'+md(factor_rr[(factor_rr.D==20)&(factor_rr.escenario=='SSP5-8.5')][['referencia','calibracion','region','N_hist','N_fut','RR','IC_inf','IC_sup','decision']])+
'\n\nFuentes: `17_comparacion_cuatro_versiones_historico.csv` y `18_comparacion_cuatro_versiones_RR.csv`. El segundo contiene las cuatro configuraciones, ambos escenarios y las diez duraciones. Las referencias globales conservan sus resultados previos; la cadena nueva vuelve a verificar sus controles históricos globales y escalares de 0,1 mm.')
new_persistence=rd(B,'resultados_tesis/contraste_persistencia_regiones.csv')
for x in new_persistence[~new_persistence.rho_IC_incluye_0].itertuples():
    summary.append(f'- **Cambio detectable de persistencia con referencia 0,1 mm, {x.region}:** Δρ={x.rho_delta:.5f}; IC95 [{x.rho_delta_IC95_inf:.5f}; {x.rho_delta_IC95_sup:.5f}].')
section(15,'Implicaciones para el paper y límites de interpretación','\n'.join(summary)+
'\n\nLa decisión entre 1 y 0,1 mm debe fundarse en qué precipitación diaria se considera suficiente para interrumpir la racha meteorológica de interés. El ajuste local responde a otra decisión: hacer equivalente la frecuencia histórica de cada celda. Son dos elecciones distintas. Este experimento no elige el umbral por el número de resultados significativos ni por el menor MAE.\n\n'
'Una formulación defendible es: «Se comparó la sensibilidad de las duraciones y sus cambios proyectados a dos referencias de día húmedo de CR2MET, recalibrando en cada caso el umbral histórico de ALADIN por celda y conservándolo fijo en el futuro». Los resultados que cambien de decisión deben describirse como sensibles a esa definición.\n\n'
'Limitaciones conservadas: una simulación de un modelo; incertidumbre de CR2MET; dependencia espacial; calibración con todo el histórico y sin validación temporal independiente; bootstrap con umbrales fijos; igualdad anual que no garantiza igualdad estacional; precipitación en mm sin corregir; umbrales estadísticos sin validación como indicadores de impacto agrícola. El dominio y el alcance de las cuatro regiones permanecen iguales.\n\n'
'Un cambio detectable de ρ y un RR de rachas largas no significativo pueden coexistir: describen estadísticas diferentes y tienen incertidumbres distintas. Detectar Δρ tampoco demuestra que sea la causa exclusiva de las rachas extremas. No se debe trasladar a la referencia de 0,1 mm una conclusión sobre ausencia de cambio detectable de persistencia obtenida con 1 mm.\n\n'
'Los cambios en medias, p99 y RR entre estas dos definiciones no son por sí solos evidencia de un cambio físico del clima: ambas procesan exactamente los mismos datos meteorológicos.')
section(16,'Verificación e integridad',md(pd.DataFrame(checks))+
f"\n\nLa cadena pasó {cons['checks']} controles internos de consistencia. Se verificaron {integ['archivos_previos_comprobados']} archivos de análisis anteriores y {integ['NetCDF_comprobados']} NetCDF; las listas de modificaciones están en `integridad.json`. El inventario compara {len(inv)} salidas emparejadas por su función: {int(inv.igual.sum())} iguales y {int((~inv.igual).sum())} diferentes o nuevas.\n\n"
'Los controles global y escalar a 0,1 mm se contrastan con el cálculo global anterior de 0,1 mm, no con la referencia de 1 mm. El campo local anterior de 0,1 mm se verifica en las celdas previamente válidas; la búsqueda ampliada conserva ese comportamiento y resuelve las celdas fuera del intervalo inicial. Los hashes y el diff registran exactamente los scripts finales. Las diferencias esperadas son el objeto del experimento, no fallos de reproducción.')
section(17,'Archivos y reproducción',
'- `ejecucion/`: scripts y resultados nuevos, incluidos todos los mapas, PDFs, bootstrap, RR y sensibilidades.\n'
'- `referencia_local_1mm/`: copia congelada del experimento local anterior.\n'
'- `referencia_global_0p1mm/`: controles del experimento global de 0,1 mm ya existente.\n'
'- `comparativa/`: dieciocho CSV comparativos y cinco figuras.\n'
'- `estado.json`, `logs/`: estado, comandos, tiempos y errores.\n'
'- `controles_consistencia.json`, `controles_comparativa.json`, `integridad.json`: verificaciones.\n'
'- `scripts_base_local_1mm/`, `cambios_codigo.diff`, `scripts_ejecutados_sha256.json`: trazabilidad de código.\n\n'
'Entorno científico existente: `~/.pyenv/shims/python3.11`; no se instalaron paquetes. `ejecutar.py` reanuda pasos terminados sin repetirlos. Después de modificar reglas o scripts debe usarse una ejecución nueva, pues no invalida dependencias automáticamente. `repetir_desde_cero.py --destino /ruta/nueva` reproduce todos los cálculos desde NetCDF en otra carpeta sin cachés de análisis. `generar_comparativa.py` regenera este informe y sus tablas desde las salidas calculadas. `preparar.py` conserva la preparación inicial y no debe ejecutarse sobre esta carpeta ya preparada.')
(R/'INFORME_COMPARATIVO_LOCAL_0p1mm_VS_1mm.md').write_text('\n'.join(P),encoding='utf-8')
(R/'resumen.json').write_text(json.dumps(dict(comparacion='local 1 mm vs local 0,1 mm',calibracion_0p1mm=cal01,espacial_mismas_celdas=spatial[spatial.muestra=='mismas celdas'].to_dict(orient='records'),RR_D20_SSP585=main.to_dict(orient='records'),controles_internos=cons['checks'],controles_comparativa_correctos=all(x['passed'] for x in checks),versiones_anteriores_intactas=integ['passed']),indent=2,ensure_ascii=False))
(R/'salidas_nuevas_sha256.json').write_text(json.dumps(manifest(outputs(B)),indent=2))
(R/'comparativa_sha256.json').write_text(json.dumps(manifest({p.relative_to(R).as_posix():p for p in list(O.iterdir())+[R/'INFORME_COMPARATIVO_LOCAL_0p1mm_VS_1mm.md'] if p.is_file()}),indent=2))
print(json.dumps(dict(informe=str(R/'INFORME_COMPARATIVO_LOCAL_0p1mm_VS_1mm.md'),passed=all(x['passed'] for x in checks),checks=checks),indent=2,ensure_ascii=False))
if not all(x['passed'] for x in checks):raise SystemExit(2)
