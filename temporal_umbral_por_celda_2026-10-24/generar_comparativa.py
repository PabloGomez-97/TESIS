"""Genera tablas, figuras e informe desde salidas calculadas; no incorpora cifras manuales."""
from pathlib import Path
import json, pickle, hashlib, difflib, sys, importlib.metadata
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent; B=R/'referencia_global'; L=R/'ejecucion'; O=R/'comparativa'
O.mkdir(exist_ok=True)
ORIGIN=Path(json.loads((R/'configuracion.json').read_text())['repo_origen'])
def read(root,rel):return pd.read_csv(root/rel)
def js(root,rel):return json.loads((root/rel).read_text())
def dump(name,df):df.to_csv(O/(name+'.csv'),index=False);return df
def fmt(x):
    if isinstance(x,(bool,np.bool_)):return 'sí' if x else 'no'
    if isinstance(x,(float,np.floating)):return '—' if not np.isfinite(x) else f'{x:.3f}'
    return str(x)
def md(df):
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
def paired(rel,keys):
    a=read(B,rel);b=read(L,rel)
    z=a.merge(b,on=keys,suffixes=('_global','_local'),validate='one_to_one')
    for c in a.select_dtypes(include='number').columns:
        if c not in keys and c in b:z[c+'_diferencia']=z[c+'_local']-z[c+'_global']
    return z
freq=dump('01_frecuencia_ventanas',paired('resultados_tesis/frecuencia_humeda_ventanas.csv',['zona']))
bias=dump('02_sesgo_por_ventana',paired('resultados_tesis/sesgo_por_ventana.csv',['metrica','zona']))
tab=dump('03_tabla3_completa',paired('resultados_tesis/tabla3_estadisticos_ventanas.csv',['ventana','producto']))
# Cambios H->F en ambas configuraciones, separados de diferencias local-global.
rows=[]
for root,label in ((B,'global'),(L,'local')):
    t=read(root,'resultados_tesis/tabla3_estadisticos_ventanas.csv')
    for v in t.ventana.unique():
        s=t[t.ventana==v].set_index('producto');h=s.loc['ALADIN hist'];f=s.loc['ALADIN fut']
        rows.append(dict(ventana=v,calibracion=label,n_hist=int(h.n_spells),n_fut=int(f.n_spells),
            cambio_n_pct=100*(f.n_spells/h.n_spells-1),media_hist=h.media,media_fut=f.media,cambio_media_dias=f.media-h.media,
            cambio_media_pct=100*(f.media/h.media-1),p99_hist=h.p99,p99_fut=f.p99,cambio_p99_dias=f.p99-h.p99,
            cambio_p99_pct=100*(f.p99/h.p99-1)))
future=dump('04_cambios_futuros_ventanas',pd.DataFrame(rows))
rrs=[];curves=[]
def decision(lo,hi):
    if not np.all(np.isfinite([lo,hi])):return 'no definido'
    return 'aumento' if lo>1 else ('disminución' if hi<1 else 'IC incluye 1')
for suffix,scenario in (('','SSP5-8.5'),('_ssp370','SSP3-7.0'),('_1mm','1 mm / SSP5-8.5')):
    a=js(B,f'outputs_pregunta5_aladin{suffix}/results.json');b=js(L,f'outputs_pregunta5_aladin{suffix}/results.json')
    for ga,lb in zip(a['rr_table'],b['rr_table']):
        assert ga['Region']==lb['Region']
        r=dict(escenario=scenario,region=ga['Region'])
        for lab,d in (('global',ga),('local',lb)):
            for key,out in [('N_hist','N_hist'),('N_fut','N_fut'),('RR observado','RR'),('IC95 inferior','IC_inf'),('IC95 superior','IC_sup'),('Replicas descartadas (de 1000)','descartadas')]:r[out+'_'+lab]=d[key]
            r['decision_'+lab]=decision(d['IC95 inferior'],d['IC95 superior'])
        r['RR_diferencia']=r['RR_local']-r['RR_global'];rrs.append(r)
    for region in a['curves']:
        for x,y in zip(a['curves'][region],b['curves'][region]):
            r=dict(escenario=scenario,region=region,D=x['Umbral (dias)'])
            for label,d in (('global',x),('local',y)):
                for k in ('RR','IC95 inferior','IC95 superior','N1','N2','Total1','Total2','Replicas descartadas'):r[k+'_'+label]=d[k]
                r['decision_'+label]=decision(d['IC95 inferior'],d['IC95 superior'])
            r['cambia_decision']=r['decision_global']!=r['decision_local'];curves.append(r)
rr=dump('05_RR_D20',pd.DataFrame(rrs));curve=dump('06_RR_todas_duraciones',pd.DataFrame(curves))
pers=dump('07_persistencia',paired('resultados_tesis/contraste_persistencia_regiones.csv',['region','meses']))
mark=dump('08_markov',paired('resultados_tesis/descomposicion_markov_rr_regiones.csv',['region','D']))
cens=dump('09_censuradas',paired('_test_regrillado/rachas_censuradas_por_cuenca.csv',['serie','cuenca']))
mins=dump('10_minimo_rachas',paired('resultados_tesis/sensibilidad_minimo_rachas.csv',['minimo_rachas']))
# Comparación espacial con y sin restricción a la misma muestra.
zg=np.load(B/'_pregunta7_outputs/metrics_integrado_cache.npz');zl=np.load(L/'_pregunta7_outputs/metrics_integrado_cache.npz')
rows=[]; cellrows=[]
for metric in ('mean','t99'):
    c=zg[metric+'_cr2'];ag=zg[metric+'_ala'];al=zl[metric+'_ala'];m=zg['chile_mask']
    vg=m&np.isfinite(c)&np.isfinite(ag);vl=m&np.isfinite(c)&np.isfinite(al);common=vg&vl
    bg=np.load(B/f'resultados_tesis/bootstrap_por_celda_A_opcion_B_integrado_{metric}.npz')
    bl=np.load(L/f'resultados_tesis/bootstrap_por_celda_C_umbral_local_{metric}.npz')
    for selection,sg,sl in (('muestra propia',vg,vl),('mismas celdas',common,common)):
        rows.append(dict(metrica=metric,muestra=selection,n_global=int(sg.sum()),n_local=int(sl.sum()),
            sesgo_global_dias=float((ag-c)[sg].mean()),sesgo_local_dias=float((al-c)[sl].mean()),
            MAE_global_dias=float(np.abs(ag-c)[sg].mean()),MAE_local_dias=float(np.abs(al-c)[sl].mean()),
            pct_menor_sig_global=float(100*bg['mas_corto'][sg].mean()),pct_menor_sig_local=float(100*bl['mas_corto'][sl].mean()),
            pct_mayor_sig_global=float(100*bg['mas_largo'][sg].mean()),pct_mayor_sig_local=float(100*bl['mas_largo'][sl].mean()),
            pct_menor_error_local_en_comunes=float(100*(np.abs(al-c)[common]<np.abs(ag-c)[common]).mean())))
    for j,i in zip(*np.where(m)):
        cellrows.append(dict(metrica=metric,j=j,i=i,lat=zg['lat'][j,i],lon=zg['lon'][j,i],valida_global=bool(vg[j,i]),valida_local=bool(vl[j,i]),
            cr2met=float(c[j,i]),global_dias=float(ag[j,i]),local_dias=float(al[j,i]),sesgo_global=float(ag[j,i]-c[j,i]),sesgo_local=float(al[j,i]-c[j,i]),
            significativo_global=bool(bg['significativo'][j,i]),significativo_local=bool(bl['significativo'][j,i])))
spatial=dump('11_resumen_espacial',pd.DataFrame(rows));dump('12_metricas_todas_celdas',pd.DataFrame(cellrows))
cal=js(L,'resultados_tesis/calibracion_local.json')
cd=read(L,'resultados_tesis/calibracion_local_por_celda.csv')
old=np.load(B/'_pregunta7_outputs/R0_local_map.npz');new=np.load(L/'_pregunta7_outputs/R0_local_map.npz');ok=old['chile']&np.isfinite(old['r0'])
legacy=dict(celdas_local_previo=int(ok.sum()),celdas_recuperadas=int((old['chile']&~np.isfinite(old['r0'])&np.isfinite(new['r0'])).sum()),
    umbrales_previos_exactamente_iguales=bool(np.array_equal(old['r0'][ok],new['r0'][ok])),max_diferencia_previas_mm=float(np.max(np.abs(old['r0'][ok]-new['r0'][ok]))))
(O/'control_local_previo.json').write_text(json.dumps(legacy,indent=2))
# El archivo global reproducido y el criterio 1 mm deben seguir iguales.
checks=[]
def ck(name,test):checks.append(dict(control=name,passed=bool(test)))
for metric in ('mean','t99'):
    ck('CR2MET mapas idénticos '+metric,np.array_equal(zg[metric+'_cr2'],zl[metric+'_cr2'],equal_nan=True))
    for criterion in ('A_opcion_B_integrado','B_mismo_umbral'):
        ga=np.load(B/f'resultados_tesis/bootstrap_por_celda_{criterion}_{metric}.npz');lb=np.load(L/f'resultados_tesis/bootstrap_por_celda_{criterion}_{metric}.npz')
        ck('Bootstrap control idéntico '+criterion+'/'+metric,all(np.array_equal(ga[k],lb[k],equal_nan=True) for k in ga.files))
ck('RR 1 mm idénticos',json.dumps(js(B,'outputs_pregunta5_aladin_1mm/results.json'),sort_keys=True)==json.dumps(js(L,'outputs_pregunta5_aladin_1mm/results.json'),sort_keys=True))
ck('Estacionalidad y acumulados idénticos',read(B,'resultados_tesis/estacionalidad_marnov_regiones.csv').equals(read(L,'resultados_tesis/estacionalidad_marnov_regiones.csv')))
ck('Máscara geográfica idéntica',np.array_equal(zg['chile_mask'],zl['chile_mask']))
ck('Umbrales locales previos preservados',legacy['umbrales_previos_exactamente_iguales'])
for metric in ('mean','t99'):
    ga=np.load(B/f'resultados_tesis/bootstrap_por_celda_C_umbral_local_{metric}.npz');lb=np.load(L/f'resultados_tesis/bootstrap_por_celda_C_umbral_local_{metric}.npz')
    mask=ga['valida']
    ck('Bootstrap local previo idéntico en celdas previamente válidas '+metric,all(np.array_equal(ga[k][mask],lb[k][mask],equal_nan=True) for k in ga.files))
ck('Cobertura completa de calibración',cal['celdas_fuera_tolerancia']==0 and np.isfinite(new['r0'][new['chile']]).all())
# Integridad de la raíz, incluyendo cambios que el usuario ya tenía antes de esta prueba.
sys.path.insert(0,str(ORIGIN/'reproducibilidad'))
from ejecutar import outputs,manifest,compare_file
before=js(R,'originales_sha256_antes.json')
paths={p.relative_to(ORIGIN).as_posix():p for p in ORIGIN.glob('_*.py')};paths.update(outputs(ORIGIN))
after=manifest(paths)
(R/'originales_sha256_despues.json').write_text(json.dumps(after,indent=2))
ck('Scripts y salidas originales sin cambios',before==after)
# Inventario de salidas, incluidas figuras y cachés.
inventory=[]
global_files=outputs(B);local_files=outputs(L)
for rel in sorted(set(global_files)|set(local_files)):
    if rel not in global_files:res={'equal':False,'method':'nuevo'}
    elif rel not in local_files:res={'equal':False,'method':'faltante'}
    else:res=compare_file(global_files[rel],local_files[rel])
    inventory.append(dict(archivo=rel,igual=res['equal'],metodo=res['method']))
dump('13_inventario_salidas',pd.DataFrame(inventory))
# Diff actualizado de todos los scripts y hashes del código efectivamente ejecutado.
diffs=[]
for p in sorted((R/'scripts_originales').glob('*.py')):
    diffs.extend(difflib.unified_diff(p.read_text().splitlines(True),(L/p.name).read_text().splitlines(True),fromfile='original/'+p.name,tofile='local/'+p.name))
(R/'cambios_codigo.diff').write_text(''.join(diffs))
(R/'scripts_locales_sha256.json').write_text(json.dumps(manifest({p.name:p for p in L.glob('*.py')}),indent=2))
cons=js(R,'controles_consistencia.json')
ck('Controles cruzados de cadena local',cons['passed'])
ck('680 NetCDF coinciden con el registro',js(R,'integridad_datos.json')['passed'])
(R/'controles_comparativa.json').write_text(json.dumps(dict(passed=all(c['passed'] for c in checks),checks=checks),indent=2,ensure_ascii=False))
# Figuras comparativas: mismas escalas y referencias para evitar comparación visual engañosa.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#436e9b','#c16436']
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
axes[0].hist(cd.umbral_local_mm,bins=40,color=colors[1],alpha=.8);axes[0].axvline(5.285,color=colors[0],ls='--',label='Global: 5,285 mm/día');axes[0].set(xlabel='Umbral local (mm/día)',ylabel='Celdas');axes[0].legend()
x=np.arange(len(freq));w=.36
axes[1].bar(x-w/2,freq.exceso_R0_pp_global,w,label='Global',color=colors[0]);axes[1].bar(x+w/2,freq.exceso_R0_pp_local,w,label='Local',color=colors[1]);axes[1].axhline(0,color='k',lw=.7);axes[1].set_xticks(x,freq.zona,rotation=25);axes[1].set(ylabel='ALADIN − CR2MET (puntos porcentuales)');axes[1].legend()
fig.suptitle('Calibración histórica de la frecuencia húmeda');fig.tight_layout();fig.savefig(O/'comparacion_calibracion.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
for ax,(region,g) in zip(axes.ravel(),curve[curve.escenario=='SSP5-8.5'].groupby('region',sort=False)):
    for lab,color in zip(('global','local'),colors):
        x=g.D.to_numpy();y=g['RR_'+lab].to_numpy(float);lo=g['IC95 inferior_'+lab].to_numpy(float);hi=g['IC95 superior_'+lab].to_numpy(float)
        ax.plot(x,y,marker='o',ms=3,label=lab.capitalize(),color=color)
        good=(g['Replicas descartadas_'+lab].to_numpy()==0)&np.isfinite(lo)&np.isfinite(hi)&(lo>0)
        ax.fill_between(x,lo,hi,where=good,color=color,alpha=.14)
    ax.set_yscale('log');ax.axhline(1,color='k',ls=':');ax.axvline(20,color='0.6',ls='--',lw=.7);ax.set(title=region,ylabel='RR de conteos');ax.grid(alpha=.15);ax.legend()
for ax in axes[-1]:ax.set_xlabel('Duración mínima D (días)')
fig.suptitle('SSP5-8.5: umbral global frente a umbrales locales\nBandas IC95 solo sin réplicas descartadas');fig.tight_layout();fig.savefig(O/'comparacion_RR.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for ax,met in zip(axes,('media','p99')):
    tt=tab[tab.producto=='ALADIN hist'];x=np.arange(len(tt));w=.35
    ax.bar(x-w/2,tt[met+'_sesgo_rel_pct_global'],w,label='Global',color=colors[0]);ax.bar(x+w/2,tt[met+'_sesgo_rel_pct_local'],w,label='Local',color=colors[1]);ax.set_xticks(x,tt.ventana);ax.axhline(0,color='k',lw=.7);ax.set(title='Media' if met=='media' else 'Percentil 99',ylabel='Sesgo relativo del pool histórico (%)');ax.legend()
fig.tight_layout();fig.savefig(O/'comparacion_sesgos_ventanas.png',dpi=180);plt.close(fig)
# Informe detallado. Cada tabla proviene de los CSV acompañantes.
parts=[]
def section(title,body):parts.append('## '+title+'\n\n'+body+'\n')
parts.append('# Comparativa del paper: umbral global frente a umbrales por celda\n\nFecha: 24 de octubre de 2026. Experimento aislado; los resultados oficiales del paper conservan su versión previa.\n')
intro_spatial=spatial[(spatial.muestra=='mismas celdas') & (spatial.metrica=='mean')].iloc[0]
rr_main=rr[rr.escenario=='SSP5-8.5']
intro=(f"**Resultado principal:** en las mismas {int(intro_spatial.n_global)} celdas, el error absoluto medio de la duración media pasa de {intro_spatial.MAE_global_dias:.3f} a {intro_spatial.MAE_local_dias:.3f} días, y su sesgo firmado de {intro_spatial.sesgo_global_dias:.3f} a {intro_spatial.sesgo_local_dias:.3f} días. A D=20, el número de regiones con un aumento cuyo IC95 excluye uno pasa de {int((rr_main.decision_global=='aumento').sum())} a {int((rr_main.decision_local=='aumento').sum())}. Las tablas siguientes identifican cuáles y cómo cambian sus conteos e intervalos.\n\n")
section('1. Qué se hizo y qué permite concluir',
intro+'Se repitió la cadena científica de la raíz con CR2MET a 1 mm/día y un umbral histórico de ALADIN específico por celda. El mismo campo histórico se aplicó a 2040–2074, tanto bajo SSP5-8.5 como en la sensibilidad SSP3-7.0. No se ajustaron las duraciones para que coincidieran.\n\n'
'El experimento permite medir cuánto cambian la evaluación histórica y las proyecciones al controlar la frecuencia húmeda local. Una reducción del sesgo después de calibrar es un resultado condicionado al ajuste; no demuestra una mejora física del modelo ni valida por sí sola sus proyecciones.')
section('2. Diseño y condiciones preservadas',
md(pd.DataFrame([
['Datos y unidades','Mismos NetCDF de CR2MET v2.5 y ALADIN CHP12; ALADIN × 86 400 a mm/día.'],
['Regrillado principal','Interpolación lineal de precipitación antes de clasificar los días.'],
['Períodos','1980–2014 / 2040–2074, 35 años por período.'],
['Dominio','Misma máscara de Chile del paper, 17,7°S–45,9°S.'],
['Ventanas','Loa, Maipo, Maule y Biobío; cajas geográficas, no cuencas oficiales.'],
['RR regionales',"Coquimbo, O’Higgins, La Araucanía y Los Lagos; inicio mar–nov."],
['Rachas','Días pr < umbral; se excluyen rachas que tocan extremos del período; se conserva su duración completa.'],
['Bootstrap','1 000 réplicas; años independientes entre productos/períodos; semillas originales.'],
['Métricas por celda','Al menos 30 rachas completas; máscara común entre productos.'],
['RR','Conteo futuro / conteo histórico, sin dividir por el total de rachas.'],
['Sensibilidades','1 mm, SSP3-7.0, regrillado por centros y mínimos de 30/50/100/200 rachas.']
],columns=['Elemento','Condición']))+
'\n\nLos criterios A (global), B (1 mm) y C (local) conservan sus identificadores dentro del bootstrap y de las tablas de sensibilidad. En esta carpeta el mapa y los análisis principales usan C. Algunas rutas heredadas contienen `i_global` o `integrado`: son identificadores de archivos preservados para mantener la cadena; no describen el umbral principal de este experimento. `calibracion_R0.json` conserva el diagnóstico escalar global como referencia, y su figura original se guarda como `fig03_referencia_umbral_global.png`. La figura 3 principal se adapta al campo local mediante `figura_calibracion_local.py`. El registro del ajuste aplicado es `calibracion_local.json` y el campo `R0_local_map.npz`.')
section('3. Calibración local y el límite previo de 15 mm/día',
f"Se calibraron {cal['celdas_dominio']} celdas del dominio. La cobertura adicional necesaria para las máscaras administrativas fue de {cal['celdas_administrativas_fuera_mascara']} celdas. El histórico tiene {cal['dias_historicos']} días; se registraron {cal['NaN_cr2met']} valores no finitos de CR2MET y {cal['NaN_aladin']} de ALADIN en las series usadas para calibrar.\n\n"
f"La implementación previa dejaba {legacy['celdas_recuperadas']} celdas sin solución por restringir la búsqueda a [0,15] mm/día. En ellas se amplió el extremo superior hasta el máximo histórico finito de la celda, desplazado al siguiente valor representable. En las demás se conservó el intervalo previo. Los {legacy['celdas_local_previo']} umbrales previos válidos permanecen exactamente iguales: {fmt(legacy['umbrales_previos_exactamente_iguales'])}. No se reemplazaron las celdas problemáticas por el umbral global.\n\n"
+md(pd.DataFrame([['Mínimo',cal['umbral_min']],['p10',cal['umbral_p10']],['Mediana',cal['umbral_mediana']],['p90',cal['umbral_p90']],['Máximo',cal['umbral_max']]],columns=['Distribución del umbral','mm/día']))+
f"\n\nLa tolerancia de búsqueda es {cal['tolerancia_fraccion']} en fracción (0,01 puntos porcentuales). El máximo error absoluto observado es {cal['max_error_absoluto_pp']:.6f} puntos porcentuales; hay {cal['celdas_fuera_tolerancia']} celdas fuera de tolerancia. Las frecuencias empíricas son discretas y no identifican necesariamente un único umbral. Se conserva el campo en float32, sin redondearlo a tres decimales.\n\n"
'**Diferencia respecto de la sensibilidad local anterior:** el experimento principal incluye las celdas recuperadas. Por eso el contraste con el criterio local previamente publicado mezcla la ampliación de cobertura con sus resultados; el contraste principal de este informe es global frente a local y se presenta también sobre las mismas celdas.\n\n![Calibración](comparativa/comparacion_calibracion.png)')
section('4. Frecuencia húmeda en el dominio y las ventanas',
md(freq[['zona','CR2MET_1mm_pct_global','ALADIN_R0_pct_global','ALADIN_R0_pct_local','exceso_R0_pp_global','exceso_R0_pp_local']].rename(columns={
'CR2MET_1mm_pct_global':'CR2MET (%)','ALADIN_R0_pct_global':'ALADIN global (%)','ALADIN_R0_pct_local':'ALADIN local (%)','exceso_R0_pp_global':'Exceso global (pp)','exceso_R0_pp_local':'Exceso local (pp)'}))+
'\n\nLa igualdad histórica anual local es un objetivo de calibración, no un resultado independiente de validación. No impone igualdad mensual, estacional, de cantidades de lluvia ni de duraciones. Fuentes: `01_frecuencia_ventanas.csv` y `ejecucion/resultados_tesis/calibracion_local_por_celda.csv`.')
regional=read(L,'resultados_tesis/frecuencia_humeda_regiones_comparada.csv')
section('5. Frecuencia húmeda en las regiones administrativas',md(regional)+
'\n\nLa tabla usa las mismas máscaras administrativas de los RR. La calibración anual no exige igualdad en marzo–noviembre. Esta distinción importa al interpretar la evaluación de rachas y el diagnóstico de persistencia estacional.')
section('6. Evaluación histórica espacial: sesgo, error y cobertura',md(spatial)+
'\n\n`mean` es la duración media por celda y `t99` su percentil 99. MAE es el promedio del error absoluto por celda. El sesgo firmado puede acercarse a cero por compensación entre errores positivos y negativos; el MAE permite revisar esa posibilidad.\n\n'
'La fila «muestra propia» reproduce la selección válida de cada configuración. La fila «mismas celdas» restringe ambas a la intersección, de modo que el contraste no se deba a incorporar o excluir celdas. Los porcentajes significativos proceden del bootstrap original de cada celda, restringido a la muestra indicada. No se aplicó corrección por comparaciones múltiples. Fuentes: `11_resumen_espacial.csv`, `12_metricas_todas_celdas.csv` y NPZ bootstrap.')
section('7. Evaluación histórica en ventanas: dos estadísticas diferentes',
'El promedio espacial de métricas por celda y la métrica del pool de rachas no son intercambiables. En el pool cada racha pesa una vez; una celda con muchas rachas aporta más eventos.\n\n'
'### Promedio espacial de las diferencias por celda\n\n'+md(bias[['metrica','zona','celdas_validas_global','celdas_validas_local','delta_promedio_global','delta_promedio_local']])+
'\n\n### Sesgo relativo de las distribuciones agrupadas\n\n'+md(tab[tab.producto=='ALADIN hist'][['ventana','media_global','media_local','media_sesgo_rel_pct_global','media_sesgo_rel_pct_local','p99_global','p99_local','p99_sesgo_rel_pct_global','p99_sesgo_rel_pct_local']])+
'\n\n![Sesgos](comparativa/comparacion_sesgos_ventanas.png)\n\nFuentes: `02_sesgo_por_ventana.csv` y `03_tabla3_completa.csv`. La tabla completa también conserva mediana, p90, máximos, conteos y fracciones ≥20 días.')
section('8. Cambios futuros de las distribuciones en ventanas',md(future)+
'\n\nEstos cambios son descriptivos: la cadena original no calcula IC95 de los cambios de media y p99 de la Tabla 3. Un cambio entre calibraciones no prueba una diferencia estadísticamente significativa entre ellas. Se comparan rachas definidas con un umbral distinto en cada configuración, fijo en el tiempo dentro de esa configuración. Fuente: `04_cambios_futuros_ventanas.csv`.')
section('9. RR de rachas de al menos 20 días bajo SSP5-8.5',
md(rr[rr.escenario=='SSP5-8.5'][['region','N_hist_global','N_fut_global','RR_global','IC_inf_global','IC_sup_global','decision_global','N_hist_local','N_fut_local','RR_local','IC_inf_local','IC_sup_local','decision_local']])+
'\n\nSe conservan los conteos, no solo sus cocientes: un RR mayor puede deberse a una base histórica menor. La decisión informa si cada IC95 excluye uno. No constituye un contraste de la diferencia entre los dos RR. Fuente: `05_RR_D20.csv`.')
section('10. Curvas RR y sensibilidades a escenario y umbral',
'![Curvas RR](comparativa/comparacion_RR.png)\n\n'+md(rr[rr.escenario!='SSP5-8.5'][['escenario','region','RR_global','IC_inf_global','IC_sup_global','decision_global','RR_local','IC_inf_local','IC_sup_local','decision_local']])+
'\n\nEn la sensibilidad de 1 mm no hay umbrales calibrados: se conserva el mismo experimento y debe reproducirse exactamente en ambas carpetas. SSP3-7.0 usa el mismo campo local calibrado en el histórico, sin recalibración por escenario.\n\n'
+'Cambios de decisión a lo largo de todas las duraciones:\n\n'+md(curve[curve.cambia_decision][['escenario','region','D','RR_global','RR_local','decision_global','decision_local']])+
'\n\nEl archivo `06_RR_todas_duraciones.csv` incluye los 120 puntos (tres análisis × cuatro regiones × diez duraciones), sus conteos e intervalos. Las réplicas sin conteo histórico se registran; un intervalo muy amplio o calculado tras descartes requiere cautela. No se interpreta la falta de significancia como ausencia de cambio.')
section('11. Frecuencia seca y persistencia de primer orden',
md(pers[['region','frec_seca_delta_global','frec_seca_delta_local','frec_seca_delta_IC95_inf_local','frec_seca_delta_IC95_sup_local','rho_delta_global','rho_delta_local','rho_delta_IC95_inf_local','rho_delta_IC95_sup_local','rho_IC_incluye_0_global','rho_IC_incluye_0_local']])+
'\n\nLas frecuencias se expresan como fracción; multiplicar por 100 para obtener puntos porcentuales. ρ = P(seco hoy | seco ayer) − P(seco hoy | húmedo ayer). La calibración anual histórica no impone un valor de ρ ni su cambio. El IC de cambio de ρ incluye incertidumbre por años, pero no incertidumbre de calibración. Fuente: `07_persistencia.csv`.')
mk20=mark[mark.D==20]
section('12. Diagnóstico Markov: compatibilidad con cambio de frecuencia',
md(mk20[['region','RR_obs_global','RR_solo_frec_global','solo_frec_dentro_IC95_obs_global','RR_obs_local','RR_solo_frec_local','cobertura_markov_hist_local','solo_frec_dentro_IC95_obs_local']])+
'\n\nCompatibilidad en los casos con al menos 100 rachas históricas:\n\n'+md(pd.DataFrame([
{'calibracion':label,'casos_Nhist_ge100':int((mark['N_hist_obs_'+label]>=100).sum()),'compatibles':int(((mark['N_hist_obs_'+label]>=100)&mark['solo_frec_dentro_IC95_obs_'+label]).sum())} for label in ('global','local')]))+
'\n\nLa columna `cobertura_markov_hist_local` es el cociente entre conteos históricos simulados por Markov y observados. Valores alejados de uno revelan limitaciones para reproducir el nivel histórico de rachas largas y deben acompañar su interpretación, especialmente en Los Lagos. El RR por solo frecuencia es una referencia puntual de Markov de primer orden. Que caiga dentro del IC95 observado expresa compatibilidad; no demuestra una causa exclusiva. La dispersión entre simulaciones no incluye incertidumbre de estimación de los parámetros. Fuente: `08_markov.csv`.')
section('13. Sensibilidades de cobertura, regrillado y censura',
'### Mínimo de rachas\n\n'+md(mins[['minimo_rachas','celdas_global','celdas_local','delta_media_promedio_global','delta_media_promedio_local','delta_t99_promedio_global','delta_t99_promedio_local']])+
'\n\nAl elevar el mínimo cambia la selección espacial; no es un experimento que solo cambie el tamaño de muestra dentro de las mismas celdas. Fuente: `10_minimo_rachas.csv`.\n\n'
'### Regrillado\n\n'+md(read(L,'_test_regrillado/mapas_lineal_vs_area.csv').query("criterio in ['i','global']"))+
'\n\nEn esta tabla `i` es el análisis local principal y `global` conserva el contraste escalar. Los umbrales locales se recalibran para cada versión histórica de CR2MET. El promedio alternativo selecciona centros de celdas y no es un regrillado conservativo por áreas de solape.\n\n'
'### Rachas censuradas\n\n'+md(cens[['serie','cuenca','rachas_completas_global','rachas_completas_local','censuradas_global','censuradas_local','pct_censuradas_global','pct_censuradas_local']])+
'\n\nLa exclusión de rachas censuradas se mantiene. El número excluido puede cambiar porque un umbral distinto une o separa segmentos secos. Fuente: `09_censuradas.csv`.')
# Síntesis automática para no asumir de antemano que se conserva una conclusión.
same=spatial[spatial.muestra=='mismas celdas'];summary=[]
for row in same.itertuples():
    summary.append(f"- **{row.metrica}, mismas celdas:** el sesgo espacial pasa de {row.sesgo_global_dias:.3f} a {row.sesgo_local_dias:.3f} días y el MAE de {row.MAE_global_dias:.3f} a {row.MAE_local_dias:.3f} días.")
for row in rr[rr.escenario=='SSP5-8.5'].itertuples():summary.append(f'- **{row.region}, RR a 20 días:** {row.RR_global:.3f} → {row.RR_local:.3f}; decisión: {row.decision_global} → {row.decision_local}.')
for root,label in ((B,'global'),(L,'local')):
    pp=read(root,'resultados_tesis/contraste_persistencia_regiones.csv')
    summary.append(f"- **Persistencia, {label}:** el IC95 de Δρ incluye cero en {int(pp.rho_IC_incluye_0.sum())} de {len(pp)} regiones.")
section('14. Qué cambia en las conclusiones y en el texto', '\n'.join(summary)+
'\n\nLa formulación metodológica del experimento es: **«Se evalúan las duraciones después de igualar la frecuencia histórica anual de días húmedos en cada celda»**. No corresponde afirmar que se corrigió toda la precipitación ni que el ajuste validó la organización temporal del modelo.\n\n'
'Para adoptar esta versión habría que actualizar el método, la figura de calibración, los mapas principales, las PDFs, Tabla 3, RR y sus intervalos, diagnóstico de persistencia, Markov y los resúmenes del suplemento. Las sensibilidades deben distinguir el umbral local principal, el global alternativo y 1 mm. Este informe no modifica los Docs del paper.\n\n'
'El territorio analizado sigue siendo el mismo dominio del gradiente climático: calibrar por celda no restringe el estudio a las ventanas. Las inferencias regionales mantienen el alcance de las cuatro regiones seleccionadas. Un título coherente posible es «Duración de las rachas secas a lo largo de un gradiente climático de Chile: evaluación de un modelo regional y cambios proyectados a mediados de siglo bajo SSP5-8.5».\n\n'
'La reducción del error medio del dominio no implica una mejora en cada ventana. En particular, debe conservarse en el texto el empeoramiento del sesgo del pool de Loa, junto con la mejora de Maipo y los resultados de Maule y Biobío. Tampoco debe identificarse la calibración de frecuencias con un ajuste de duraciones: ambas siguen siendo estadísticas distintas.\n\n'
'La decisión de adoptar el ajuste local debe responder a esa pregunta científica, no a que produzca errores menores o más resultados significativos.')
section('15. Verificación, integridad y límites',md(pd.DataFrame(checks))+
f"\n\nLa cadena local pasó {cons['checks']} controles cruzados de consistencia: {fmt(cons['passed'])}. El inventario compara {len(inventory)} salidas: {sum(x['igual'] for x in inventory)} iguales y {sum(not x['igual'] for x in inventory)} diferentes o nuevas. Las diferencias esperadas no se tratan como fallos de reproducción: son el objeto del experimento.\n\n"
'Los resultados globales se congelaron desde la raíz al iniciar la prueba. El bootstrap vuelve a calcular los tres criterios desde NetCDF; los controles global y 1 mm permiten verificar que no se alteraron reglas ajenas al umbral. Los scripts originales y sus salidas se verificaron por SHA-256 antes y después. Los datos crudos se enlazan y no se escriben. Sus hashes se verifican adicionalmente en `integridad_datos.json`.\n\n'
'Limitaciones conservadas: una simulación de un modelo; CR2MET es una referencia grillada con incertidumbre; la dependencia espacial no desaparece al agrupar rachas; los umbrales se estiman con el histórico completo y no se validan en un período independiente; el bootstrap no recalibra el umbral en cada réplica; igualar la frecuencia anual no iguala su ciclo estacional; las cantidades de precipitación permanecen intactas. El umbral local es estadístico y no representa un umbral físico de impacto agrícola.\n\n'
'Los controles se adaptaron para verificar relaciones matemáticas y trazabilidad, sin exigir que se repitan las conclusiones científicas del análisis global. El diff preserva los cambios exactos de código.')
section('16. Archivos y reproducción',
'- `ejecucion/`: scripts adaptados, resultados, tablas, figuras y cachés regeneradas.\n'
'- `referencia_global/`: copia congelada de los resultados actuales.\n'
'- `scripts_originales/`: código raíz congelado antes de adaptar.\n'
'- `comparativa/`: trece CSV comparativos y tres figuras.\n'
'- `logs/` y `estado.json`: comandos, tiempos y errores.\n'
'- `cambios_codigo.diff`: diferencias de la adaptación.\n'
'- `controles_consistencia.json`, `controles_comparativa.json`: evidencia de verificación.\n'
'- `scripts_locales_sha256.json`, `originales_sha256_antes.json`, `originales_sha256_despues.json`: integridad.\n\n'
'Entorno: `~/.pyenv/shims/python3.11`; no se instalaron paquetes. Para reanudar pasos interrumpidos: `python ejecutar_local.py`. Para regenerar el informe tras completar los cálculos: `python generar_comparativa.py`. `preparar.py` documenta la preparación inicial y no debe ejecutarse sobre una carpeta ya preparada. Para repetir desde cero en una carpeta nueva: `python repetir_desde_cero.py --destino /ruta/nueva`. Este programa conserva los scripts finales y la referencia congelada, enlaza los NetCDF originales y regenera todas las cachés y salidas locales.')
(R/'INFORME_COMPARATIVO.md').write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps(dict(controles=checks,informe=str(R/'INFORME_COMPARATIVO.md')),ensure_ascii=False,indent=2))
if not all(c['passed'] for c in checks):raise SystemExit(2)
