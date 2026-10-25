"""Prepara experimento nuevo sin alterar ningún análisis anterior."""
from pathlib import Path
import sys,shutil,json,hashlib,difflib
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent;ROOT=R.parent
BASE=ROOT/'temporal_umbral_por_celda_2026-10-24';OLDG=ROOT/'umbral_0p1mm_nacional'
sys.path.insert(0,str(ROOT/'reproducibilidad'))
from ejecutar import outputs,manifest,STEPS
W=R/'ejecucion'
for p in (W,R/'referencia_local_1mm',R/'referencia_global_0p1mm',R/'scripts_base_local_1mm',R/'comparativa',R/'logs'):p.mkdir(exist_ok=True)
for p in (BASE/'ejecucion').glob('*.py'):
    shutil.copy2(p,W/p.name);shutil.copy2(p,R/'scripts_base_local_1mm'/p.name)
for name in ('pr','pr1'):(W/name).symlink_to(ROOT/name,target_is_directory=True)
for source,dest in ((BASE/'ejecucion',R/'referencia_local_1mm'),(OLDG,R/'referencia_global_0p1mm')):
    for rel,p in outputs(source).items():
        if source==OLDG and rel=='_test_regrillado/cr2met_lineal_vs_area.npz':continue
        q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
# Hash de todos los archivos de la prueba local previa, más scripts/salidas de raíz y del global 0,1.
protected={}
for p in BASE.rglob('*'):
    if p.is_file() and not p.is_symlink() and 'pr' not in p.relative_to(BASE).parts and 'pr1' not in p.relative_to(BASE).parts:
        protected[p.relative_to(ROOT).as_posix()]=p
for source in (ROOT,OLDG):
    for p in source.glob('_*.py'):protected[p.relative_to(ROOT).as_posix()]=p
    for rel,p in outputs(source).items():protected[p.relative_to(ROOT).as_posix()]=p
(R/'versiones_anteriores_sha256_antes.json').write_text(json.dumps(manifest(protected),indent=2))
shutil.copy2(ROOT/'resultados_tesis/sha256_datos_crudos.txt',R/'datos_sha256_registrados.txt')
(R/'configuracion.json').write_text(json.dumps(dict(repo_origen=str(ROOT),fecha='2026-10-25',referencia_cr2met_mm=0.1,
    comparacion_principal='local 1 mm frente a local 0,1 mm',umbral_global_auxiliar_mm=3.668,
    conservados=[str(BASE),str(OLDG),str(ROOT)]),indent=2,ensure_ascii=False))

def ch(name,old,new):
    p=W/name;s=p.read_text();assert old in s,(name,old);p.write_text(s.replace(old,new))
# Umbrales de referencia en todas las operaciones que construyen rachas o frecuencias.
ch('_run_pregunta7_bootstrap.py','TAU_CR2MET_REF = 1.0','TAU_CR2MET_REF = 0.1')
ch('_run_pregunta7_bootstrap.py','TAU_ALADIN_DOMINIO = 5.285','TAU_ALADIN_DOMINIO = 3.668')
ch('_generar_pregunta9_pdf_loglog_cuencas.py','TAU_CR2MET_REF = 1.0','TAU_CR2MET_REF = 0.1')
ch('_generar_figS1_sensibilidad_umbral.py',"'ALADIN R0 global': f6.extract_basin_spell_records(pr_ala_m, 5.285, bm)","'ALADIN R0 global': f6.extract_basin_spell_records(pr_ala_m, 3.668, bm)")
for name in ('_generar_figS1_sensibilidad_umbral.py','_calcular_tabla3_ventanas.py'):
    ch(name,'CR2MET 1 mm','CR2MET 0.1 mm');ch(name,'ALADIN 1 mm','ALADIN 0.1 mm')
ch('_calcular_tabla3_ventanas.py','ii (1 mm)','ii (0.1 mm)')
ch('_generar_tablaS1_sensibilidad_chile.py','i — R0 global (5,285 mm)','i — R0 global (3,668 mm)')
ch('_generar_tablaS1_sensibilidad_chile.py','ii — 1 mm en ambos','ii — 0.1 mm en ambos')
ch('_calcular_frecuencia_humeda_ventanas.py','f(c, 1.0)','f(c, 0.1)')
ch('_calcular_frecuencia_humeda_ventanas.py','f(a, 1.0)','f(a, 0.1)')
ch('_calcular_frecuencia_humeda_ventanas.py','1mm','0p1mm')
ch('_test_regrillado_conservativo.py','TAU_CR2 = 1.0','TAU_CR2 = 0.1')
ch('_test_regrillado_conservativo.py','R0_ACTUAL = 5.285','R0_ACTUAL = 3.668')
ch('_test_regrillado_conservativo.py','cr2_1mm','cr2_0p1mm')
ch('_cuantificar_censuradas.py',"'CR2MET 1980-2014 (1 mm)': (np.load(CACHE_CR2)['lin'], 1.0)","'CR2MET 1980-2014 (0.1 mm)': (np.load(CACHE_CR2)['lin'], 0.1)")
ch('_generar_figT01_referencia_sin_memoria.py','THRESHOLD = 1.0','THRESHOLD = 0.1')
ch('_generar_fig02_esquema_dryspell.py','R = 1.0','R = 0.1')
ch('_generar_figS2_mapa_R0_local.py','5.285','3.668')
# No emplear el valor numérico 5.285 como selector de un campo: CLI explícita.
p=W/'_run_pregunta5_aladin.py';s=p.read_text()
s=s.replace("parser.add_argument('--umbral', type=float, default=5.285,","parser.add_argument('--umbral', type=float, default=None,")
s=s.replace("is_main = DRY_THRESHOLD == 5.285 and FUTURE_SCENARIO == 'ssp585'", "is_main = args.umbral is None and FUTURE_SCENARIO == 'ssp585'")
s=s.replace("('' if DRY_THRESHOLD == 5.285 else f'_{DRY_THRESHOLD:g}mm')", "('' if args.umbral is None else f'_{args.umbral:g}mm')")
s=s.replace('if args.umbral == 5.285:', 'if args.umbral is None:')
s=s.replace('El valor 5.285 actúa como selector heredado de ese campo.', 'Un valor explícito selecciona la sensibilidad escalar.')
p.write_text(s)
# Calibración de campo, con el mismo algoritmo y ampliación del límite si se necesita.
ch('calibrar_local.py','ft=(cv>=1).mean(0)','ft=(cv>=0.1).mean(0)')
ch('calibrar_local.py','glob=(av>=5.285).mean(0)','glob=(av>=3.668).mean(0)')
ch('calibrar_local.py','referencia_cr2met_mm=1.0','referencia_cr2met_mm=0.1')
ch('diagnostico_regional.py','ref=f(cv,1); fg=f(av,5.285);fl=f(av,t[s])','ref=f(cv,0.1); fg=f(av,3.668);fl=f(av,t[s])')
ch('figura_calibracion_local.py',"referencia_global/paper_figuras/fig03_calibracion_R0_ftau.png", "referencia_global_0p1mm/paper_figuras/fig03_calibracion_R0_ftau.png")
ch('figura_calibracion_local.py','5.285','3.668');ch('figura_calibracion_local.py','5,285','3,668');ch('figura_calibracion_local.py','CR2MET a 1 mm/día','CR2MET a 0,1 mm/día')
# La calibración escalar auxiliar se regenera desde NetCDF (misma definición que global 0,1 previo).
shutil.copy2(OLDG/'_generar_fig03_calibracion_R0.py',W/'_generar_fig03_calibracion_R0.py')
# Comprobaciones de relaciones matemáticas: no exigir conclusiones ni coberturas del experimento 1 mm.
ch('verificar_consistencia_local.py','CR2MET 1 mm','CR2MET 0.1 mm')
ch('verificar_consistencia_local.py','_1mm','_0.1mm')
ch('verificar_consistencia_local.py','CR2MET 1980-2014 (1 mm)','CR2MET 1980-2014 (0.1 mm)')
ch('verificar_consistencia_local.py','== 5.285','== 3.668')
# Descripciones: reflejar la referencia 0,1, sin reemplazar constantes probabilísticas.
for p in W.glob('*.py'):
    s=p.read_text();s=s.replace('CR2MET 1980-2014, umbral 1 mm','CR2MET 1980-2014, umbral 0,1 mm').replace('umbral 1 mm en CR2MET','umbral 0,1 mm en CR2MET').replace('CR2MET a 1 mm','CR2MET a 0,1 mm').replace('CR2MET con 1 mm','CR2MET con 0,1 mm')
    p.write_text(s)
steps=[['calibrar_local.py']]+STEPS
steps=[([c[0],'--umbral','0.1'] if c==['_run_pregunta5_aladin.py','--umbral','1'] else c) for c in steps]
steps += [['diagnostico_regional.py'],['figura_calibracion_local.py'],['verificar_consistencia_local.py','--ejecucion',str(W),'--salida',str(R/'controles_consistencia.json')],['../verificar_integridad.py'],['../generar_comparativa.py']]
(R/'pasos.json').write_text(json.dumps(steps,indent=2))
d=[]
for p in (R/'scripts_base_local_1mm').glob('*.py'):
    d.extend(difflib.unified_diff(p.read_text().splitlines(True),(W/p.name).read_text().splitlines(True),fromfile='local_1mm/'+p.name,tofile='local_0p1mm/'+p.name))
(R/'cambios_codigo.diff').write_text(''.join(d));print('Preparado:',R)
