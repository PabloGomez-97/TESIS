"""Copia y adapta exclusivamente la cadena raíz del paper; no modifica originales."""
from pathlib import Path
import shutil, json, hashlib, difflib
ROOT=Path(__file__).resolve().parents[1]
RUN=Path(__file__).resolve().parent
W=RUN/'ejecucion'; REF=RUN/'referencia_global'; ORIG=RUN/'scripts_originales'
for p in (W,REF,ORIG,RUN/'logs',RUN/'comparativa'): p.mkdir(exist_ok=True)
import sys
sys.path.insert(0,str(ROOT/'reproducibilidad'))
from ejecutar import OUTPUT_DIRS, outputs, manifest, STEPS
sources=sorted(ROOT.glob('_*.py'))
for p in sources:
    shutil.copy2(p,W/p.name); shutil.copy2(p,ORIG/p.name)
for name in ('pr','pr1'): (W/name).symlink_to(ROOT/name,target_is_directory=True)
for rel,p in outputs(ROOT).items():
    d=REF/rel; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,d)
for name in ('GUIA_EXPLICATIVA_DEL_PAPER_PARA_TESIS.md','REGISTRO_RESULTADOS.md','PROCEDENCIA_DATOS.md','CLAUDE.md'):
    shutil.copy2(ROOT/name,RUN/name)
tracked={p.relative_to(ROOT).as_posix():p for p in sources}
tracked.update(outputs(ROOT))
(RUN/'originales_sha256_antes.json').write_text(json.dumps(manifest(tracked),indent=2))
(RUN/'datos_sha256_registrados.txt').write_bytes((ROOT/'resultados_tesis/sha256_datos_crudos.txt').read_bytes())

def change(name,old,new):
    p=W/name; s=p.read_text(); assert old in s,(name,old); p.write_text(s.replace(old,new))

# Conservamos los tres criterios originales en el bootstrap. El principal mostrado pasa a C.
change('_run_pregunta7_bootstrap.py',"if scen_key == 'A_opcion_B_integrado':","if scen_key == 'C_umbral_local':")
# Ampliar el intervalo SOLO en series que no se pueden calibrar en [0,15].
change('_run_pregunta7_bootstrap.py',
"    fmax = wet_fraction_1d(pr_series, tau_max)",
"    fmax = wet_fraction_1d(pr_series, tau_max)\n    if target_fraction < fmax - tol:\n        tau_max = float(np.nextafter(np.nanmax(pr_series), np.inf))\n        fmax = wet_fraction_1d(pr_series, tau_max)")
change('_generar_pregunta9_pdf_loglog_cuencas.py',
"    lo, hi = 0.0, BISECTION_TAU_MAX",
"    upper = BISECTION_TAU_MAX\n    if target_fraction < fmax - BISECTION_TOL:\n        upper = float(np.nextafter(np.nanmax(pr_series), np.inf))\n        fmax = wet_fraction_1d(pr_series, upper)\n    lo, hi = 0.0, upper")
# El chequeo del rango debe ocurrir después de expandir el intervalo.
change('_generar_pregunta9_pdf_loglog_cuencas.py',
"    if target_fraction > f0 + BISECTION_TOL or target_fraction < fmax - BISECTION_TOL:\n        return np.nan\n    upper = BISECTION_TAU_MAX",
"    upper = BISECTION_TAU_MAX")
change('_generar_pregunta9_pdf_loglog_cuencas.py',"    lo, hi = 0.0, upper",
"    if target_fraction > f0 + BISECTION_TOL or target_fraction < fmax - BISECTION_TOL:\n        return np.nan\n    lo, hi = 0.0, upper")
# Única fuente del campo de umbrales, calculado antes de los demás pasos.
change('_generar_pregunta9_pdf_loglog_cuencas.py','TAU_ALADIN_DOMINIO = 5.285',
'from umbral_local import campo_umbral\nTAU_ALADIN_DOMINIO = campo_umbral()')
change('_generar_pregunta9_pdf_loglog_cuencas.py',"SCEN_TITLE = 'Criterio i — R0 global (5.285 mm en ALADIN)'", "SCEN_TITLE = 'Umbral local por celda, calibrado en 1980–2014 y fijo en el futuro'")
# S1 conserva los TRES criterios reales: la serie global sigue siendo 5.285.
change('_generar_figS1_sensibilidad_umbral.py',"'ALADIN R0 global': f6.extract_basin_spell_records(pr_ala_m, f6.TAU_ALADIN_DOMINIO, bm)","'ALADIN R0 global': f6.extract_basin_spell_records(pr_ala_m, 5.285, bm)")
# RR: selector CLI escalar para conservar las rutas; convertir al campo después de elegirlas.
change('_run_pregunta5_aladin.py','DRY_THRESHOLD = 5.285  # tau* ALADIN (Pregunta 6)',
'from umbral_local import campo_umbral\nDRY_THRESHOLD = campo_umbral()  # histórico fijo, usado también por Markov y contraste')
change('_run_pregunta5_aladin.py',"parser.add_argument('--umbral', type=float, default=DRY_THRESHOLD,", "parser.add_argument('--umbral', type=float, default=5.285,")
change('_run_pregunta5_aladin.py',"    print('1/4: Cargando ALADIN historico y futuro...')", "    if args.umbral == 5.285:\n        DRY_THRESHOLD = campo_umbral()\n    print('1/4: Cargando ALADIN historico y futuro...')")
change('_run_pregunta5_aladin.py',"print(f'Dia seco ALADIN: pr < {DRY_THRESHOLD} mm/dia | Mar-Nov')", "print('Dia seco ALADIN: umbral local por celda (o sensibilidad CLI) | Mar-Nov')")
# Comparaciones NumPy necesitan el campo seleccionado en el MISMO orden de celdas.
for name,old,new in (
('_referencia_markov_rr_regiones.py',' < R0',' < R0.values[jj, ii]'),
('_contraste_persistencia_regiones.py',' < rr.DRY_THRESHOLD',' < rr.DRY_THRESHOLD.values[jj, ii]')): change(name,old,new)
# Frecuencias por ventana: seleccionar también el vector de umbrales de esa ventana.
change('_calcular_frecuencia_humeda_ventanas.py',"f = lambda x, t: float((x[:, cols] >= t).mean(axis=0).mean() * 100)",
"f = lambda x, t: float((x[:, cols] >= (t.values[jj, ii][cols] if hasattr(t, 'dims') else t)).mean(axis=0).mean() * 100)")
# Mapa local: línea negra de referencia global, no un arreglo en axvline.
change('_generar_figS2_mapa_R0_local.py','f6.TAU_ALADIN_DOMINIO','5.285')
change('_sensibilidad_minimo_rachas.py','import _run_pregunta7_bootstrap as b', 'import _run_pregunta7_bootstrap as b\nfrom umbral_local import campo_umbral')
change('_sensibilidad_minimo_rachas.py','b.TAU_ALADIN_DOMINIO, mask','campo_umbral(), mask')
change('_sensibilidad_minimo_rachas.py','bootstrap_por_celda_A_opcion_B_integrado_', 'bootstrap_por_celda_C_umbral_local_')
change('_cuantificar_censuradas.py','R0 = 5.285','from umbral_local import campo_umbral\nR0 = campo_umbral()')
change('_cuantificar_censuradas.py','(pr[:, k] < thr)', '(pr[:, k] < (thr[k] if np.ndim(thr) else thr))')
change('_cuantificar_censuradas.py',"(load_aladin('historical', '1980-01-01', '2014-12-31', jj, ii), R0)","(load_aladin('historical', '1980-01-01', '2014-12-31', jj, ii), R0.values[jj, ii])")
change('_cuantificar_censuradas.py',"(load_aladin('ssp585', '2040-01-01', '2074-12-31', jj, ii), R0)","(load_aladin('ssp585', '2040-01-01', '2074-12-31', jj, ii), R0.values[jj, ii])")
change('_cuantificar_censuradas.py','spells_with_censoring(pr[:, sel], thr)', 'spells_with_censoring(pr[:, sel], thr[sel] if np.ndim(thr) else thr)')
# Sensibilidad de regrillado: i pasa a local calibrado para cada versión de referencia.
change('_test_regrillado_conservativo.py',"for crit, (ma, ta) in (('i', m_ala['i']), ('ii', m_ala['ii']), ('iii', m_ala['iii_' + name])):",
"for crit, (ma, ta) in (('i', m_ala['iii_' + name]), ('ii', m_ala['ii']), ('iii', m_ala['iii_' + name]), ('global', m_ala['i'])):")
# Validación reutilizada, reemplazando solo aserciones específicas del resultado global.
s=(ROOT/'reproducibilidad/verificar_consistencia.py').read_text()
s=s.replace('(\"ALADIN (hist)\", \"ALADIN R0 global\")','(\"ALADIN (hist)\", \"ALADIN R0 local\")')
s=s.replace('bootstrap_por_celda_A_opcion_B_integrado_{metric}', 'bootstrap_por_celda_C_umbral_local_{metric}')
s=s.replace('valid.sum() == 3526 and np.array_equal(valid, b["valida"])','np.array_equal(valid, b["valida"])')
# Hipótesis científicas se INFORMAN en el informe, no son requisitos de consistencia.
a=s.index('    selected = markov[markov.N_hist_obs >= 100]'); b=s.index('    censored = ',a)
s=s[:a]+s[b:]
(W/'verificar_consistencia_local.py').write_text(s)
diffs=[]
for p in sources:
    diffs.extend(difflib.unified_diff(p.read_text().splitlines(True),(W/p.name).read_text().splitlines(True),fromfile='original/'+p.name,tofile='local/'+p.name))
(RUN/'cambios_codigo.diff').write_text(''.join(diffs))
(RUN/'pasos.json').write_text(json.dumps(STEPS,indent=2))
print(RUN)
