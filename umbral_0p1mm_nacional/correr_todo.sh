#!/bin/zsh
# Cadena completa (orden de REGISTRO_RESULTADOS.md §3) con umbral CR2MET = 0,1 mm y R0 = 3,668 mm
cd "$(dirname "$0")"
PY=~/.pyenv/shims/python3.11
run() { n=$1; shift; echo "[$(date +%T)] INICIO $n $*" >> logs/_progreso.log; $PY "$@" > logs/$n.log 2>&1; echo "[$(date +%T)] FIN $n (exit $?)" >> logs/_progreso.log; }
run 01_fig03 _generar_fig03_calibracion_R0.py
run 02_bootstrap7 _run_pregunta7_bootstrap.py
run 03_sesgo_ventanas _calcular_sesgo_ventanas.py
run 04_tablaS1 _generar_tablaS1_sensibilidad_chile.py
run 05a_rr_principal _run_pregunta5_aladin.py
run 05b_rr_0p1mm _run_pregunta5_aladin.py --umbral 0.1
run 05c_rr_ssp370 _run_pregunta5_aladin.py --escenario ssp370
run 06a_markov _referencia_markov_rr_regiones.py
run 06b_contraste _contraste_persistencia_regiones.py
run 07_pdfs _generar_pregunta9_pdf_loglog_cuencas.py
run 08_tabla3 _calcular_tabla3_ventanas.py
run 09a_frec_humeda _calcular_frecuencia_humeda_ventanas.py
run 09b_estacionalidad _calcular_estacionalidad_marnov_regiones.py
run 10a_figS1 _generar_figS1_sensibilidad_umbral.py
run 10b_figS2 _generar_figS2_mapa_R0_local.py
run 11_regrillado _test_regrillado_conservativo.py
run 12a_censuradas _cuantificar_censuradas.py
run 12b_figT01 _generar_figT01_referencia_sin_memoria.py
run 13a_fig_rr _generar_fig_rr_regiones.py
run 13b_fig05 _generar_fig05_mapas_persistencia.py
run 13c_dominio _generar_fig_dominio_ventanas_regiones.py
run 14_min_rachas _sensibilidad_minimo_rachas.py
echo "[$(date +%T)] TODO TERMINADO" >> logs/_progreso.log
