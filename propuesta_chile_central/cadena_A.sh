#!/bin/zsh
cd "$(dirname "$0")"
PY=~/.pyenv/shims/python3.11
for s in _generar_pregunta9_pdf_loglog_cuencas.py _generar_figS1_sensibilidad_umbral.py _calcular_tabla3_ventanas.py _calcular_frecuencia_humeda_ventanas.py _calcular_sesgo_ventanas.py _cuantificar_censuradas.py _calcular_estacionalidad_marnov_regiones.py; do
  echo "=== $s $(date +%T)" >> logs/cadena_A.log
  $PY -u $s > logs/${s%.py}.log 2>&1 || { echo "FALLO $s" >> logs/cadena_A.log; exit 1; }
done
echo "=== FIN A $(date +%T)" >> logs/cadena_A.log
