#!/bin/zsh
cd "$(dirname "$0")"
PY=~/.pyenv/shims/python3.11
for s in _referencia_markov_rr_regiones.py _contraste_persistencia_regiones.py; do
  echo "=== $s $(date +%T)" >> logs/cadena_C.log
  $PY -u $s > logs/${s%.py}.log 2>&1 || { echo "FALLO $s" >> logs/cadena_C.log; exit 1; }
done
echo "=== FIN C $(date +%T)" >> logs/cadena_C.log
