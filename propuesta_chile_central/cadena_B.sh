#!/bin/zsh
cd "$(dirname "$0")"
PY=~/.pyenv/shims/python3.11
echo "=== RR principal $(date +%T)" >> logs/cadena_B.log
$PY -u _run_pregunta5_aladin.py > logs/rr_principal.log 2>&1 || { echo "FALLO rr principal" >> logs/cadena_B.log; exit 1; }
echo "=== RR 1mm $(date +%T)" >> logs/cadena_B.log
$PY -u _run_pregunta5_aladin.py --umbral 1 > logs/rr_1mm.log 2>&1 || { echo "FALLO rr 1mm" >> logs/cadena_B.log; exit 1; }
echo "=== RR ssp370 $(date +%T)" >> logs/cadena_B.log
$PY -u _run_pregunta5_aladin.py --escenario ssp370 > logs/rr_ssp370.log 2>&1 || { echo "FALLO rr ssp370" >> logs/cadena_B.log; exit 1; }
echo "=== FIN B $(date +%T)" >> logs/cadena_B.log
