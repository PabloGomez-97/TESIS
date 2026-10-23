# Registro de trazabilidad: script → salida → figura/tabla/cifra

Estado al **2026-10-21**. Fuente única de qué archivo respalda qué número. Todo lo que no figura aquí no está respaldado y no debe citarse.
Procedencia de los datos crudos: `PROCEDENCIA_DATOS.md`. Hashes SHA-256: `resultados_tesis/sha256_datos_crudos.txt` (entradas) y `resultados_tesis/sha256_salidas.txt` (salidas).

## 1. Verificación hecha el 2026-10-21

1. **Réplica desde datos crudos.** Se copiaron los scripts a una carpeta aparte (sin salidas previas, con `pr/` y `pr1/` enlazados) y se ejecutaron completos, en el orden de §3. Las salidas coinciden **exactamente** (diferencia máxima 0, arreglos idénticos) con las guardadas en el repo: `resumen_pregunta7.csv`, `bootstrap_summary.csv`, `metrics_integrado_cache.npz`, los 6 `bootstrap_por_celda_*.npz`, `sesgo_por_ventana.csv`, `tablaS1`, los 3 `results.json` de risk ratios, `descomposicion_markov_rr_regiones.csv`, las duraciones por ventana (`pdf_durations_i_global.pkl`, recalculadas sin caché), `resumen_sensibilidad_umbral_4cuencas.csv` (S1), `R0_local_map.npz`, los tres CSV y el JSON del regrillado, el caché `cr2met_lineal_vs_area.npz`, `rachas_censuradas_por_cuenca.csv` y `figT01_referencia_sin_memoria.csv`. Calibración: R₀ = 5,2846 mm (se usa 5,285), F = 20,94 % y 30,68 %. Tiempo total ≈ 20 min en un Mac de 10 núcleos / 16 GB.
1b. **Segunda réplica completa (2026-10-21, 19:07), tras la regla de celdas comunes.** Se repitió toda la cadena (22 scripts, desde datos crudos, en carpeta aparte con los scripts actuales). Se compararon 46 salidas: todos los CSV (incluidos `resumen_pregunta7.csv` con la columna nueva `n_pixeles_propios`, la Tabla S1, `sesgo_por_ventana`, `contraste_persistencia_regiones`, `tabla3`, `frecuencia_humeda_ventanas`, `estacionalidad_marnov_regiones`), los `.npz` (caché de mapas, 6 bootstrap por celda, R₀ local, caché de regrillado), los JSON (calibración de R₀, 3 `results.json`, regrillado), los `.pkl` de duraciones y **todas las figuras PNG de `paper_figuras/` píxel a píxel**. Resultado: 46 de 46 idénticas. No se compararon `bootstrap_summary.json` ni los PNG de `_pregunta9_outputs/` (copias de las figuras ya comparadas).
2. **Código independiente.** Los conteos de rachas ≥ 20 d de Coquimbo y La Araucanía (1980–2014 y SSP5-8.5 2040–2074; inicio mar–nov; censuradas excluidas; R₀ = 5,285) se recalcularon con otro código (`scipy.ndimage.label`, máscara con `shapely.contains_xy`) y coinciden con `results.json`: Coquimbo 25 367 → 28 458 (RR 1,1219); La Araucanía 2 397 → 4 118 (RR 1,718); totales 165 734 / 145 786 y 322 205 / 301 531.
3. **Contraste de persistencia** (`_contraste_persistencia_regiones.py`, escrito el 2026-10-21): frecuencia seca, p_dd y ρ de Coquimbo, hist y fut, recalculados con otra implementación (bucles por celda y mes) y coinciden (frec. 0,8636 / 0,8842; p_dd 0,9097 / 0,9226; ρ 0,3073 / 0,2883). No se replicó en carpeta aparte (semilla 11, determinista).
4. **Tabla 3.** Las 12 filas (n, media, mediana, p90, p99, máx.) se reproducen con `_calcular_tabla3_ventanas.py` desde duraciones recalculadas de cero. Coinciden con la Tabla 3 del Doc del paper.

Entorno: Python 3.11.12 (`~/.pyenv/shims/python3.11`), numpy 2.3.4, pandas 2.3.3, xarray 2025.10.1, scipy 1.17.1, shapely 2.1.2, cartopy 0.25.0, matplotlib 3.10.7. `python3` del sistema **no** tiene cartopy.

## 2. Reglas vigentes (idénticas en todos los scripts)

| Regla | Valor |
|---|---|
| Período histórico / futuro | 1980–2014 / 2040–2074 (35 años cada uno; SSP5-8.5 principal, SSP3-7.0 solo sensibilidad de RR) |
| Máscara | celdas ALADIN cuyo centro cae en Chile (Natural Earth 1:10m, `admin_0`): 3 587 celdas; 17,7–45,9°S |
| CR2MET | interpolación lineal a los centros de celda ALADIN; umbral 1 mm |
| ALADIN | pr × 86 400; criterio i: R₀ = 5,285 mm; ii: 1 mm; iii: R₀ local por píxel (bisección) |
| Calibración de R₀ | F(R) medio espacial con igual peso por celda; `calibracion_R0.json` |
| Racha seca | días consecutivos con pr < umbral; se **excluyen** las censuradas (empiezan el 1.er día o terminan el último de la ventana); se asigna al año de inicio |
| Estadísticos por celda | solo con ≥ 30 rachas; los mapas por producto de la Fig. 4 ahora también se dibujan solo en las celdas comunes (la caché sigue guardando las celdas válidas de cada producto: 3 549 CR2MET, 3 550 ALADIN en el criterio i). **Δ, su significancia, los promedios espaciales de cada producto (resumen y Tabla S1) y los paneles de la Fig. 4: solo las celdas comunes** (3 526 en el criterio i; 3 549 en ii; 3 471 en iii) |
| Bootstrap del sesgo | 1 000 réplicas, semilla 42 + índice de celda; **años sorteados de forma independiente** para CR2MET y ALADIN |
| Risk ratios | rachas con inicio mar–nov, D = 5…60 d; 1 000 réplicas, semilla 42; años de cada período sorteados de forma independiente; réplicas sin N_hist* se cuentan y se informan |
| Markov | 1.er orden por celda y mes; R = 40 réplicas; semilla 7 |

## 3. Orden de ejecución (desde datos crudos)

```
~/.pyenv/shims/python3.11 <script>      # desde la raíz del repo
1. _generar_fig03_calibracion_R0.py          → calibración de R0 (Fig. 3, calibracion_R0.json)
2. _run_pregunta7_bootstrap.py               → mapas y bootstrap del sesgo (≈ 11 min, 8 procesos)
3. _calcular_sesgo_ventanas.py
4. _generar_tablaS1_sensibilidad_chile.py
5. _run_pregunta5_aladin.py                  → risk ratios principal
   _run_pregunta5_aladin.py --umbral 1       → sensibilidad 1 mm
   _run_pregunta5_aladin.py --escenario ssp370
6. _referencia_markov_rr_regiones.py         → necesita el paso 5 (principal)
   _contraste_persistencia_regiones.py       → necesita el paso 6 (importa sus funciones)
7. _generar_pregunta9_pdf_loglog_cuencas.py  → duraciones por ventana y Fig. 5 (borrar el .pkl para recalcular)
8. _calcular_tabla3_ventanas.py              → necesita 7 y 10
9. _calcular_frecuencia_humeda_ventanas.py, _calcular_estacionalidad_marnov_regiones.py
10. _generar_figS1_sensibilidad_umbral.py     → necesita 7 (importa sus funciones)
    _generar_figS2_mapa_R0_local.py
11. _test_regrillado_conservativo.py         → caché de CR2MET lineal vs área (350 MB)
12. _cuantificar_censuradas.py, _generar_figT01_referencia_sin_memoria.py   → necesitan 11
13. _generar_fig_rr_regiones.py (necesita el paso 5), _generar_fig05_mapas_persistencia.py, _generar_fig_dominio_ventanas_regiones.py, _generar_fig02_esquema_dryspell.py
14. _sensibilidad_minimo_rachas.py              → necesita el paso 2 (caché de mapas y bootstrap por celda); reimporta sus funciones
```

## 4. Mapa de resultados

Numeración de las figuras y tablas = la del Doc del paper al 2026-10-22 (Fig. 1–6: dominio, esquema, calibración, mapas, PDFs por ventana, RR). `CAMBIOS_AL_PAPER.md` usa la numeración anterior.

| Elemento | Script | Salida |
|---|---|---|
| Fig. 1 (dominio, ventanas, regiones) | `_generar_fig_dominio_ventanas_regiones.py` | `paper_figuras/fig02_dominio_cuencas_regiones.png` |
| Fig. 2 (esquema; ilustrativa, sin datos) | `_generar_fig02_esquema_dryspell.py` | `paper_figuras/fig_esquema_dryspell.png` |
| Fig. 3; F = 20,94 / 30,68 %; R₀ = 5,285 | `_generar_fig03_calibracion_R0.py` | `paper_figuras/fig03_calibracion_R0_ftau.png`, `resultados_tesis/calibracion_R0.json` |
| Fig. 4 (mapas media y t99; los tres paneles de cada fila muestran las mismas 3 526 celdas, ≥ 30 rachas en ambos productos) | `_generar_fig05_mapas_persistencia.py` (lee la caché del paso 2) | `paper_figuras/fig05_mapas_persistencia_criterio_i.png` |
| Δ medio −8,2 d, t99 −46,3 d; % de celdas significativas; Tabla S1 | `_run_pregunta7_bootstrap.py` → `_generar_tablaS1_sensibilidad_chile.py` | `_pregunta7_outputs/resumen_pregunta7.csv`, `outputs_pregunta7_bootstrap/bootstrap_summary.csv`, `paper_figuras/tablaS1_sensibilidad_umbral_chile.csv`, `resultados_tesis/bootstrap_por_celda_*.npz` |
| Sesgo por ventana (promedio de Δ por celda) | `_calcular_sesgo_ventanas.py` | `resultados_tesis/sesgo_por_ventana.csv` |
| Fig. 5 (PDFs por ventana; vigente: panel `fig42_…`, sin letras (a)–(d) en la imagen) | `_generar_pregunta9_pdf_loglog_cuencas.py` | `paper_figuras/fig42_pdf_cuencas_loglog_panel.png`, `paper_figuras/pdf_*_i_global_loglog.png`, `_pregunta9_outputs/pdf_durations_i_global.pkl` |
| Tabla 3 | `_calcular_tabla3_ventanas.py` | `resultados_tesis/tabla3_estadisticos_ventanas.csv` |
| Sensibilidad del sesgo (criterio i) al mínimo de rachas por celda (30, 50, 100, 200): Δ media, Δ t99, medianas, % de celdas significativas. Decidida **después** de ver el resultado con 30; va al suplemento como sensibilidad | `_sensibilidad_minimo_rachas.py` | `resultados_tesis/sensibilidad_minimo_rachas.csv` |
| Sesgo relativo por ventana y criterio (i, ii, iii) | `_calcular_tabla3_ventanas.py` | `resultados_tesis/sesgo_relativo_criterios_ventanas.csv` |
| Frecuencia de días húmedos por ventana; cambio futuro con R₀ fijo | `_calcular_frecuencia_humeda_ventanas.py` | `resultados_tesis/frecuencia_humeda_ventanas.csv` |
| Rachas censuradas (< 1,1 % Loa; < 0,2 % resto) | `_cuantificar_censuradas.py` | `_test_regrillado/rachas_censuradas_por_cuenca.csv` |
| Fig. de las curvas RR(D) por región, versión 2×2 con eje y logarítmico, banda IC95 solo donde no se descartó ninguna réplica y RR/N a D = 20 anotados (reemplaza `fig08_rr_curves_aladin.png` en el paper) | `_generar_fig_rr_regiones.py` (lee `outputs_pregunta5_aladin/results.json`; no recalcula) | `paper_figuras/fig_rr_regiones.png` |
| Fig. 6 (versión antigua; la vigente es `fig_rr_regiones.png`) y Tabla 4 (RR principal) | `_run_pregunta5_aladin.py` | `outputs_pregunta5_aladin/results.json`, `paper_figuras/fig08_rr_curves_aladin.png` (nombre de archivo ≠ número de figura), `spells_*.csv` |
| RR con 1 mm (S4) | `_run_pregunta5_aladin.py --umbral 1` | `outputs_pregunta5_aladin_1mm/results.json`, `paper_figuras/figS_rr_curves_aladin_1mm.png` |
| RR con SSP3-7.0 (S6) | `_run_pregunta5_aladin.py --escenario ssp370` | `outputs_pregunta5_aladin_ssp370/results.json`, `paper_figuras/figS_rr_curves_aladin_ssp370.png` |
| Frecuencia vs agrupamiento: RR esperado solo por frecuencia | `_referencia_markov_rr_regiones.py` | `resultados_tesis/descomposicion_markov_rr_regiones.csv` |
| ¿Cambia la persistencia de 1.er orden? Δ de frecuencia seca, p_dd y ρ por región (mar–nov), IC95 bootstrap de años | `_contraste_persistencia_regiones.py` | `resultados_tesis/contraste_persistencia_regiones.csv` |
| Fracción de precipitación mar–nov (CR2MET 85,0–97,3 %; ALADIN 81,4–95,8 %) | `_calcular_estacionalidad_marnov_regiones.py` | `resultados_tesis/estacionalidad_marnov_regiones.csv` |
| Fig. S1 | `_generar_figS1_sensibilidad_umbral.py` | `paper_figuras/figS1_sensibilidad_umbral_cuencas.png`, `_pregunta9_calibracion_outputs/resumen_sensibilidad_umbral_4cuencas.csv` y `.pkl` |
| Fig. S2 (R₀ local) | `_generar_figS2_mapa_R0_local.py` | `paper_figuras/figS2_mapa_R0_local.png`, `_pregunta7_outputs/R0_local_map.npz` |
| Regrillado lineal vs área | `_test_regrillado_conservativo.py` | `_test_regrillado/*.csv`, `r0_lineal_vs_area.json` |
| Referencia sin memoria (Cap. 2) | `_generar_figT01_referencia_sin_memoria.py` | `paper_figuras/figT01_referencia_sin_memoria.png`, `.csv` |

## 5. Lo que la verificación dejó como límites conocidos (declarar junto a las conclusiones)

1. **R₀ global no iguala la frecuencia dentro de las ventanas.** Fracción de días húmedos CR2MET (1 mm) / ALADIN (R₀): Loa 3,20 / 2,96 %; Maipo 16,12 / 20,27 %; Maule 19,89 / 23,18 %; Biobío 30,74 / 31,64 % (`frecuencia_humeda_ventanas.csv`). El sesgo de persistencia en Maipo y Maule con el criterio i incluye ese exceso residual.
2. **Con R₀ local el sesgo cambia de patrón** (`sesgo_relativo_criterios_ventanas.csv`, media de las duraciones del pool): Loa −23,0 %, Maipo −3,5 %, Maule −13,5 %, Biobío −12,6 %. Con el criterio i: −16,7 / −21,2 / −22,4 / −14,7 %.
3. **ALADIN tiene un sesgo húmedo grande en total anual** (media regional 1980–2014, `estacionalidad_marnov_regiones.csv`): Coquimbo 247 mm (CR2MET) frente a 1 090 mm (ALADIN); O'Higgins 949 / 2 288; La Araucanía 2 048 / 3 635; Los Lagos 2 722 / 5 016. R₀ = 5,285 mm es un umbral estadístico (igual frecuencia), no un umbral físico de «día seco».
4. **61 de 3 587 celdas** no entran en Δ (menos de 30 rachas en algún producto), todas entre 18,4°S y 22,5°S (verificado en la caché de mapas): 23 solo con CR2MET válida, 24 solo con ALADIN válida y 14 con ninguna. Son celdas con muy pocas rachas por período; el promedio de Δ del dominio no las incluye.
5. **Rachas censuradas excluidas**: las más largas se pierden; en Loa, 22–40 rachas censuradas por producto superan el p99 de las completas, y en Maule y Maipo (ALADIN futuro) 39 y 12.
6. **Réplicas bootstrap sin RR definido** (`results.json`, campo «Replicas descartadas»): en el análisis principal 0 a D = 20; Los Lagos pierde 128 de 1 000 a D = 30 y todas desde D = 35; La Araucanía 2 / 39 / 101 a D = 40 / 50 / 60. Con 1 mm, Los Lagos a D = 20 pierde 42 (RR = 0,25, IC [0,014; 28,1]: inestimable).
7. **Metadatos ALADIN inconsistentes** (escenario solo identificable por el nombre de archivo; `driving_experiment_name = historical` en los archivos SSP): ver `PROCEDENCIA_DATOS.md`.
8. **Potencia del contraste de persistencia:** el IC95 de Δρ mide ±0,03 aproximadamente (ρ histórico ≈ 0,30–0,36); cambios menores no son detectables. ρ es de 1.er orden: no mide memoria larga.
9. **Un solo RCM, un solo miembro, un solo escenario principal**: el bootstrap de años mide variabilidad de muestreo, no incertidumbre de la señal.

## 6. Eliminado el 2026-10-21 (respaldo: `~/Documents/respaldo_TESIS_obsoletos_2026-10-21.zip`, 199 archivos, verificado por hash)

- **Salidas obsoletas** (sin script vigente o hechas antes de la regla de rachas censuradas, 13-oct): `fig04_calibracion_umbral_panel`, `fig05_mapas_persistencia_{mean,t99,combined}`, `fig32_persistencia_biases_sec3_2`, `fig33_tendencias_aladin_null_summary` (valores copiados a mano de un notebook), `demostracion_alargamiento_futuro`, `mapas_aladin_*` (la «Figura 8» del paper), `fig_umbral_1mm_vs_R0_*`, `fig_tabla01_datasets*`, `tabla01_datasets*.csv`, y el contenido de `_pregunta9_outputs/` y `_pregunta9_calibracion_outputs/` de julio salvo lo de §4.
- **Scripts**: `_generar_fig33_tendencias_null.py`, `_generar_fig_umbral_1mm_vs_R0.py`, `_generar_tabla01_datasets.py`, `_test_boot.py` (vacío), `_generar_fig32_persistencia_biases.py` y `_generar_figuras_paper_faltantes.py` (reemplazados por `_generar_fig05_mapas_persistencia.py` y `_generar_fig_dominio_ventanas_regiones.py`, que reproducen las figuras 1 y 5 vigentes; la 5 es idéntica píxel a píxel).
- **Notebooks y `Anteriores/`** (12 + 5): exploratorios, ningún resultado vigente depende de ellos.
- **Heredado**: los `.docx` antiguos y sus 6 generadores, `presentacion_tesis/`, `disertacion/`, `tesis/` (LaTeX, con zip de Overleaf), `__pycache__/`.
- **Se conservan**: `Introducción v1 (1).docx`, el PDF de la tesis de olas de calor (modelo de redacción), `Shifting dry spells…pdf` (Latoja et al.), `s40641-016-0050-x.pdf` (Maraun 2016), `Writing-an-Effective-Abstract-2023.pdf`, `VERIFICACION_CITAS.md`.
