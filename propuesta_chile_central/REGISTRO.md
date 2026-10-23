# Registro de la propuesta «Chile central» (script → salida)

Estado al **2026-10-22**. Carpeta autocontenida: borrarla no afecta nada del repo. Reglas en `REGLA.md`. Datos crudos y cachés del dominio completo se leen mediante enlaces simbólicos (`pr`, `pr1`, `_pregunta7_outputs/metrics_integrado_cache.npz`, `_test_regrillado/cr2met_lineal_vs_area.npz`). Hashes SHA-256 de las salidas: `resultados_tesis/sha256_salidas.txt` (no incluye la caché de 350 MB, que es del repo).

Entorno: `~/.pyenv/shims/python3.11` (desde esta carpeta). Todos los scripts se ejecutaron **una sola vez** con estas unidades (la figura de RR se repitió solo para corregir rótulos de los ejes). Tiempo total ≈ 3 min en un Mac de 10 núcleos.

## Unidades
`unidades.py`: Valparaíso, Metropolitana de Santiago, O'Higgins y Maule (Natural Earth admin_1, 1:10 m), de norte a sur: 108, 100, 110 y 184 celdas (iguales con y sin máscara de Chile; sin solapes, comprobado con `assert`).

## Orden de ejecución (desde esta carpeta)
```
_generar_pregunta9_pdf_loglog_cuencas.py     → duraciones por región (caché) y PDFs
_generar_figS1_sensibilidad_umbral.py        → duraciones con los criterios ii, i y iii (R0 local por bisección)
_calcular_tabla3_ventanas.py                 → Tabla 3 y sesgo relativo por criterio
_calcular_frecuencia_humeda_ventanas.py      → fracción de días húmedos por región, hist y futuro
_calcular_sesgo_ventanas.py                  → Δ (mapas de dominio completo) promediado por región
_cuantificar_censuradas.py                   → rachas censuradas por región y producto
_calcular_estacionalidad_marnov_regiones.py  → precipitación anual y fracción mar–nov
_run_pregunta5_aladin.py                     → risk ratios, principal
_run_pregunta5_aladin.py --umbral 1          → sensibilidad 1 mm
_run_pregunta5_aladin.py --escenario ssp370  → sensibilidad SSP3-7.0
_referencia_markov_rr_regiones.py            → referencia Markov (necesita el RR principal)
_contraste_persistencia_regiones.py          → frecuencia seca y ρ
_regrillado_unidades.py                      → regrillado lineal vs área por región
_generar_fig_dominio_ventanas_regiones.py    → Figura 1 nueva
_generar_fig_rr_regiones.py <carpeta> <png>  → curvas RR (3 variantes)
_verificacion_independiente.py               → comprobación con otro código
```

## Mapa de resultados
| Elemento | Script | Salida |
|---|---|---|
| Figura 1 (regiones) | `_generar_fig_dominio_ventanas_regiones.py` | `paper_figuras/fig01_dominio_regiones_chile_central.png` |
| Figura de PDFs por región (mismo estilo que el panel anterior, con letras) | `_generar_pregunta9_pdf_loglog_cuencas.py` | `paper_figuras/fig_pdf_regiones_loglog_panel.png`, `_pregunta9_outputs/` |
| Tabla 3 | `_calcular_tabla3_ventanas.py` | `resultados_tesis/tabla3_estadisticos_ventanas.csv` |
| Sesgo relativo por criterio (Tabla S2) | `_calcular_tabla3_ventanas.py` | `resultados_tesis/sesgo_relativo_criterios_ventanas.csv` |
| Figura S1 y resumen S2 | `_generar_figS1_sensibilidad_umbral.py` | `paper_figuras/figS1_sensibilidad_umbral_regiones.png`, `_pregunta9_calibracion_outputs/` |
| Días húmedos por región | `_calcular_frecuencia_humeda_ventanas.py` | `resultados_tesis/frecuencia_humeda_ventanas.csv` |
| Δ medio y t99 por región (mapas) | `_calcular_sesgo_ventanas.py` | `resultados_tesis/sesgo_por_ventana.csv` |
| Rachas censuradas | `_cuantificar_censuradas.py` | `_test_regrillado/rachas_censuradas_por_cuenca.csv` |
| Precipitación anual y fracción mar–nov | `_calcular_estacionalidad_marnov_regiones.py` | `resultados_tesis/estacionalidad_marnov_regiones.csv` |
| Tabla 4 y curvas RR | `_run_pregunta5_aladin.py` | `outputs_pregunta5_aladin/results.json`, `paper_figuras/fig_rr_regiones.png` |
| RR con 1 mm (S3/S4) | `_run_pregunta5_aladin.py --umbral 1` | `outputs_pregunta5_aladin_1mm/results.json`, `paper_figuras/figS_rr_regiones_1mm.png` |
| RR con SSP3-7.0 (S5/S6) | `_run_pregunta5_aladin.py --escenario ssp370` | `outputs_pregunta5_aladin_ssp370/results.json`, `paper_figuras/figS_rr_regiones_ssp370.png` |
| Frecuencia frente a agrupamiento | `_referencia_markov_rr_regiones.py` | `resultados_tesis/descomposicion_markov_rr_regiones.csv` |
| Persistencia de 1.er orden | `_contraste_persistencia_regiones.py` | `resultados_tesis/contraste_persistencia_regiones.csv` |
| Regrillado por región | `_regrillado_unidades.py` | `_test_regrillado/regiones_cr2met_lineal_vs_area.csv` |
| Verificación independiente | `_verificacion_independiente.py` | `resultados_tesis/verificacion_independiente.txt` |

## Lo que NO se recalcula (no depende de las unidades; vale lo del repo)
Calibración de R₀ y Figura 3; mapas de duración media y t99, Δ, significancia y Tabla S1 (`_run_pregunta7_bootstrap.py`); sensibilidad al mínimo de rachas (`_sensibilidad_minimo_rachas.py`); mapa de R₀ local (Fig. S2); regrillado de dominio completo (Tabla S4, filas de Δ).

## Diferencias respecto de los scripts del repo
- `unidades.py` (nuevo) define las unidades; `_run_pregunta5_aladin.py` solo cambia `REGION_SPECS`; `_generar_pregunta9_…`, `_calcular_frecuencia_…`, `_calcular_sesgo_ventanas.py`, `_cuantificar_censuradas.py` y `_generar_figS1_…` sustituyen las cajas de latitud–longitud por las máscaras de región; `_generar_fig_rr_regiones.py` recibe carpeta y nombre de salida por argumento y ajusta el eje y a los datos.
- `_run_pregunta7_bootstrap.py` y `_test_regrillado_conservativo.py` son copias **sin cambios**, necesarias solo para importar funciones. No ejecutar sus `main()` aquí.
- Los nombres de archivo de las salidas (`…ventanas…`, `…cuencas…`) se conservaron para poder comparar con el repo; en esta propuesta designan las regiones.

## Textos v2 (Introducción v2 y Material suplementario v2)
Los textos se generan desde las salidas de esta carpeta; ninguna cifra de las tablas ni de los párrafos con números se escribió a mano.

| Qué | Script | Salida |
|---|---|---|
| Cifras derivadas citadas en el texto (RR por D, frecuencias, razones de precipitación, censuradas > p99, etc.) | `_cifras_derivadas.py` | `resultados_tesis/cifras_derivadas.json` |
| Formateo y lectura de salidas | `v2/_v2_common.py` | (módulo) |
| Párrafos y tablas T1–T4 de la Introducción v2 | `v2/_v2_paper.py` | `v2/v2_paper_preview.txt` |
| Párrafos y tablas TS1–TS6 del suplemento v2 | `v2/_v2_supl.py` | `v2/v2_supl_preview.txt` |
| Troceado para subir a los Docs y metadatos | `v2/_v2_build.py` | `v2/paper_chunk_*.json`, `v2/supl_chunk_1.json`, `v2/v2_meta.json` |
| Estilos y tablas (solicitudes a la API de Docs) | `v2/_v2_stageB.py` | `v2/stageB_*.json` |
| Verificación Doc ↔ texto generado (párrafos, tablas celda a celda, títulos, marcadores de figura) | `v2/_v2_check_full.py` | (impresión en consola) |

Las figuras que el usuario pega en los Docs están copiadas, sin modificar, en `figuras_para_pegar/` (hashes en `figuras_para_pegar_sha256.txt`). Las cuatro de dominio completo (Fig. 2, 3, 4 y S2) vienen de `../paper_figuras/`; las seis restantes, de `paper_figuras/` de esta carpeta.

Verificación hecha el 2026-10-22: el texto de ambos Docs coincide con los párrafos generados (Introducción 116/116, suplemento 34/34), las 11 tablas coinciden celda a celda con `v2_meta.json`, y las Tablas 3 y 4 se contrastaron además contra `tabla3_estadisticos_ventanas.csv` y `outputs_pregunta5_aladin/results.json` con un código aparte (104 celdas, 0 diferencias).

## Figura extra (no está en los Docs v2)
| Qué | Script | Salida |
|---|---|---|
| Cuánto acorta ALADIN las rachas, por región y todo el país (Δ media y Δ p99, promedio y mediana por celda) | `_generar_fig_acorta_regiones.py` | `paper_figuras/fig_acorta_regiones.png` (copia en `figuras_para_pegar/`) |
| % de celdas con diferencia significativa ALADIN − CR2MET por región (no está en los Docs v2) | `_calcular_significancia_celdas_regiones.py` | `resultados_tesis/significancia_celdas_regiones.csv` (usa `../resultados_tesis/bootstrap_por_celda_A_opcion_B_integrado_*.npz`, solo lectura; reproduce 66,3 % y 43,1 % de todo el dominio) |
| Barras: fracción de días húmedos y duración media (CR2MET frente a ALADIN con 1 mm y con R₀), valores de la Tabla S1 y de `calibracion_R0.json` (no está en los Docs v2) | `_generar_fig_barras_calibracion.py` | `paper_figuras/fig_barras_calibracion.png` (copia en `figuras_para_pegar/`) |
| Duración media de CR2MET (1 mm), ALADIN (1 mm) y ALADIN (R₀) sobre las mismas 3 526 celdas | `_calcular_duracion_media_celdas_comunes.py` | `resultados_tesis/duracion_media_celdas_comunes_i.csv` (ALADIN 1 mm recalculado con `_run_pregunta7_bootstrap.py`; no está en la Tabla S1) |
| Dominio: 3 587 celdas ALADIN, 4 regiones y franja de Latoja (32°–40°S, solo referencia; no está en los Docs v2) | `_generar_fig_dominio_celdas.py` | `paper_figuras/fig_dominio_celdas.png` (copia en `figuras_para_pegar/`) |
