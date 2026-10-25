# Procedencia de los datos

Registro de lo que **está en los archivos** y de lo que **no está documentado**. Todo lo de la columna «Evidencia» se leyó de los atributos NetCDF o de los nombres de archivo el 2026-10-21 (script ad hoc, sin salida guardada; los hashes sí se guardaron).
Hashes SHA-256 de los 8 + 672 archivos: `resultados_tesis/sha256_datos_crudos.txt`.

## CNRM-ALADIN64 CHP12 (`pr1/`, 8 archivos)

| Dato | Valor | Evidencia |
|---|---|---|
| Variable / unidad | `pr`, kg m⁻² s⁻¹ (×86 400 → mm día⁻¹) | atributo `units` |
| Modelo | `model_id = CNRM-ALADIN64`; `arpege_minor_version = 6.4.1`; `comment = "CHP12 CNRM-ALADIN 6.4 L91"` | atributos globales (idénticos en los 8 archivos) |
| Forzante | CNRM-ESM2-1, miembro r1i1p1f2 | nombre de archivo y `driving_model_id` / `driving_model_ensemble_member` |
| Escenarios | `historical` 1950–1978 y 1979–2014; `ssp370` y `ssp585` 2015–2039, 2040–2064, 2065–2100 | nombre de archivo y eje temporal |
| Calendario | gregoriano estándar (12 784 días en 1980–2014; 9 131 días en 2015–2039, 9 132 en 2040–2064) | eje temporal |
| Grilla | Lambert conforme, 255 × 68 celdas (y × x), lat −45,88° a −17,29°, lon −75,15° a −64,22° | dimensiones y coordenadas |
| Recorte | `ncks -d x,137,204 -d y,0,254` aplicado el 2025-09-09 a la salida original | atributo `history` (solo en el archivo histórico 1979–2014 se inspeccionó completo) |
| Contacto | katerina.goubanova@ceaza.cl | atributo `contact` |
| Creación de los archivos | histórico 2022-09-29 / 2022-10-22; SSP5-8.5 2023-01/02; SSP3-7.0 2024-05/06 | atributo `creation_date` |

**Inconsistencias de metadatos (no resueltas, preguntar a Goubanova):**
1. `driving_experiment_name = historical` en **todos** los archivos, incluidos los `ssp370` y `ssp585`. El escenario se identifica solo por el nombre de archivo y por la fecha del eje temporal.
2. `driving_experiment = "CNRM-ESM2-1, evaluation, r1i1p1"` en los 8 archivos, aunque el miembro del forzante es r1i1p1f2 y no hay corrida de evaluación.
3. El dominio recibido termina en 45,9°S: las 3 587 celdas dentro del polígono de Chile van de 17,7°S a 45,9°S (la grilla completa llega a 17,3°S); el análisis no cubre «Chile continental» completo. La tesis de referencia habla de 17–56°S; preguntar si el recorte es de Goubanova.
4. Los archivos SSP5-8.5 y SSP3-7.0 se generaron en fechas distintas de los históricos (hasta 2024); no hay documento que confirme que usan la misma versión de configuración. El atributo `arpege_minor_version` sí es igual (6.4.1).

**Sin documentar:** referencia bibliográfica de la configuración (Nabat et al. 2020 describiría ALADIN63, no 64: sin verificar); Goubanova et al. 2024 (IMSC, resumen p. 91) solo respalda la existencia del dominio norte-centro, el forzante y los escenarios, no precipitación ni versión; disponibilidad pública de los datos (no verificada).

## CR2MET v2.5 (`pr/`, 672 archivos)

| Dato | Valor | Evidencia |
|---|---|---|
| Cobertura | 1966-01 a 2021-12, un archivo por mes (56 años × 12 = 672, sin huecos) | nombres de archivo |
| Grilla | 0,05°, 800 × 220 (lat −56,98° a −17,02°; lon −76,97° a −66,03°) | dimensiones |
| Variables | `pr`, `pr_sd`, `cl_mask` | archivo |
| Autoría y fecha | `authors = Boisier et al.`; `creation_date = 29-Aug-2022 18:52:32` | atributos globales |
| Observaciones base | `obs_file = MERGE5_QC_UCC_pr_day_1979_2021.csv`; ensamble de 50 miembros, mediana (`ens_agg_metric = quantile`, `ens_qtl_value = 0.5`) | atributos globales |

**Sin documentar:** DOI/cita del producto y condiciones de uso (no verificadas); el valor de `pr` es la mediana del ensamble, de modo que no representa la incertidumbre de los días secos/húmedos.

## Lo que usa cada análisis

| Análisis | Datos |
|---|---|
| Calibración de R₀, mapas, PDFs, sesgo | CR2MET 1980–2014 + ALADIN histórico 1979–2014 (recortado a 1980–2014) |
| Risk ratios (principal) | ALADIN histórico 1980–2014 + `ssp585` 2040–2074 (archivos 2040–2064 y 2065–2100) |
| Risk ratios, sensibilidad S6 | ídem con `ssp370` |
| No se usa | ALADIN histórico 1950–1978; `ssp585`/`ssp370` 2015–2039; CR2MET 1966–1979 y 2015–2021 |
