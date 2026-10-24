# Revisión de la raíz frente al paper y su suplemento

**Actualización del 24 de octubre:** se completó la reproducción desde NetCDF sin cachés. Las 68 salidas coinciden y pasaron 90 controles cruzados. Véase [REPRODUCCION_COMPLETA_PAPER_2026-10-24.md](REPRODUCCION_COMPLETA_PAPER_2026-10-24.md). Las limitaciones de ejecución descritas abajo corresponden a la revisión inicial; los ajustes metodológicos y editoriales siguen pendientes.

Fecha: 23 de octubre de 2026. Revisión de los documentos vivos de Drive, los 21 scripts `_*.py` de la raíz y sus entradas/salidas. No se modificaron scripts, resultados, figuras ni documentos de Drive. Este informe es el único archivo añadido al repositorio.

**Dictamen:** la raíz contiene la cadena de cálculo del paper original y los resultados principales comprobados coinciden. Sin embargo, todavía hay diferencias entre documento y figura, una aproximación numérica en precipitación y problemas de reproducción desde una carpeta vacía. No corresponde declarar toda la cadena certificada de extremo a extremo con esta revisión.

## 1. Fuentes y alcance

- [Paper: Introducción v1](https://docs.google.com/document/d/1QG5xHTIcqvRqTvF8k3DVCsJ9If_iT5YmWFmIj3CDdJE/edit).
- [Material suplementario v1](https://docs.google.com/document/d/1WbTTi7W3Z1LzmHBDiIljFgcH0cO4THieM43yMrFOeaU/edit).
- Texto, tablas e imágenes obtenidos con el MCP de Google Drive. Las diez imágenes incrustadas se inspeccionaron y compararon con las locales.
- Se distinguieron inserciones y eliminaciones sugeridas: el suplemento tiene una corrección pendiente en S2. No se interpretó su representación combinada como texto duplicado.
- Se leyeron los 21 scripts de la raíz, `CLAUDE.md`, `REGISTRO_RESULTADOS.md`, `PROCEDENCIA_DATOS.md` y la auditoría anterior. Esta última se usó como antecedente, no como comprobación actual.

El análisis revisado usa 1980–2014 / 2040–2074, CR2MET a 1 mm, ALADIN a 5,285 mm, cuatro ventanas y cuatro regiones. Los análisis de `paso_a_paso_v2/`, `chile_central_paso_a_paso/`, `propuesta_chile_central/` y `umbral_0p1mm_nacional/` son trabajos separados: sus resultados no se usaron para validar este paper.

## 2. Comprobaciones realizadas

| Comprobación de esta revisión | Resultado y alcance |
|---|---|
| Manifiesto de salidas | **49 de 49 hashes SHA-256 coinciden** con `resultados_tesis/sha256_salidas.txt`. Los 16 PNG de `paper_figuras/` están incluidos. Un hash acredita integridad, no validez científica. |
| Rachas de ALADIN desde NetCDF | Se releyeron histórico, SSP5-8.5 y SSP3-7.0. Se detectaron rachas con `scipy.ndimage.label`, usando máscaras regionales calculadas con `shapely.intersects_xy`. Los registros agregados año/mes/duración coinciden con los CSV regionales: análisis principal, sensibilidad de 1 mm y SSP3-7.0. |
| Calendario y valores de ALADIN | Cada período seleccionado tiene 12 784 días consecutivos, sin fechas duplicadas. Las 3 587 celdas seleccionadas no tienen NaN ni precipitación negativa en los tres períodos revisados. |
| PDFs de ALADIN | Las duraciones de las cuatro ventanas, históricas y futuras, recalculadas desde NetCDF coinciden exactamente con el pickle principal. |
| Mapas de ALADIN, criterio i | Media y p99 recalculados desde NetCDF coinciden con la caché dentro de la precisión de almacenamiento. |
| CR2MET | PDFs y mapas del criterio i se recalcularon desde la caché de interpolación lineal y coinciden. Esa caché tiene 12 784 × 3 587 valores, sin NaN. **No se reconstruyó la interpolación desde los archivos mensuales crudos.** |
| RR e intervalos | Se recalcularon **120 puntos**: 10 duraciones × 4 regiones × 3 análisis. Conteos, mediana bootstrap, IC95 y réplicas descartadas coinciden con los JSON a su precisión guardada. El recálculo usó conteos por año, sin concatenar duraciones, conservando los sorteos de semilla 42. |
| Tabla 3 | Las 12 filas del Doc coinciden con las duraciones del pickle y el CSV a la precisión publicada: n, media, mediana, p90, p99, máximo y fracción ≥ 20 días. |
| Tabla 4 | Las cuatro filas del Doc coinciden con los conteos y los intervalos recalculados. |
| Tablas suplementarias | S1 y S1b coinciden con sus CSV; las 16 filas de S2 coinciden con el pickle de sensibilidad; S3 y S5 con los JSON; S4 con las salidas de regrillado; S6 con el CSV de persistencia. Esto verifica correspondencia, no una nueva ejecución completa de todos sus métodos. |
| Mapas y bootstrap guardado | Máscaras y diferencias de las dos métricas coinciden entre la caché de mapas y los NPZ del bootstrap principal: **3 526 celdas comunes**. No se repitieron las 1 000 réplicas por celda. |
| Bins de PDFs | 23, 22, 21 y 21 bins efectivos en Loa, Maipo, Maule y Biobío; los histogramas incluyen todas las rachas del pool. |
| Diagnóstico Markov | El CSV contiene 31 casos con N_hist observado ≥ 100 y el RR de solo frecuencia cae dentro del IC95 observado en los 31. No se repitieron las simulaciones de Markov. |

Algunos valores centrales comprobados:

| Resultado | Valor vigente |
|---|---|
| R₀ registrado | 5,284599 mm por bisección; 5,285 mm usado |
| Fracción húmeda registrada | CR2MET 20,944772 %; ALADIN a 1 mm 30,684945 % |
| Sesgo espacial, celdas comunes | Media −8,23023 días; t99 −46,25986 días |
| Coquimbo, RR(20) | 25 367 → 28 458; 1,1219; IC95 [1,0075; 1,2501] según JSON |
| O'Higgins, RR(20) | 7 015 → 8 096; publicado 1,15 [0,95; 1,40] |
| La Araucanía, RR(20) | 2 397 → 4 118; publicado 1,72 [1,10; 2,88] |
| Los Lagos, RR(20) | 601 → 1 291; publicado 2,15 [0,90; 8,50] |

## 3. Correspondencia de scripts, figuras y tablas

Los nombres de archivo conservan numeración histórica: el número del nombre no siempre es el del Doc.

| Elemento del paper | Script de la raíz | Archivo o dependencia principal |
|---|---|---|
| Figura 1, dominio | `_generar_fig_dominio_ventanas_regiones.py` | `paper_figuras/fig02_dominio_cuencas_regiones.png` |
| Figura 2, esquema | `_generar_fig02_esquema_dryspell.py` | `paper_figuras/fig_esquema_dryspell.png`; ejemplo ilustrativo |
| Figura 3 y calibración | `_generar_fig03_calibracion_R0.py` | `fig03_calibracion_R0_ftau.png`, `resultados_tesis/calibracion_R0.json` |
| Mapas, bootstrap y S1 | `_run_pregunta7_bootstrap.py` | `_pregunta7_outputs/`, `outputs_pregunta7_bootstrap/`, seis NPZ por celda |
| Figura 4 | `_generar_fig05_mapas_persistencia.py` | `fig05_mapas_persistencia_criterio_i.png`; lee la caché anterior |
| Sesgo espacial por ventana | `_calcular_sesgo_ventanas.py` | `resultados_tesis/sesgo_por_ventana.csv` |
| Figura 5, PDFs | `_generar_pregunta9_pdf_loglog_cuencas.py` | `fig42_pdf_cuencas_loglog_panel.png`, cuatro PNG individuales y pickle |
| Tabla 3 y sesgos relativos | `_calcular_tabla3_ventanas.py` | `tabla3_estadisticos_ventanas.csv`, `sesgo_relativo_criterios_ventanas.csv`; necesita ambos pickles de duraciones |
| Tabla 4, RR y sensibilidades | `_run_pregunta5_aladin.py` | Tres carpetas `outputs_pregunta5_aladin*`; opciones `--umbral 1` y `--escenario ssp370` |
| Figura 6 vigente | `_generar_fig_rr_regiones.py` | `paper_figuras/fig_rr_regiones.png`; lee JSON principal |
| Tabla S1 | `_generar_tablaS1_sensibilidad_chile.py` | `paper_figuras/tablaS1_sensibilidad_umbral_chile.csv` |
| Figura S1 y Tabla S2 | `_generar_figS1_sensibilidad_umbral.py` | PNG, CSV y pickle en `_pregunta9_calibracion_outputs/` |
| Figura S2 | `_generar_figS2_mapa_R0_local.py` | PNG y `_pregunta7_outputs/R0_local_map.npz` |
| Figuras S3–S4, Tablas S3 y S5 | `_run_pregunta5_aladin.py` con las opciones anteriores | `figS_rr_curves_aladin_1mm.png`, `figS_rr_curves_aladin_ssp370.png` |
| Tabla S4, regrillado | `_test_regrillado_conservativo.py` | `_test_regrillado/`: caché, CSV y JSON |
| Tabla S6 | `_contraste_persistencia_regiones.py` | `resultados_tesis/contraste_persistencia_regiones.csv` |
| RR esperado solo por frecuencia | `_referencia_markov_rr_regiones.py` | `resultados_tesis/descomposicion_markov_rr_regiones.csv` |
| Frecuencia húmeda por ventana | `_calcular_frecuencia_humeda_ventanas.py` | `resultados_tesis/frecuencia_humeda_ventanas.csv` |
| Precipitación y fracción mar–nov | `_calcular_estacionalidad_marnov_regiones.py` | `resultados_tesis/estacionalidad_marnov_regiones.csv`; requiere ajuste descrito abajo |
| Rachas censuradas | `_cuantificar_censuradas.py` | `_test_regrillado/rachas_censuradas_por_cuenca.csv` |
| Tabla S1b | `_sensibilidad_minimo_rachas.py` | `resultados_tesis/sensibilidad_minimo_rachas.csv` |
| Referencia auxiliar para tesis | `_generar_figT01_referencia_sin_memoria.py` | `figT01_referencia_sin_memoria.png/.csv`; **no está en ninguno de los dos Docs** |

Las cuatro imágenes `pdf_*_i_global_loglog.png` son componentes de la Figura 5, no cuatro figuras adicionales del paper. `fig08_rr_curves_aladin.png` es una versión anterior de presentación del RR; el Doc usa `fig_rr_regiones.png`. La figura T01 es auxiliar y no debe contarse como resultado publicado en estos Docs.

## 4. Hallazgos pendientes

### A. MAYOR — La Figura 1 incrustada no es la versión local vigente

La imagen recuperada del Doc dice literalmente **«Dominio de estudio — Chile continental»**. El archivo local dice **«Chile entre 17,7°S y 45,9°S (grilla ALADIN CHP12)»**. El pie y §2.1 del Doc ya declaran el dominio correcto.

Esto se comprobó mirando la imagen incrustada, no deduciendo su versión por la relación de aspecto. Las otras nueve figuras corresponden visualmente a sus versiones locales; el reescalado de Drive impide presentar esta comparación como identidad de bytes.

**Acción:** reemplazar la imagen de la Figura 1 por el PNG local vigente y conservar sus proporciones.

### B. MAYOR para la precisión de las cifras auxiliares — Precipitación anual calculada con meses de igual duración

`_calcular_estacionalidad_marnov_regiones.py:24–31` multiplica la media diaria de cada mes por **30,4375** y después suma los doce meses. Esto aproxima los acumulados: febrero, junio y julio reciben el mismo peso temporal.

Se recalculó la media de los acumulados de los 35 años usando el calendario real, con ALADIN desde NetCDF y CR2MET desde la caché de interpolación:

| Región | CR2MET guardado → calendario real, mm/año | ALADIN guardado → calendario real, mm/año |
|---|---:|---:|
| Coquimbo | 246,916 → 248,107 | 1 089,527 → 1 093,379 |
| O'Higgins | 949,109 → 953,015 | 2 288,476 → 2 298,978 |
| La Araucanía | 2 048,088 → 2 054,526 | 3 634,746 → 3 647,022 |
| Los Lagos | 2 722,019 → 2 728,473 | 5 016,169 → 5 025,723 |

La aproximación subestima estos acumulados entre **0,19 % y 0,48 %**. Las fracciones mar–nov cambian como máximo unos **0,204 puntos porcentuales**: por ejemplo, Los Lagos ALADIN pasa de 81,387 % a 81,591 %.

El efecto no cambia los RR, las PDFs ni los mapas: esos scripts no usan estos acumulados. Sí cambia las cifras auxiliares citadas en §2.6 y §4.3. El rango redondeado del sesgo de precipitación sigue siendo aproximadamente 1,8–4,4 veces.

**Acción:** sumar precipitación diaria por año y estación, promediar los años y regenerar este CSV y sus citas. Alternativamente, identificar expresamente los valores actuales como aproximaciones; para resultados precisos se prefiere corregir el cálculo.

### C. MAYOR — El orden publicado no funciona literalmente desde cero

`REGISTRO_RESULTADOS.md:45` coloca `_calcular_tabla3_ventanas.py` en el paso 8, aunque indica que necesita el paso 10. Ese script abre el pickle de sensibilidad que genera `_generar_figS1_sensibilidad_umbral.py`.

En una carpeta vacía faltará la entrada y el proceso fallará. Con resultados anteriores presentes, el error puede quedar oculto y se puede usar una sensibilidad vieja.

**Acción:** ejecutar S1 antes de Tabla 3 y expresar el orden como dependencias válidas. Esto es un defecto del procedimiento documentado; no una discrepancia en la Tabla 3 actual.

### D. MAYOR — Las cachés no comprueban su vigencia

Los lectores de `pdf_durations_i_global.pkl`, `spell_durations_sensibilidad_4cuencas.pkl`, `R0_local_map.npz` y `cr2met_lineal_vs_area.npz` aceptan la caché si existe. No verifican parámetros, versión del código, datos ni geometrías.

Los resultados actuales comprobados coinciden; el riesgo aparece cuando se cambia una regla y se vuelve a ejecutar. R₀ también está repetido como constante en varios scripts: recalibrarlo no actualiza automáticamente los consumidores.

**Acción:** introducir una regeneración explícita y trazabilidad de entradas/configuración, o ejecutar la reproducción completa en una carpeta vacía. No borrar las cachés de trabajo como parte de esta revisión.

### E. MENOR — Pie de la Figura 6 incompleto respecto del PNG vigente

La figura actual distingue círculos huecos y suprime la banda donde se descartó alguna réplica con denominador histórico cero (`_generar_fig_rr_regiones.py:49–69`). El pie del Doc describe la banda, pero no explica esos símbolos ni esa omisión.

**Acción:** añadir que los círculos huecos señalan RR con réplicas descartadas y que allí no se representa el IC. Los intervalos siguen existiendo en el JSON como intervalos condicionados a denominador positivo.

Las sensibilidades S3 y S4 mantienen el diseño antiguo con eje vertical limitado a 4: la parte alta de varias curvas y bandas queda recortada. Sus tablas permiten recuperar las cifras, pero las figuras no muestran su magnitud completa.

### F. MENOR — Sensibilidad al mínimo de rachas: aclarar qué se reutiliza

`_sensibilidad_minimo_rachas.py` cambia las celdas incluidas al exigir 50/100/200 rachas. Reutiliza la significancia del bootstrap principal, que mantiene **30 rachas por réplica**. No vuelve a calcular intervalos exigiendo 50/100/200 en cada réplica.

Esto es coherente con una sensibilidad de selección espacial, y sus cifras coinciden con S1b. Debe describirse así para evitar que se interprete como un cambio completo del criterio de validez bootstrap.

### G. MENOR — Regrillado: alcance más limitado que un conservativo estricto

`_test_regrillado_conservativo.py:51–68` usa pesos 1/0 según si el centro de una celda CR2MET cae dentro de ALADIN, y promedia los centros seleccionados. No calcula áreas de intersección de celdas. El suplemento sí explica el procedimiento, pero «promedio por área» y el nombre del script pueden sugerir una conservación de masa que no se implementa.

Además, el criterio i conserva R₀ = 5,285 en ambas versiones (`:170–183`), aunque también registra la recalibración a aproximadamente 5,280 para el promedio por centros. Las máscaras válidas finales varían: por ejemplo, 3 526 frente a 3 529 celdas en el criterio i.

**Acción:** describir la prueba como promedio de celdas seleccionadas por su centro con R₀ fijo. Si se quiere aislar estrictamente el efecto del regrillado, comparar también sobre una máscara final común. La prueba actual no es una validación de un regrillado conservativo por áreas de solape.

### H. MENOR — Documentación anterior y sugerencias pendientes

- La auditoría anterior conserva pendientes que ya están corregidos: rango 1,8–4,4, excepciones al sesgo negativo, 4,1 pp en Maipo, censura futura y numeración de figuras en el registro.
- En S2 del suplemento permanece una sugerencia pendiente para corregir la afirmación de que la cola de Biobío termina antes que la observada. El texto con la sugerencia aceptada es coherente con los datos: su p99 es menor, aunque el máximo de ALADIN es mayor. No es un error de duplicación del documento.
- El pie de la Figura 5 puede ser más preciso diciendo «subestima el p99»; el máximo histórico de Biobío es 153 días en ALADIN y 109 en CR2MET.
- `requirements.txt` no fija versiones; existen tres entornos virtuales locales. La revisión usó el Python 3.11 de pyenv indicado en el registro. Falta una definición reproducible del entorno y de la versión de geometrías Natural Earth.
- En RR, `np.concatenate([])` puede fallar antes del control de muestra vacía si un remuestreo no contiene ningún año con rachas. Es una fragilidad para reutilizar el código con otras selecciones, sin efecto detectado en las salidas vigentes.

## 5. Qué corresponde a la raíz del paper

| Archivos/carpetas | Papel |
|---|---|
| Los 21 scripts de la raíz | Cálculos del paper, gráficos y una referencia auxiliar de tesis |
| `pr/`, `pr1/` | Datos; conservar |
| `paper_figuras/` | Diez figuras usadas en los Docs, cuatro paneles individuales, una versión previa de RR y T01 auxiliar; también contiene CSV |
| `resultados_tesis/` | Estadísticos y diagnóstico del paper, pese al nombre |
| `_pregunta7_outputs/`, `_pregunta9_outputs/`, `_pregunta9_calibracion_outputs/` | Cachés y salidas de la cadena |
| Las tres carpetas `outputs_pregunta5_aladin*` | RR principal y sensibilidades |
| `outputs_pregunta7_bootstrap/`, `_test_regrillado/` | Bootstrap y sensibilidad al regrillado |
| Markdown de auditoría, registro y procedencia | Documentación de trabajo; parte está desactualizada |
| DOCX y PDF de la raíz | Copias o bibliografía; no sustituyen los Docs vivos ni generan sus resultados |
| Otras cuatro carpetas de análisis | Proyectos distintos; no mezclar sus cifras con el paper original |
| `.venv*`, `__pycache__/`, `.DS_Store` | Entorno o archivos auxiliares; no son resultados científicos |

No se recomienda eliminar carpetas a partir de esta revisión: varias son dependencias reales aunque sus nombres parezcan antiguos.

## 6. Límite del dictamen y siguiente trabajo

No se encontraron diferencias en los conteos, RR, intervalos regionales ni duraciones principales comprobados. Sí se encontró una diferencia de versión en la Figura 1 y una aproximación corregible en las cifras de precipitación.

No se repitieron la interpolación completa de CR2MET desde sus archivos mensuales, el bootstrap de mapas para los tres criterios, las simulaciones Markov ni toda la cadena gráfica desde una carpeta vacía. Tampoco se verificó el contenido de las referencias bibliográficas ni se resolvieron las inconsistencias de procedencia de ALADIN documentadas en `PROCEDENCIA_DATOS.md`.

El siguiente paso útil sería corregir primero B–D, actualizar las figuras y sus pies y después ejecutar la cadena completa en un directorio independiente, comparando todas las salidas. Esa ejecución final permitiría cerrar la afirmación de reproducción completa desde datos crudos.

Las comprobaciones de esta sesión, scripts temporales y comparaciones de imágenes se guardaron fuera del repositorio en `/tmp/tesis_revision_actual_20261023/`. Son evidencia temporal; las conclusiones y el alcance están contenidos en este informe.
