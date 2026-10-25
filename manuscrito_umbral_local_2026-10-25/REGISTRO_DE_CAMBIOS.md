# Adaptación del manuscrito y del material suplementario

## Decisiones aplicadas

Se conservan el español, las secciones del artículo, las cuatro ventanas y las cuatro regiones administrativas de los DOCX adjuntos. Las copias fuente permanecen en `fuentes/`, con sus hashes y comentarios en `qa/`. Los dos enlaces facilitados por el usuario estaban vacíos; no se han usado como fuente científica ni se han sobrescrito. La publicación se realiza importando los DOCX nuevos como documentos nativos, con enlaces nuevos.

El criterio principal es CR2MET ≥1 mm y umbral de ALADIN por celda calibrado en 1980–2014, fijo en 2040–2074. Se conservan las etiquetas de los experimentos i global, ii fijo y iii local: iii pasa a principal. El suplemento mantiene S1–S7 y agrega S8 (referencia de 0,1 mm), S9 (Markov y censura) y S10 (comparación global/local). El título pasa a «a lo largo de un gradiente climático de Chile», conservando el objeto y el horizonte del original.

## Cambios científicos y trazabilidad

| Contenido | Cambio aplicado | Fuente numérica o metodológica |
|---|---|---|
| Calibración, §2.3 y Figura 3 | Campo de 3 587 celdas; mediana 6,489 mm; búsqueda ampliada en 82 celdas; error <0,008 pp | `temporal_umbral_por_celda_2026-10-24/ejecucion/resultados_tesis/calibracion_local.json`, `calibracion_local_por_celda.csv`; `umbral_local.py`, `calibrar_local.py` |
| Tabla 1 y Tabla 2 | Clasificación local principal; global y fijo como sensibilidades; referencia 0,1 mm separada | Configuraciones y mapas de ambos experimentos locales |
| Mapas y promedios, §3.1.2 | 3 549 celdas comunes entre productos en el criterio local; sesgos −6,7 y −35,7 días; porcentajes significativos actualizados | `ejecucion/paper_figuras/tablaS1_sensibilidad_umbral_chile.csv`, `_run_pregunta7_bootstrap.py` |
| Comparación local/global | Comparación sobre 3 526 celdas idénticas: MAE 14,890→6,673 y 71,473→48,825 días | `temporal_umbral_por_celda_2026-10-24/comparativa/11_resumen_espacial.csv` |
| Tabla 3, Figura 5 y §3.2.2 | Todos los conteos y estadísticos reemplazados por la ejecución local; p99 de Maipo 121 frente a 119 días; se elimina la afirmación de déficit en las cuatro ventanas | `ejecucion/resultados_tesis/tabla3_estadisticos_ventanas.csv`, `sesgo_por_ventana.csv`, `frecuencia_humeda_ventanas.csv` |
| Tabla 4, Figura 6 y conclusiones | Solo La Araucanía significativa a 20 días en SSP5-8.5 con 1 mm; Coquimbo incluye 1 | `ejecucion/outputs_pregunta5_aladin/results.json` |
| Tabla S1b | Mínimos 30/50/100/200 con sus máscaras propias | `ejecucion/resultados_tesis/sensibilidad_minimo_rachas.csv` |
| Tabla S2 y Figura S1 | Campo local actualizado; ya no se usa la versión limitada a 15 mm | `ejecucion/_pregunta9_calibracion_outputs/resumen_sensibilidad_umbral_4cuencas.csv` |
| Figura S2 | Comparación de los campos contra 1 y 0,1 mm | `temporal_umbral_por_celda_0p1mm_2026-10-25/comparativa/comparacion_mapas_umbrales.png` y ambos `calibracion_local.json` |
| Tabla S3 y Figura S3 | Umbral escalar 1 mm en ALADIN, separado de la referencia observacional 1 mm; 42 descartes en Los Lagos a D20 | `ejecucion/outputs_pregunta5_aladin_1mm/results.json` |
| Tabla S4 | Se distingue promedio de centros de un regrillado conservativo; máscaras propias explícitas | `ejecucion/_test_regrillado/mapas_lineal_vs_area.csv` |
| Tabla S5 y Figura S4 | SSP3-7.0 con campo local; límite inferior de Araucanía 1,0037 | `ejecucion/outputs_pregunta5_aladin_ssp370/results.json` |
| Tabla S6 y discusión | Frecuencia seca +1,6 a +3,9 pp; Δρ incluye cero en las cuatro regiones con 1 mm | `ejecucion/resultados_tesis/contraste_persistencia_regiones.csv`, `_contraste_persistencia_regiones.py` |
| Tabla S7 | Referencias 1 y 0,1 mm comparadas sobre las mismas 3 549 celdas; máscara propia de 0,1 mm = 3 583 | `temporal_umbral_por_celda_0p1mm_2026-10-25/comparativa/04_resumen_espacial.csv` |
| Tabla S8 y Figura S5 | Coquimbo y Araucanía significativos con 0,1 mm bajo SSP5-8.5; ninguno bajo SSP3-7.0; 1 descarte en Los Lagos | `comparativa/09_RR_D20.csv`, `comparacion_RR.png` del experimento 0,1 mm |
| Tabla S9 | Δρ significativo en Los Lagos con 0,1 mm: 0,02800 [0,00387; 0,05209] | `temporal_umbral_por_celda_0p1mm_2026-10-25/ejecucion/resultados_tesis/contraste_persistencia_regiones.csv` |
| Tabla S10 | Referencia Markov: 30/30 y 29/29 compatibilidades con N histórico ≥100; cobertura histórica explícita | Ambos `ejecucion/resultados_tesis/descomposicion_markov_rr_regiones.csv` |
| Tabla S11 | Censura actualizada, sin interpretar duraciones truncadas como duraciones totales conocidas | `ejecucion/_test_regrillado/rachas_censuradas_por_cuenca.csv` de 1 mm |
| Tabla S12 | Cuatro combinaciones global/local × 1/0,1 mm | `temporal_umbral_por_celda_0p1mm_2026-10-25/comparativa/18_comparacion_cuatro_versiones_RR.csv` |

## Interpretaciones corregidas

La calibración local controla la frecuencia anual, no la frecuencia de cada estación, la dependencia temporal ni los acumulados. Se aclara por qué no se calibra directamente contra la duración: aquí permanece como variable de evaluación fuera del objetivo de ajuste. Las diferencias históricas residuales no identifican por sí solas su causa física.

Los RR son cocientes de conteos en períodos de igual duración; no son cocientes de proporciones ni probabilidades individuales de daño. Se declaran réplicas sin denominador, intervalos condicionales, falta de corrección por multiplicidad y ausencia de recalibración dentro del bootstrap. D20 se presenta como referencia descriptiva. Los cambios futuros por ventana no reciben una significancia que no fue calculada.

La diferencia de significancia de Coquimbo entre referencias no demuestra una diferencia significativa entre los RR. El aumento de persistencia de Los Lagos con 0,1 mm se declara tanto en la discusión principal como en el suplemento. La compatibilidad Markov no se convierte en atribución causal, y su cobertura histórica baja en Los Lagos queda visible en la tabla.

La referencia de 1 mm se justifica por convención ETCCDI y comparabilidad con Latoja et al., no como un óptimo físico universal. La alternativa 0,1 mm cuenta con un antecedente explícito (Breinl et al.). Las lecturas y metadatos utilizados están en `revision_bibliografica_umbral_2026-10-25/`; no se afirma haber realizado una revisión bibliométrica exhaustiva.

## Bibliografía y notas de los originales

Se retiraron las notas literales `[VERIFICAR]` resolviendo las afirmaciones afectadas: no se atribuye sesgo de llovizna a Petrova, ni resultados de duración a trabajos sobre acumulaciones/intensidades; no se conservan Sun y Chen sin referencias. Se distinguen el máximo anual de Petrova y el pool de rachas completas. Se completaron los autores de Leung desde los metadatos ya verificados y se agregó Breinl al suplemento. Las referencias eliminadas corresponden a pasajes retirados; no se inventó evidencia para conservarlas. Los demás antecedentes bibliográficos del manuscrito se preservan; esta adaptación no constituye una nueva revisión exhaustiva de toda su bibliografía.

Se extrajeron los 16 comentarios del DOCX a `qa/comentarios_fuente.json`. Son antecedentes editoriales, no instrucciones nuevas. La versión final se entrega limpia; las notas y sus documentos originales se preservan. Las propuestas antiguas de mover las Figuras 2–3 no se ejecutan, respetando la organización solicitada. La motivación de calibración se concentra en §2.3 y las sensibilidades en §2.7.

## Figuras y fidelidad

Las figuras se incluyen en los DOCX y quedan además en `figuras/`, numeradas como en el texto. El índice JSON registra origen y SHA256. Tres RR se redibujan con `generar_rr_publicacion.py` desde los JSON existentes: se elimina el volcado de xarray en el título, se usa eje logarítmico sin recortar valores altos y se omiten bandas en duraciones con descartes, marcándolas con símbolos huecos. No se recalculan las estimaciones ni se sobrescriben las figuras de los experimentos.

La Figura 5 se redibuja con `generar_distribuciones_publicacion.py` desde las duraciones almacenadas, usando exactamente los bordes, centros y conteos originales. Se sustituye «cuenca» por «ventana» y se simplifican títulos. El control de las doce series verifica que sus integrales recuperan todos los eventos. Se protege cada fila de tabla contra división entre páginas y se ajustan sus anchos al área útil.

La importación a Google Docs conservó exactamente el contenido de todos los párrafos, comprobado por comparación normalizada sin párrafos faltantes ni añadidos. Se verificaron cuatro tablas y seis imágenes en el artículo, y trece tablas y cinco imágenes en el suplemento. Se agregó un hipervínculo desde el primer párrafo del artículo al nuevo suplemento. Una tipografía Gungsuh heredada de los DOCX originales producía texto monoespaciado en Google: se sustituyó únicamente esa familia por Times New Roman, tanto en el respaldo local como en el documento nativo. Los tamaños, cursivas y negritas se mantienen.

Se mantienen página carta, márgenes de una pulgada y familias tipográficas fuente. Se unifica el estilo de los párrafos añadidos al suplemento, se eliminan resaltados y bordes de título y se conservan las proporciones de las imágenes. Las referencias originales se guardan sin cambios. Los renders y verificaciones estructurales están en `qa/` y son material de control, no las entregas principales.

## Preguntas de control de coherencia

Introducción: ¿se distingue día seco de sequía hidrológica? Sí, §4.2. ¿Se confunden máximos anuales con todas las rachas? No, §§1 y 2.2. ¿Se generaliza a todo Chile? El dominio efectivo se declara en §2.1 y el título se ajusta.

Métodos: ¿se usa información futura en la calibración? No. ¿El bootstrap incluye incertidumbre del umbral? No, se declara. ¿Se ocultan réplicas sin denominador? No, tablas y símbolos las identifican.

Resultados: ¿se mezclan máscaras? Se indican muestras propias y comunes. ¿Se mantiene la antigua significancia de Coquimbo? No. ¿Se atribuye significancia a los cambios descriptivos por ventana? No.

Discusión: ¿la cadena Markov prueba el mecanismo? No. ¿Se oculta la sensibilidad de Los Lagos? No. ¿Se presenta 1 mm como físicamente correcto en cualquier clima? No.
