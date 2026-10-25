# Guía explicativa del paper para comprender y preparar la tesis

Actualizada el 24 de octubre de 2026 a partir de los dos Docs vivos de Drive y las salidas vigentes, incluida la corrección de precipitación de calendario.

Título de la tesis previsto: **Duración de las rachas secas en Chile a lo largo de un gradiente climático: evaluación de un modelo regional y cambio proyectado a mediados de siglo bajo SSP5-8.5**.

Esta es una explicación del orden metodológico del estudio y de sus evidencias. No es un capítulo de tesis ni una reconstrucción fechada de todas las exploraciones anteriores.

## La idea que une el trabajo

El estudio pregunta cuánto duran los períodos consecutivos con precipitación bajo un umbral, cómo cambia su distribución desde el norte árido hacia el sur húmedo y qué cambio proyecta una simulación regional hacia mediados de siglo.

Tiene dos preguntas centrales:

1. **Evaluación histórica:** ¿en qué medida ALADIN representa las duraciones de las rachas de CR2MET, después de controlar la frecuencia media de días húmedos?
2. **Cambio futuro:** ¿cómo cambia la distribución de duraciones y el número de rachas largas entre ALADIN histórico y ALADIN futuro bajo SSP5-8.5?

La comparación histórica describe las capacidades y limitaciones del modelo. La comparación futura describe su proyección bajo un escenario. La evaluación histórica informa la interpretación de la proyección; no garantiza por sí sola su exactitud futura.

## 1. Primero: definir qué queremos medir

La variable central es la **duración de cada racha seca**, medida en días consecutivos. Estudiamos todas las rachas completas del período y prestamos especial atención a la **cola**, es decir, las duraciones poco frecuentes y prolongadas.

Un acumulado anual informa cuánto llovió. Nuestra variable informa cuánto tiempo transcurrió sin alcanzar un umbral diario de precipitación. El ejemplo conceptual de dos series con igual acumulado y distintas fechas de lluvia sirve para explicar por qué interesa mirar la secuencia diaria; no es un resultado calculado para Chile.

La media resume una parte de la distribución. El percentil 99 (p99 o t99) caracteriza su extremo superior: es un valor de duración asociado al 99 % acumulado de las rachas. No es el máximo y no significa «99 días». El máximo es el récord muestral y puede comportarse de manera distinta.

**Lo que no estamos midiendo directamente:** sequía agrícola, humedad del suelo, déficit de caudal, daños en cultivos o desempeño de embalses. La duración de rachas de precipitación es una dimensión meteorológica relevante para esas preguntas, pero sus impactos no se calcularon aquí.

## 2. Segundo: elegir los datos, períodos y dominio

Se utilizan:

- **CR2MET v2.5:** referencia observacional grillada, con precipitación diaria en una grilla de 0,05°. Es un producto construido a partir de información observacional y modelación, no una medición directa sin incertidumbre en cada celda.
- **CNRM-ALADIN64 CHP12:** modelo regional de aproximadamente 12 km, con un histórico y proyecciones de escenario.

Se seleccionan dos períodos de **35 años**: 1980–2014 para comparar CR2MET y ALADIN, y 2040–2074 para comparar ALADIN futuro con su propio histórico. SSP5-8.5 es el escenario principal; SSP3-7.0 se usa como sensibilidad de los RR.

El dominio efectivo abarca **17,7°S–45,9°S** y contiene **3 587 celdas** cuyo centro pertenece al polígono de Chile. Esa es la extensión de nuestra evidencia; no incluye todo Chile continental.

Las comparaciones no cumplen la misma función:

| Comparación | Qué responde |
|---|---|
| CR2MET histórico frente a ALADIN histórico | Evaluación del modelo |
| ALADIN histórico frente a ALADIN SSP5-8.5 | Cambio proyectado por ese modelo |
| ALADIN histórico frente a ALADIN SSP3-7.0 | Sensibilidad de los RR al escenario |

**Evidencia:** `PROCEDENCIA_DATOS.md`, `resultados_tesis/calibracion_R0.json` e inventario de NetCDF de la reproducción.

## 3. Tercero: hacer comparables las series

Antes de identificar rachas:

1. Se seleccionan las fechas de cada período.
2. Se convierte ALADIN de kg m⁻² s⁻¹ a mm/día multiplicando por 86 400.
3. Se interpola CR2MET linealmente a los centros de las celdas de ALADIN.
4. Se aplica la máscara geográfica correspondiente.

El orden es importante: **primero se interpola la precipitación y luego se clasifica cada día**. Interpolar precipitación e interpolar una clasificación seco/húmedo no son operaciones equivalentes.

La comparación climática no exige que ALADIN reproduzca el evento observado en la misma fecha. Evaluamos estadísticas de 35 años. El remuestreo histórico se hace de forma independiente para ambos productos.

## 4. Cuarto: establecer qué es un día seco y una racha

Con un umbral R, un día es húmedo cuando pr ≥ R y seco cuando pr < R. En CR2MET se usa **R = 1 mm/día**.

Una racha es una sucesión de días secos. Puede cruzar el 31 de diciembre: se conserva entera y se asigna al año en que empezó.

Se excluyen las rachas que tocan el primer o último día del período porque no sabemos su duración completa fuera de esa ventana. Son las **rachas censuradas**. Excluirlas evita presentar fragmentos como duraciones completas, pero también elimina algunos eventos muy largos; esa pérdida se cuantificó.

Ejemplo ilustrativo: con R = 1 mm, la secuencia diaria 2; 0; 0,4; 0; 3 mm contiene una racha seca de tres días, delimitada por dos días húmedos.

**Evidencia:** Figura 2 y `_test_regrillado/rachas_censuradas_por_cuenca.csv`.

## 5. Quinto: resolver la diferencia de frecuencia húmeda

Al usar 1 mm en ambos productos, las frecuencias medias históricas de días húmedos son **20,94 % en CR2MET** y **30,68 % en ALADIN**.

Esto importa porque cada día que supera el umbral interrumpe una racha seca. El exceso de días clasificados como húmedos puede fragmentar las rachas del modelo.

Para controlar esa diferencia se busca el umbral de ALADIN que iguala la frecuencia media de CR2MET. Se obtiene por bisección **R₀ = 5,284599 mm**, aplicado como **5,285 mm**.

La calibración no ajusta las duraciones para que coincidan. Ajusta una condición previa —la frecuencia media húmeda— y después se evalúan las duraciones. Tampoco modifica las cantidades diarias de precipitación: cambia su clasificación.

Dos límites son esenciales:

- **R₀ es estadístico:** 5,285 mm en ALADIN no se interpreta como ausencia física de agua para un cultivo.
- **La igualdad es del dominio:** no se impone en cada ventana. En Maipo queda un exceso de días húmedos de aproximadamente 4,15 puntos porcentuales y en Maule de 3,30.

De hecho, los acumulados históricos corregidos de ALADIN siguen siendo aproximadamente 1,8–4,4 veces los de CR2MET en las cuatro regiones. Igualar frecuencia no equivale a corregir toda la precipitación.

**Evidencia:** Figura 3, `calibracion_R0.json`, `frecuencia_humeda_ventanas.csv` y `estacionalidad_marnov_regiones.csv`.

## 6. Sexto: construir las rachas y usar dos escalas espaciales

Se extraen las duraciones completas de cada celda y período, con 1 mm en CR2MET y R₀ en el análisis principal de ALADIN.

Luego se usan dos tipos de agrupación:

| Unidad | Para qué se utiliza |
|---|---|
| Ventanas de Loa, Maipo, Maule y Biobío | Describir las distribuciones de duración mediante PDFs y Tabla 3 |
| Regiones de Coquimbo, O'Higgins, La Araucanía y Los Lagos | Cuantificar cambios en el conteo de rachas largas mediante RR |

Las ventanas son cajas de latitud–longitud, **no cuencas hidrográficas oficiales**. Ventanas y regiones no son intercambiables.

En las PDFs se reúnen todas las rachas de las celdas de una ventana en un pool. Su altura se calcula como conteo por ancho de intervalo: el área del histograma conserva el número de rachas. Aunque el paper las denomina PDFs, no son curvas normalizadas a área uno; muestran frecuencia de eventos y forma distributiva a la vez.

Miles de rachas de celdas vecinas no equivalen a miles de observaciones independientes.

## 7. Séptimo: establecer el gradiente observado y evaluar el histórico

Primero se describe la referencia CR2MET:

| Ventana | Media de duración, días | p99, días |
|---|---:|---:|
| Loa | 65,6 | 456 |
| Maipo | 11,0 | 119 |
| Maule | 9,5 | 83 |
| Biobío | 6,2 | 44 |

El gradiente norte–sur es claro. Después se pregunta cuánto de ese patrón representa ALADIN y cuánto difieren sus duraciones.

Los mapas calculan media y p99 por celda y sus diferencias **Δ = ALADIN − CR2MET**. Se comparan las mismas celdas con al menos 30 rachas en ambos productos: **3 526 celdas comunes**.

En ese conjunto, la diferencia media espacial es **−8,2 días para la duración media** y **−46,3 días para t99**. ALADIN reproduce cualitativamente el gradiente, pero subestima ambas métricas en buena parte del dominio. Existen sectores del norte con diferencias positivas: no es una subestimación en cada celda.

Los intervalos se obtienen con bootstrap de años: se sortean 35 años con reemplazo por producto, de forma independiente, y se repite 1 000 veces. El IC95 de Δ queda bajo cero en **66,3 % de las celdas para la media** y **43,1 % para t99**.

Estos porcentajes son de celdas, no de superficie ponderada ni una prueba de significancia del campo espacial completo.

En las cuatro ventanas, el análisis principal subestima la media y el p99 del pool. En términos relativos, el déficit de p99 es especialmente importante en Maipo y Maule. No se debe confundir p99 con máximo: en Biobío, el máximo histórico de ALADIN supera al de CR2MET.

**Evidencia:** Figuras 4–5, Tabla 3, `bootstrap_summary.csv`, `tabla3_estadisticos_ventanas.csv` y `sesgo_relativo_criterios_ventanas.csv`.

## 8. Octavo: comparar el futuro con el histórico del mismo modelo

Se conserva R₀ y se compara ALADIN SSP5-8.5 con ALADIN histórico. No existe una observación CR2MET del futuro con la cual verificar esa proyección.

En las cuatro ventanas disminuye el número total de rachas completas. Al mismo tiempo, aumentan sus medias y p99:

| Ventana | Media histórica → futura, días | p99 histórico → futuro, días |
|---|---:|---:|
| Loa | 54,6 → 56,6 | 360 → 385 |
| Maipo | 8,6 → 10,0 | 87 → 112 |
| Maule | 7,3 → 8,6 | 57 → 76 |
| Biobío | 5,3 → 6,0 | 34 → 39 |

Es compatible que haya menos rachas y sean más largas: una interrupción húmeda menos puede unir segmentos secos antes separados. Este ejemplo explica una posibilidad; el diagnóstico específico de frecuencia y persistencia se presenta en el paso 11.

Estos cambios de las ventanas son **resultados descriptivos**. No se calcularon aquí IC95 para cada cambio de media o p99 de la Tabla 3. La evaluación estadística regional del cambio de rachas largas se hace mediante RR.

**Evidencia:** Figura 5 y Tabla 3.

## 9. Noveno: cuantificar el cambio de rachas largas con RR

Se define:

**RR(D) = número futuro de rachas con duración ≥ D / número histórico de esas rachas.**

RR mayor que uno significa más rachas largas; menor que uno, menos. Es un cociente de conteos. No es el cociente entre las fracciones de rachas largas dentro del total de cada período.

Se utilizan las cuatro regiones y solo rachas cuyo inicio ocurre entre marzo y noviembre. Se conserva su duración completa aunque continúen en verano. Las PDFs anteriores usan todo el año.

La selección estacional se apoya en que marzo–noviembre concentra 85,1–97,3 % de la precipitación anual regional en CR2MET y 81,6–95,9 % en ALADIN. Es una decisión de diseño; no demuestra que toda racha iniciada en verano carezca de importancia.

Con D = 20 días:

| Región | Conteo histórico → futuro | RR e IC95 |
|---|---:|---|
| Coquimbo | 25 367 → 28 458 | 1,12 [1,01; 1,25] |
| O'Higgins | 7 015 → 8 096 | 1,15 [0,95; 1,40] |
| La Araucanía | 2 397 → 4 118 | 1,72 [1,10; 2,88] |
| Los Lagos | 601 → 1 291 | 2,15 [0,90; 8,50] |

El IC95 está completamente por encima de uno en Coquimbo y La Araucanía. En O'Higgins y Los Lagos incluye uno: sus aumentos puntuales no tienen el mismo respaldo estadístico.

El bootstrap remuestrea años completos de manera independiente en cada período, conservando juntas las celdas de la región. Así mantiene la covariación espacial de los conteos dentro de cada año. Su incertidumbre no representa un conjunto de modelos ni separa la respuesta forzada de la variabilidad interna.

La rareza de eventos puede volver inestable el cociente. Si el conteo histórico es cero, el RR no está definido. Las réplicas bootstrap con denominador cero se descartan y se cuentan; sus intervalos son condicionados a denominador positivo.

D = 20 es una referencia del análisis, no un umbral de daño agrícola validado.

**Evidencia:** Figura 6, Tabla 4 y `outputs_pregunta5_aladin/results.json`.

## 10. Décimo: probar la sensibilidad de las conclusiones

Se contrastan:

- **Criterio ii:** 1 mm en ambos productos.
- **Criterio i:** R₀ global en ALADIN, análisis principal.
- **Criterio iii:** umbral local por celda en ALADIN.
- RR con 1 mm y con SSP3-7.0.
- Interpolación lineal frente al promedio de celdas CR2MET seleccionadas por su centro.
- Selección de celdas con al menos 30, 50, 100 o 200 rachas.

La evaluación histórica muestra un sesgo negativo medio con los tres criterios, aunque su magnitud y patrón varían. En Maipo el sesgo de la media casi desaparece con el criterio local (−3,5 %): parte del sesgo global dependía de la frecuencia húmeda residual. No se puede atribuir todo ese sesgo directamente a agrupamiento temporal.

La significancia futura depende de las decisiones: con SSP3-7.0, los IC95 de Coquimbo y La Araucanía a D = 20 incluyen uno. Con 1 mm mantienen aumentos significativos, mientras Los Lagos presenta un cociente muy impreciso. Tampoco corresponde afirmar que todo resultado desaparece con SSP3-7.0: Los Lagos tiene un aumento significativo a D = 20 en esa sensibilidad, con IC amplio.

La sensibilidad de regrillado se basa en promedio por centros, no en áreas de solape de un conservativo estricto. En la sensibilidad al mínimo de rachas, se cambia la selección de celdas y se reutiliza la significancia del bootstrap principal.

**Evidencia:** secciones S1–S6 del suplemento y sus CSV/JSON registrados.

## 11. Undécimo: interpretar frecuencia y persistencia sin atribuir mecanismos

Una cosa es cuántos días secos existen; otra, cuánto depende el estado de hoy del estado de ayer.

Se mide la persistencia de primer orden mediante:

**ρ = P(seco hoy | seco ayer) − P(seco hoy | húmedo ayer).**

En las cuatro regiones, la frecuencia seca de marzo–noviembre aumenta entre **2,1 y 3,9 puntos porcentuales**. Sin embargo, el IC95 del cambio de ρ incluye cero en las cuatro.

Además, se construye una referencia Markov que cambia la frecuencia seca mensual futura, manteniendo la persistencia de primer orden histórica. El RR esperado de esa referencia cae dentro del IC95 del RR observado en **31 de 31 casos con al menos 100 rachas históricas observadas**.

Esto permite decir: **el aumento de rachas largas es de magnitud compatible con la menor frecuencia de días húmedos, sin un cambio detectable en la persistencia de primer orden**.

No permite decir que la frecuencia sea la causa única o que no cambie ninguna forma de organización temporal. Los IC son amplios, la potencia es limitada y Markov representa imperfectamente el histórico: a D = 20 reproduce aproximadamente el 70 % de las rachas largas históricas de Los Lagos y el 89 % de La Araucanía.

Tampoco se extiende automáticamente este diagnóstico a Loa: se hizo para las cuatro regiones de RR y, en la ventana Loa, la frecuencia húmeda anual del modelo aumenta ligeramente.

**Evidencia:** `contraste_persistencia_regiones.csv` y `descomposicion_markov_rr_regiones.csv`.

## 12. Finalmente: comprobar la trazabilidad y la reproducción

La cadena original se ejecutó desde NetCDF en un directorio sin cachés: 21 scripts, 23 comandos, 68 salidas reproducidas y 90 controles cruzados aprobados. Una imagen auxiliar solo difirió en su etiqueta de software, con píxeles idénticos.

Después se corrigió el cálculo de precipitación anual para usar el calendario real. Se regeneró únicamente su CSV y una segunda ejecución aislada produjo el mismo archivo. Las otras 67 salidas se conservaron.

Esta comprobación cierra la pregunta «¿los resultados proceden de los datos y scripts identificados?». No sustituye la evaluación de supuestos, procedencia externa, literatura o generalización climática.

## Qué es sostenible y qué no

| Afirmación sostenible | Afirmación que excede la evidencia |
|---|---|
| Hay un fuerte gradiente de duración en CR2MET dentro del dominio | Se describió todo Chile continental |
| ALADIN representa cualitativamente el gradiente y subestima media y p99 en buena parte del dominio | El modelo subestima toda duración en cada celda o reproduce perfectamente las rachas |
| La calibración global reduce las diferencias de frecuencia húmeda y parte del sesgo de duración | R₀ corrige toda la precipitación o iguala la frecuencia en cada celda |
| En las cuatro ventanas de ALADIN hay menos rachas y mayores medias y p99 futuros | Todos esos cambios descriptivos se probaron significativos |
| Aumentan significativamente las rachas ≥ 20 días en Coquimbo y La Araucanía en el análisis principal | Hay un aumento significativo en las cuatro regiones o robusto a todas las sensibilidades |
| El cambio es compatible con menor frecuencia húmeda y no se detecta cambio en ρ | Se demostró una reorganización temporal, una causa única o ausencia de cualquier cambio de memoria |
| Es una proyección de una simulación bajo un escenario | Es una predicción exacta del futuro o una atribución separada de la variabilidad interna |
| Las rachas prolongadas son una dimensión relevante para estudiar riesgo hídrico | Se cuantificaron pérdidas agrícolas, caudales futuros o impactos en embalses |
| La cadena es reproducible con entradas y entorno identificados | Toda elección metodológica, referencia y metadato quedó validado por la reproducción |

## Qué tenemos para la tesis

Tenemos las preguntas, datos identificados, un método explícito, resultados reproducibles, análisis de sensibilidad y límites concretos. El título propuesto recoge los dos componentes centrales: evaluación histórica y cambio proyectado.

La tesis deberá explicar por qué cada decisión era necesaria, qué encontró y qué limitó la interpretación. La pregunta de interpretación de frecuencia/persistencia ayuda a la segunda pregunta central; no necesita presentarse como una tercera investigación independiente.

Quedan por cerrar las referencias marcadas para verificar y aspectos de procedencia de ALADIN: por ejemplo, atributos de escenario inconsistentes y documentación de la configuración recibida. Son pendientes documentales distintos de la consistencia de los cálculos.

Una formulación defendible del aporte es:

> El estudio caracteriza la distribución de duraciones de rachas secas en un gradiente climático de Chile, evalúa su representación histórica por ALADIN después de controlar la frecuencia media de días húmedos y cuantifica su cambio proyectado bajo SSP5-8.5. El modelo subestima parte de la persistencia histórica y proyecta, en las ventanas analizadas, menos rachas y duraciones mayores. El aumento regional de rachas largas tiene respaldo estadístico en casos específicos del análisis principal y es compatible con una menor frecuencia húmeda, con límites de sensibilidad y de generalización propios de una única simulación.

## Fuentes principales para volver a los datos

- [Paper vivo](https://docs.google.com/document/d/1QG5xHTIcqvRqTvF8k3DVCsJ9If_iT5YmWFmIj3CDdJE/edit) y [suplemento](https://docs.google.com/document/d/1WbTTi7W3Z1LzmHBDiIljFgcH0cO4THieM43yMrFOeaU/edit).
- [Registro de resultados](/Users/pablogomez/Documents/GitHub/TESIS/REGISTRO_RESULTADOS.md).
- [Tabla 3: duraciones](/Users/pablogomez/Documents/GitHub/TESIS/resultados_tesis/tabla3_estadisticos_ventanas.csv).
- [RR principales](/Users/pablogomez/Documents/GitHub/TESIS/outputs_pregunta5_aladin/results.json).
- [Bootstrap histórico](/Users/pablogomez/Documents/GitHub/TESIS/outputs_pregunta7_bootstrap/bootstrap_summary.csv).
- [Contraste de persistencia](/Users/pablogomez/Documents/GitHub/TESIS/resultados_tesis/contraste_persistencia_regiones.csv).
- [Referencia Markov](/Users/pablogomez/Documents/GitHub/TESIS/resultados_tesis/descomposicion_markov_rr_regiones.csv).
- [Precipitación corregida](/Users/pablogomez/Documents/GitHub/TESIS/resultados_tesis/estacionalidad_marnov_regiones.csv).
- [Reproducción completa](/Users/pablogomez/Documents/GitHub/TESIS/REPRODUCCION_COMPLETA_PAPER_2026-10-24.md).
- [Corrección incremental de precipitación](/Users/pablogomez/Documents/TESIS_reproducciones/correccion_precipitacion_2026-10-24_003845/verificacion.json).
