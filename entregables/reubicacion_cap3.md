# Reubicación del Capítulo 3 (solo metodología) y bloques para Cap. 4, Cap. 5, Discusión y Anexos

Texto exacto para pegar en el Doc. Cada bloque dice **dónde** va y **qué reemplaza**.
Numeración supuesta (hay que numerar estos encabezados en el Doc): Cap. 4 = 4.1 Calibración del umbral y muestra efectiva · 4.2 Gradiente observado en CR2MET · 4.3 Sesgo de ALADIN frente a CR2MET · 4.4 Sesgo en la cantidad de lluvia. Cap. 5 = 5.1 Frecuencia de días húmedos · 5.2 Risk ratios · 5.3 Referencia de Markov · 5.4 Sensibilidad.

Lo marcado `[VERIFICAR]` o `[PENDIENTE]` no se puede dar por hecho. Las cifras de los bloques B, C y E vienen de los archivos citados, que revisé contra los CSV el 19-oct-2026; lo que no pude revisar está dicho en la sección F.

---

## A. CAPÍTULO 3 COMPLETO (reemplaza todo desde «Capítulo 3: Datos y metodología» hasta antes de «Capítulo 4»)

# Capítulo 3: Metodología

Este capítulo describe los datos y los métodos con que se responde cada objetivo específico. Los datos y la definición de rachas sirven a todos los objetivos. La calibración del umbral, los estadísticos por celda y su significancia sirven al objetivo 2 (evaluar a ALADIN). Las curvas de duración por macrozona sirven a los objetivos 1 y 2. Los risk ratios sirven al objetivo 3 (cambio proyectado), la referencia sin memoria y la prueba de control al objetivo 4 (qué parte del cambio se explica por la frecuencia de días húmedos), y las sensibilidades al objetivo 5.

## 3.1 Arquitectura general del estudio

El análisis sigue una secuencia fija, y el orden importa porque cada paso usa la salida del anterior. Primero se llevan CR2MET y ALADIN a las mismas celdas (regrillado, Sección 3.2.3). Segundo, se define el día húmedo con un umbral (Sección 3.4). Tercero, se extraen las rachas secas de todas las celdas con una sola regla (Sección 3.3). De ese conjunto de rachas salen todos los resultados: los estadísticos por celda y su significancia (Sección 3.5), las curvas de duración (Sección 3.6), los risk ratios (Sección 3.7) y la separación entre frecuencia y persistencia (Sección 3.8).

Los estadísticos por celda y las curvas de duración usan las rachas que empiezan en cualquier mes. Los risk ratios usan solo las que empiezan entre marzo y noviembre (Sección 3.7).

## 3.2 Datos

### 3.2.1 Producto de referencia: CR2MET

Como referencia usamos CR2MET v2.5 (Boisier, 2023), un producto de precipitación diaria en una grilla regular de 0,05° (unos 5 km) que cubre 1960-2021. Es un producto estadístico calibrado con estaciones, que combina reanálisis ERA5, topografía y datos satelitales [VERIFICAR: esta descripción contra Boisier et al. (2023)]. No es una observación directa, y por eso lo llamamos producto de referencia. Usamos el período 1980-2014. En el norte hay pocas estaciones, de modo que allí CR2MET depende más del modelo estadístico que de mediciones [VERIFICAR: respaldo de esta afirmación en la documentación de CR2MET].

### 3.2.2 Modelo: CNRM-ALADIN64

El modelo es CNRM-ALADIN64 (versión 6.4.1, 91 niveles verticales) sobre el dominio CHP12 (unos 12 km), forzado por el modelo global CNRM-ESM2-1 (miembro r1i1p1f2). Usamos la precipitación diaria, entregada en kg m-2 s-1, que convertimos a mm por día multiplicando por 86 400. Analizamos el período histórico 1980-2014, que termina en 2014, año final del experimento histórico de CMIP6, y la proyección 2040-2074 bajo SSP5-8.5, con SSP3-7.0 como sensibilidad. Ambos períodos tienen 35 años y 12 784 días. Es una sola simulación, así que la comparación entre pasado y futuro se hace dentro de un mismo modelo. [PENDIENTE: cita y procedencia de la simulación, según la respuesta de Katerina Goubanova, CEAZA.]

### 3.2.3 Celdas, regiones y macrozonas

La unidad de análisis es la celda de la grilla de ALADIN. La grilla recortada tiene 255 por 68 celdas (17 340) y 3 587 tienen su centro dentro del polígono de Chile (Natural Earth, 1:10 millones). De ellas, 3 331 pertenecen a las 14 regiones administrativas analizadas, entre la región de Arica y Parinacota y la de Los Lagos (aproximadamente 17,7°S a 44,0°S). Las 256 celdas restantes corresponden a la parte norte de la región de Aysén, que queda fuera del análisis porque el dominio cubre solo una fracción de ella.

Usamos una sola geografía en dos niveles. El nivel fino son las 14 regiones administrativas, y el nivel grueso son cuatro macrozonas formadas por esas mismas regiones: Norte árido (Arica y Parinacota, Tarapacá y Antofagasta; 1 179 celdas), Norte semiárido (Atacama y Coquimbo; 753 celdas), Centro mediterráneo (Valparaíso, Metropolitana, O'Higgins, Maule y Ñuble; 589 celdas) y Sur templado (Biobío, La Araucanía, Los Ríos y Los Lagos; 810 celdas). Las macrozonas cubren el 93 % de las celdas del dominio.

Para comparar píxel a píxel, interpolamos linealmente CR2MET a los centros de las celdas de ALADIN. En CR2MET hablamos de punto de grilla (0,05°) para distinguirlo de la celda. La interpolación suaviza levemente CR2MET y no conserva exactamente la precipitación integrada. Como sensibilidad, promediamos los puntos de grilla de CR2MET cuyo centro cae dentro de cada celda de ALADIN (Anexo A.1).

Hay dos maneras de resumir una zona, y cada resultado dice cuál usa. Por celda: se calcula el estadístico en cada celda y luego se promedian las celdas. Por zona: se juntan todas las rachas de la zona y se calcula un solo estadístico. Los sesgos y su significancia usan el primero, y las curvas de duración y la tabla de estadísticos por macrozona usan el segundo.

## 3.3 Definiciones operativas: días secos, rachas y rachas censuradas

Usamos tres términos con un significado fijo. La frecuencia es la fracción de días húmedos. La persistencia es la tendencia de un día seco a ser seguido por otro día seco. La duración de una racha es el número de días consecutivos que la componen. Los distinguimos porque una racha puede alargarse por haber más días secos (frecuencia) o por que estos se agrupen más (persistencia), y el objetivo 4 busca separar ambas causas.

Un día es seco si la precipitación es menor que el umbral de día húmedo R. Una racha seca es una secuencia de días secos seguidos, delimitada por días húmedos, y su duración es el número de días. Cada racha se etiqueta con el año y el mes en que empieza. Excluimos las rachas censuradas, que son las que empiezan el primer día o terminan el último día de la ventana de 35 años, porque su duración verdadera es desconocida. Como la ventana es continua, no se corta en el cambio de año.

## 3.4 Calibración del umbral de día húmedo

En CR2MET usamos R = 1 mm por día, la convención del ETCCDI. Un modelo con más días húmedos que lo observado produce rachas más cortas, porque cada día húmedo de más corta una racha. Por eso, antes de comparar duraciones, igualamos la frecuencia de días húmedos: buscamos, por bisección, el umbral de ALADIN que iguala la fracción espacial media de días húmedos de CR2MET en 1980-2014, es decir, el promedio entre celdas de la fracción de días húmedos de cada celda. Llamamos R0 a ese umbral (criterio i); su valor se informa en la Sección 4.1. Como sensibilidades usamos 1 mm en ambos productos (criterio ii) y un umbral por celda (criterio iii).

Para la duración de una racha solo importa si cada día es seco o húmedo, y eso lo resuelve el umbral, por lo que no corregimos la serie de ALADIN (Sección 3.10).

La calibración iguala la frecuencia de días húmedos y no la cantidad de lluvia; el sesgo en la cantidad se cuantifica en la Sección 4.4. En el análisis principal usamos el mismo R0 en el pasado y en el futuro, lo que supone que el sesgo del modelo en la frecuencia de días húmedos no cambia con el clima. Ese supuesto no se puede verificar con datos futuros, y por eso se examina en la prueba de control (Sección 3.8).

## 3.5 Estadísticos por celda y su significancia

En cada celda juntamos todas las rachas de los 35 años, sin importar el mes en que empiezan, y calculamos la duración media y el percentil 99, que es la duración que superan el 1 % de las rachas y describe la cola. La diferencia Δ es ALADIN menos CR2MET. Resumimos cada zona con el promedio de las diferencias por celda y con su mediana. Solo calculamos estos estadísticos en celdas con al menos 30 rachas en ambos productos, porque con menos rachas la media y sobre todo el percentil 99 son inestables. El mínimo de 30 es una convención, y no probamos otros valores. Cuántas celdas cumplen la regla se informa en la Sección 4.1.

Para saber si la diferencia de una celda es real o puede ser azar usamos un bootstrap por años. En cada una de las 1 000 réplicas sorteamos 35 años con reemplazo, de manera independiente para CR2MET y ALADIN, porque el modelo es una simulación libre y sus años no corresponden a los mismos eventos que los observados. Cada racha se asigna al año en que empieza. Con las rachas de los años sorteados recalculamos Δ, aplicando el mismo mínimo de 30 rachas: si alguno de los dos productos no lo cumple en una réplica, esa réplica se descarta (cuántas se descartan se informa en la Sección 4.3). La diferencia es significativa si el intervalo de confianza del 95 %, formado por los percentiles 2,5 y 97,5 de las réplicas, no incluye el cero. Este procedimiento se hace por separado para la media y para el percentil 99, y se resume como el porcentaje de celdas de cada zona en que ALADIN es significativamente más corto, más largo o no se distingue de CR2MET. El promedio de las diferencias por celda no tiene un intervalo de confianza propio, y las celdas vecinas no son independientes, de modo que esos porcentajes no equivalen a puntos independientes.

## 3.6 Curvas de duración por macrozona

Para describir la forma completa de la distribución usamos histogramas de duración con intervalos uniformes en el logaritmo de la duración. En cada macrozona los tres conjuntos (CR2MET histórico, ALADIN histórico y ALADIN futuro) comparten los mismos intervalos: 26 bordes entre 1 día y la duración máxima conjunta, que se redondean a enteros y se desplazan en medio día para que cada intervalo contenga un número entero de días, eliminando los repetidos. La altura es el número de rachas por intervalo dividido por su ancho, de modo que el área bajo la curva es el número total de rachas. Las rachas de todas las celdas de la macrozona se agrupan en un solo conjunto, lo que aumenta el tamaño de muestra y permite resolver la cola, a costa de mezclar la heterogeneidad espacial dentro de la zona.

## 3.7 Risk ratio de rachas largas

El cambio proyectado en el número de rachas largas es el risk ratio RR(D) = N_fut(t ≥ D) / N_hist(t ≥ D), donde N es el número de rachas de al menos D días en 2040-2074 y 1980-2014, ambos con R0 y las reglas de rachas de la Sección 3.3. Como los dos períodos tienen la misma longitud, RR(D) mayor que 1 indica más rachas largas en el futuro. Calculamos RR(D) para D entre 5 y 60 días y usamos D = 20 como valor de referencia. Ese valor no corresponde al mismo percentil de la distribución en todas las zonas (Sección 5.2), por lo que el resultado es la curva completa RR(D) y RR(20) es un punto de lectura.

Contamos las rachas por celda y las sumamos en cada región, macrozona y en el dominio. El intervalo de confianza del 95 % sale de un bootstrap por años con 1 000 réplicas: se sortean 35 años del histórico y, de manera independiente, 35 del futuro, y se recalcula el cociente. Como se remuestrean años completos para todas las celdas a la vez, el procedimiento conserva la correlación espacial.

Cuando el histórico tiene pocas rachas largas, el cociente es inestable o no está definido. Por eso informamos para cada región y valor de D la fracción de réplicas con N_hist = 0 y declaramos el RR estimable solo si N_hist es al menos 30 y ninguna réplica cae en cero. Además informamos el exceso de conteos, N_fut menos N_hist, que no tiene denominador. El 30 es una convención.

Restringimos los risk ratios a las rachas que empiezan entre marzo y noviembre, como en el paper, porque en el verano de Chile central las rachas largas forman parte del ciclo estacional esperado [VERIFICAR: respaldo de esta afirmación]. Esta restricción no representa el régimen de lluvia del norte, donde llueve en verano (Sección 5.2); por eso los resultados del Norte árido se reportan, pero no se interpretan como caracterización del norte. Mantenemos marzo a noviembre como análisis principal y mostramos la variante con rachas que empiezan en cualquier mes como sensibilidad (Sección 3.9). [Decisión del autor.]

## 3.8 Separación entre frecuencia de días húmedos y persistencia

Para saber qué parte del aumento de rachas largas se explica por la menor frecuencia de días húmedos hacemos dos pruebas independientes. La primera es una referencia de Markov estacional. En cada celda y mes estimamos la probabilidad de que un día sea seco según el estado de los últimos k días, con k igual a 1, 2 y 3, suavizando hacia el orden anterior las combinaciones raras. Con esas probabilidades simulamos 35 años, con 30 réplicas por celda y números aleatorios comunes entre escenarios, en tres mundos: con los parámetros del pasado, con los del futuro, y "solo frecuencia", que mantiene la dependencia del pasado pero usa la frecuencia seca mensual del futuro. Con las mismas reglas de rachas obtenemos el RR esperado en cada mundo y lo comparamos con el observado.

La segunda es una prueba de control. Repetimos el cálculo del RR imponiendo que el futuro tenga, en cada región, la misma fracción de días húmedos que el pasado. Para ello buscamos, por bisección y región por región, el umbral del futuro que conserva esa fracción. El análisis principal no cambia: sigue usando el mismo R0 en ambos períodos. La prueba de control es un diagnóstico y no propone un nuevo umbral.

## 3.9 Análisis de sensibilidad

Evaluamos la robustez del análisis variando una decisión a la vez: (a) el mes de inicio de las rachas, con rachas que empiezan en cualquier mes en lugar de marzo a noviembre; (b) el umbral de ALADIN, con 1 mm en ambos productos (criterio ii); (c) el escenario, con SSP3-7.0 en lugar de SSP5-8.5; (d) el umbral por celda en el pasado (criterio iii); y (e) el método de regrillado de CR2MET (Anexo A.1). Los resultados se presentan en la Sección 5.4 y en el Anexo A.

## 3.10 Alternativas descartadas

- **Bootstrap pareado entre CR2MET y ALADIN:** se descartó porque los años de una simulación libre no son los mismos eventos que los observados.
- **Cuatro ventanas de latitud y longitud (Loa, Maipo, Maule y Biobío), que usaba el paper:** se reemplazaron por macrozonas porque cubrían el 24 % de las celdas, frente al 93 % de las macrozonas, y se habían elegido sin una regla explícita.
- **Cadenas de Markov de orden superior a 3:** no se probaron. Los resultados con los órdenes 1 a 3 están en la Sección 5.3.
- **Corrección de sesgo de la precipitación (quantile mapping):** un quantile mapping por celda con una sola función anual conserva el orden de los valores diarios y por eso clasifica los días igual que el umbral calibrado en esa celda (criterio iii); con funciones por mes o estación esa equivalencia no se cumple. Corregir además la intensidad no cambia las duraciones, y el quantile mapping ajusta solo la distribución marginal de cada celda, por lo que no corrige de forma explícita la dependencia temporal, como la duración de las rachas (Maraun, 2016) [VERIFICAR: leer el pasaje de Maraun (2016) que respalda esta afirmación]. Su efecto sobre ellas pasa por la frecuencia de días húmedos, que ya controla el umbral.

## 3.11 Reproducibilidad

Todo el análisis se hace en Python (xarray, numpy, pandas, scipy y cartopy). Primero se extraen las rachas de todas las celdas con una sola regla, y de ese conjunto salen los estadísticos, las curvas, los risk ratios y las pruebas de causa. Las semillas son fijas: 42 en el bootstrap y 7 en la simulación de Markov. La relación entre cada resultado y su script está en el Anexo B.

---

## B. CAPÍTULO 4

### B.1 Sección nueva «4.1 Calibración del umbral y muestra efectiva» (va primero en el Cap. 4, antes de «Gradiente observado en CR2MET»)

**Calibración del umbral.** Con R = 1 mm por día, ALADIN tiene 30,68 % de días húmedos y CR2MET 20,94 %, un exceso de 9,7 puntos porcentuales (promedio entre las 3 587 celdas dentro del polígono de Chile, 1980-2014). El umbral de ALADIN que iguala la fracción de días húmedos de CR2MET es R0 = 5,285 mm por día (criterio i); con él, ALADIN tiene 20,94 %. Fuente: frecuencia_dias_humedos_hist.csv y umbral_frecuencia.csv.

La calibración usa las 3 587 celdas del polígono de Chile, que incluyen las 256 celdas del norte de Aysén que no entran en el análisis. En las 14 regiones analizadas, CR2MET tiene 17,72 % de días húmedos y ALADIN con R0 tiene 18,22 %, 0,50 puntos porcentuales más; con 1 mm, ALADIN tiene 27,56 %. [PENDIENTE: decidir con el profesor si se recalibra R0 sobre las 14 regiones o se mantiene y se declara.]

**Muestra efectiva.** De las 3 331 celdas de las 14 regiones, 3 270 tienen al menos 30 rachas en ambos productos. Las 61 excluidas están todas en el Norte árido (14 en Arica y Parinacota, 42 en Tarapacá y 5 en Antofagasta), que conserva 1 118 de sus 1 179 celdas. Fuente: bootstrap_por_celda_A_opcion_B_integrado_mean.npz y grilla.npz.

**Rachas censuradas.** Las rachas censuradas representan menos del 1 % de las rachas en todas las macrozonas y conjuntos de datos (máximo 0,78 %, en CR2MET histórico en el Norte árido; Tabla A.3). Como cuanto más larga es una racha, más probable es que toque un borde de la ventana, las censuradas son sobre todo rachas largas. Fuente: censuradas_por_macrozona.csv.

### B.2 Agregar al final de «4.3 Sesgo de ALADIN frente a CR2MET», después del párrafo «Fuente: bootstrap_significancia_por_zona.csv…»

El promedio de las diferencias por celda queda por debajo de la mediana: −8,9 frente a −2,9 días en la media y −49,9 frente a −21,0 días en el percentil 99. La mayoría de las celdas (1 871 de 3 270, el 57 %) están en el Norte árido y el Norte semiárido, donde las rachas de CR2MET duran en promedio 88,8 y 45,9 días por celda y las diferencias son mayores (promedio −12,1 y −18,1 días, medianas −7,5 y −11,6). En el Sur templado la diferencia es pequeña (−0,4 días en la media). Fuente: sesgo_por_macrozona.csv.

En el bootstrap por celda, 105 de las 1 118 celdas válidas del Norte árido pierden alguna réplica por no alcanzar las 30 rachas, y en 31 de ellas se pierde más del 20 % de las réplicas (máximo 65,4 %). En las otras tres macrozonas no se descarta ninguna. No cuantificamos el efecto de este descarte sobre el intervalo de esas celdas. Fuente: bootstrap_replicas_descartadas_zonas.csv.

---

## C. CAPÍTULO 5

### C.1 En «5.2 Risk ratios de rachas de 20 días o más», antes de la tabla

**Elección de D.** D = 20 días no corresponde al mismo percentil en todas las zonas. Es cercano al percentil 90 solo en el Centro mediterráneo (22 días en CR2MET). En el Norte árido el percentil 90 es de 129 días, y en el Sur templado es de 9 días, con un percentil 99 de 25 días; por eso 20 días es un evento frecuente en el primero y raro en el segundo. Fuente: tabla3_macrozonas.csv.

**Ventana de marzo a noviembre.** Entre marzo y noviembre cae entre el 85 % y el 97 % de la precipitación anual de CR2MET en las regiones de Atacama a Los Lagos, pero solo entre el 32 % y el 48 % en Arica y Parinacota, Tarapacá y Antofagasta, donde llueve en verano. Por eso la restricción no es representativa del régimen de lluvia del Norte árido, y el resultado depende de ella en magnitud (Sección 5.4). Fuente: estacionalidad_marnov.csv.

### C.2 Reemplaza el contenido de «5.4 Sensibilidad» (conserva el [PENDIENTE] para lo que falta)

**Mes de inicio de las rachas.** Con rachas que empiezan en cualquier mes, el RR(20) es 1,32 (IC95 %: 1,08 a 1,63) en el Sur templado, 1,10 (0,99 a 1,21) en el Centro mediterráneo, 1,04 (0,97 a 1,11) en el Norte semiárido y 0,99 (0,91 a 1,09) en el Norte árido. La forma se mantiene (sin cambio en el norte y aumento creciente hacia el sur), pero la magnitud baja. En Coquimbo el RR(20) pasa de 1,12 (1,01 a 1,25) con marzo a noviembre a 1,05 (0,96 a 1,16) con cualquier mes, y deja de ser significativo. Fuente: rr_ssp585_todoanio.csv y rr_ssp585.csv.

**Umbral de 1 mm en ALADIN.** El RR(20) es 1,31 (1,06 a 1,62) en el Centro mediterráneo y 2,53 (1,21 a 5,80) en el Sur templado. Fuente: rr_ssp585_1mm.csv.

**Escenario SSP3-7.0.** El RR(20) es 1,15 (0,99 a 1,35) en el Centro mediterráneo y 1,60 (1,11 a 2,44) en el Sur templado. Fuente: rr_ssp370.csv.

[PENDIENTE: umbral por celda en el pasado (criterio iii) y método de regrillado, Anexo A; y la tabla por macrozona de las demás sensibilidades.]

---

## D. DISCUSIÓN

### D.1 Agregar a los puntos de «Capítulo 6: Discusión» (antes de «Limitaciones»)

El umbral calibrado de 5,285 mm compensa a la vez el exceso de días con lluvia y la mayor intensidad de la lluvia de ALADIN, porque la calibración iguala la frecuencia y la precipitación anual de ALADIN es mayor que la de CR2MET en todas las zonas (Capítulo 4, Sección 4.4).
Como la referencia de Markov da estimaciones puntuales y el análisis usa un solo modelo y un solo miembro, la conclusión sobre la frecuencia de días húmedos se formula como «consistente con», no como demostrada.

### D.2 Agregar a «Limitaciones»

- Las rachas censuradas, menos del 1 % del total, son sobre todo rachas largas; por eso la cola extrema del Norte árido, donde hay rachas de años, queda subestimada.
- En 105 celdas del Norte árido el bootstrap descarta réplicas por no alcanzar 30 rachas (31 celdas pierden más del 20 %), y el efecto sobre sus intervalos no está cuantificado.
- La calibración de R0 usa las 3 587 celdas de Chile, que incluyen 256 celdas de Aysén fuera del análisis; en las 14 regiones la frecuencia de ALADIN con R0 queda 0,5 puntos porcentuales sobre la de CR2MET.
- Los resultados del Norte árido no representan su régimen de lluvia, que es de verano (Sección 5.2).

---

## E. ANEXOS

### E.1 «A.1 Sensibilidad al método de regrillado de CR2MET»: agregar al final del texto previo a la Tabla A.1

Con este método el umbral calibrado cambia de 5,285 a 5,280 mm y la fracción de días húmedos de CR2MET de 20,945 % a 20,951 %. El sesgo de ALADIN cambia como máximo 0,4 días en la duración media y 2,8 días en el percentil 99 (criterios i y ii, 14 regiones), y la duración media, el p90 y el p99 de CR2MET por macrozona cambian como máximo 0,1, 1 y 3 días. Las conclusiones no dependen del método de regrillado (Tablas A.1 y A.2). [VERIFICAR: estas cifras se movieron desde la antigua Sección 3.2 sin recalcular; contrastarlas con las Tablas A.1 y A.2.]

### E.2 Anexo B: filas por agregar a la tabla «Resultado | Script»

| Resultado | Script |
|---|---|
| Réplicas del bootstrap descartadas por celda (Sección 4.3) | _diagnostico_replicas_bootstrap.py |
| Fracción de días húmedos para la calibración del umbral (Sección 4.1) | _calcular_frecuencia_calibracion.py |

(La fila de `_sesgo_criterios_14regiones.py` ya está agregada a la fila del bootstrap.)

### E.3 «Decisiones vigentes» (inicio del Doc): corregir la última viñeta

Cambiar «Las rachas se cuentan desde marzo hasta noviembre y el Norte árido se declara no representativo…» por: «Los risk ratios cuentan las rachas que empiezan de marzo a noviembre; los estadísticos por celda y las curvas usan cualquier mes. El Norte árido se declara no representativo de su régimen de lluvia (decisión del autor).»

---

## F. Lo que no pude verificar

- Cifras de las Tablas A.1 y A.2 (bloque E.1): se movieron tal cual.
- La descripción de CR2MET (Boisier, 2023), el respaldo de «pocas estaciones en el norte», la cita de Maraun (2016) y la afirmación del ciclo estacional en Chile central: marcadas `[VERIFICAR]`.
- El 24 % de las ventanas del paper (bloque A, 3.10): viene del Doc; no revisé `cobertura_ventanas_macrozonas.csv` para esa cifra.
- Los porcentajes «93 %» de las macrozonas: vienen del Doc; no recalculados.
