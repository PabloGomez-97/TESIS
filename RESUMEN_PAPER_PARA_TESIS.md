# Resumen del paper para escribir la tesis sin dudas

Estado al **2026-10-22**. Se construyó leyendo el Doc del paper (texto + 5 sugerencias pendientes + 38 comentarios sin resolver), `REGISTRO_RESULTADOS.md`, `CAMBIOS_AL_PAPER.md` y las salidas en `resultados_tesis/`, `outputs_pregunta5_aladin/` y `outputs_pregunta7_bootstrap/`. Las cifras de este archivo se leyeron de esas salidas hoy; las que solo son una división de dos cifras de una tabla se marcan «derivada».

---

## 1. El paper en cinco líneas

1. **Objeto:** la distribución de duraciones de *dry spells* (rachas de días con pr < umbral) en Chile entre 17,7°S y 45,9°S, no el acumulado de precipitación ni el CDD anual.
2. **Datos:** CR2MET v2.5 (referencia, 1980–2014, umbral 1 mm) y CNRM-ALADIN64 CHP12 (un solo RCM, un solo miembro; histórico 1980–2014 y SSP5-8.5 2040–2074; SSP3-7.0 solo como sensibilidad).
3. **Paso previo:** ALADIN llueve «demasiado seguido» (30,68 % de días húmedos a 1 mm frente a 20,94 % en CR2MET), así que se calibra un umbral R₀ = 5,285 mm que iguala esa frecuencia en el promedio del dominio.
4. **Pregunta 1 (histórica):** con la frecuencia igualada, ¿reproduce ALADIN la distribución de duraciones de CR2MET? Respuesta: **reproduce el gradiente norte–sur, pero subestima la duración y sobre todo la cola.**
5. **Pregunta 2 (futura):** ¿cambia, bajo SSP5-8.5, la frecuencia de rachas largas? Respuesta: **en ALADIN hay menos rachas totales y más rachas largas; el aumento es significativo (a D = 20 d) solo en Coquimbo y La Araucanía y no resiste todas las sensibilidades; su magnitud es compatible con la menor frecuencia de días húmedos, sin evidencia de cambio en la persistencia día a día.**

---

## 2. Las preguntas, cómo se responden y qué se puede concluir

### P1. Evaluación histórica (§3.1 del paper)

| Subpregunta | Método (§) | Resultado | Conclusión permitida | Límite que va junto a la conclusión |
|---|---|---|---|---|
| P1a. ¿Cómo es la persistencia en CR2MET? | Pool por ventana, estadísticos de la distribución (§2.4–2.5; Tabla 3) | Gradiente: Loa media 65,6 d, p99 456 d; Maipo 11,0 / 119; Maule 9,5 / 83; Biobío 6,2 / 44 | Existe un gradiente norte–sur fuerte en media y en cola | Las cajas son 4 ventanas ilustrativas, no cuencas oficiales |
| P1b. ¿Reproduce ALADIN el patrón espacial y su magnitud? | Δ = ALADIN − CR2MET por celda, bootstrap de años independiente (§2.5; Fig. 4) | Patrón sí. Δ media −8,2 d (41,0 → 32,8), Δ t99 −46,3 d (215,8 → 169,6); ALADIN significativamente más corto en 66 % de las celdas (media) y 43 % (t99) | ALADIN **subestima** la duración, más en la cola | Las medias de Δ están dominadas por el norte (ver §3, «Δ mediano»); 61 celdas (18,4–22,5°S) no entran |
| P1c. ¿Cuánto del sesgo es umbral y cuánto es modelo? | Criterios i / ii / iii (Tabla 2; Tabla S1) | 1 mm en ambos: Δ −27,2 d / −134,9 d. R₀ global: −8,2 / −46,3. R₀ local: −6,8 / −36,0 | El grueso del sesgo con 1 mm es de umbral; **queda un sesgo de cola que no se elimina** | R₀ local lo reduce algo más; se usa el global por parsimonia (ver §6, pregunta 3) |
| P1d. ¿Dónde y cuánto, por ventana? | Pool por ventana (Fig. 5; Tabla 3; sesgo relativo) | Sesgo relativo de la media / p99 (criterio i): Loa −16,7 / −21,1 %; Maipo −21,2 / −26,9 %; Maule −22,4 / **−31,3 %**; Biobío −14,7 / −22,7 % | ALADIN subestima en las cuatro ventanas; en porcentaje el mayor déficit está en Maipo–Maule, en días en Loa | Con R₀ global la frecuencia de días húmedos en Maipo y Maule queda 4,2 y 3,3 pp sobre CR2MET: parte del sesgo ahí es frecuencia residual, no organización temporal |

### P2. Proyección (§3.2)

| Subpregunta | Método | Resultado | Conclusión permitida | Límite |
|---|---|---|---|---|
| P2a. ¿Cambia el número de rachas largas? | RR(D) = N_fut(t ≥ D)/N_hist(t ≥ D), regiones, inicio mar–nov, bootstrap de años independiente (§2.6; Tabla 4; Fig. 6) | D = 20 d: Coquimbo 1,12 [1,01; 1,25]; O'Higgins 1,15 [0,95; 1,40]; La Araucanía 1,72 [1,10; 2,88]; Los Lagos 2,15 [0,90; 8,50] | Más rachas largas en ALADIN en las cuatro regiones (estimación puntual); significativo solo en Coquimbo y La Araucanía | Con SSP3-7.0 esos dos IC cruzan 1; con 1 mm Los Lagos da 0,25 (no estimable). **No afirmar «aumento en las cuatro regiones»** |
| P2b. ¿Cambia la forma de la distribución? | PDFs futuro vs histórico (Fig. 5; Tabla 3) | n_spells baja en las 4 ventanas; media y p99 suben; la fracción de rachas ≥ 20 d sube en las 4 (Loa 53,7 → 56,4; Maipo 8,8 → 10,8; Maule 7,9 → 9,6; Biobío 4,1 → 5,4 %) | Menos rachas y más largas, con el exceso a la derecha | Un solo miembro: no separa señal de variabilidad interna |
| P2c. ¿Es cambio de frecuencia o de agrupamiento? | Frecuencia seca y ρ = P(seco\|seco) − P(seco\|húmedo) por región; Markov de 1.er orden (`contraste_persistencia_regiones.csv`, `descomposicion_markov_rr_regiones.csv`) | Frecuencia seca sube +2,1 a +3,9 pp (IC95 sin 0); Δρ con IC95 que incluye 0 en las 4; RR «solo por frecuencia» cae dentro del IC95 observado en 31 de 31 casos | **Sin evidencia de cambio de agrupamiento; aumento compatible con la menor frecuencia de días húmedos** | Potencia baja (IC de Δρ ≈ ±0,03); ρ de 1.er orden no mide memoria larga; «no hay evidencia» ≠ «no hay cambio». Esto es **pregunta de apoyo para interpretar P2a**, no una tercera pregunta |

**Frase de cierre que los datos sostienen:** *«ALADIN, bajo SSP5-8.5 hacia 2040–2074, proyecta menos rachas secas y más rachas largas; el aumento es significativo a D = 20 d solo en Coquimbo y La Araucanía y es de una magnitud compatible con la menor frecuencia de días húmedos, sin evidencia de cambio en la persistencia día a día. Es la respuesta de un solo RCM, un solo miembro y un solo escenario principal.»*

---

## 3. Los números: de dónde salen y qué significan

Regla: ningún número fuera de esta lista (o de `REGISTRO_RESULTADOS.md`) entra a la tesis.

| Número | Qué es exactamente | Script → salida | Qué NO significa |
|---|---|---|---|
| **3 587 celdas** | Celdas ALADIN cuyo centro cae dentro de Chile (Natural Earth 1:10m); 17,7–45,9°S | `_generar_fig03_calibracion_R0.py` → `calibracion_R0.json` | No es «Chile continental»; la latitud es la del dominio ALADIN recibido |
| **F = 20,94 % / 30,68 %** | Fracción espacial media (igual peso por celda) de días con pr ≥ 1 mm, 1980–2014, en CR2MET / ALADIN | `calibracion_R0.json` | Es un promedio del dominio, no vale celda a celda |
| **R₀ = 5,285 mm** | Umbral que hace que ALADIN tenga F = 20,94 %. Bisección: 5,2846, se usa 5,285 | `calibracion_R0.json` | **No es un umbral físico de «día seco».** ALADIN llueve 2–4 veces lo de CR2MET (Coquimbo 1 090 vs 247 mm/año; O'Higgins 2 288 / 949; La Araucanía 3 635 / 2 048; Los Lagos 5 016 / 2 722; `estacionalidad_marnov_regiones.csv`) |
| **Racha seca** | Días consecutivos con pr < umbral, asignada al año de inicio; **se excluyen las censuradas** (empiezan el primer día o terminan el último de la ventana) | todos los scripts | Una racha puede cruzar el 1 de enero; no se corta por año |
| **Censuradas < 1,1 % (Loa), < 0,2 % (resto)** | Fracción de rachas excluidas por ventana | `_cuantificar_censuradas.py` → `rachas_censuradas_por_cuenca.csv` | No son poco importantes: en Loa 22–40 censuradas por producto superan el p99 de las completas |
| **Δ = −8,2 d (media), −46,3 d (t99)** | Promedio, sobre 3 526 celdas comunes, de (media por celda de ALADIN − media por celda de CR2MET); idem con el p99 de cada celda | `_run_pregunta7_bootstrap.py` → `bootstrap_summary.csv` | Es un promedio de diferencias por celda, **no** la diferencia de los pools. El **Δ mediano es −2,2 d / −17,2 d**: la mayoría de las celdas tiene un sesgo pequeño y el promedio lo arrastra el norte |
| **41,0 / 32,8 d y 215,8 / 169,6 d** | Media y t99 espaciales de CR2MET / ALADIN sobre las mismas 3 526 celdas | `sesgo_por_ventana.csv` («Todo el dominio») | Los 42,1 / 34,0 de una versión anterior eran sobre celdas distintas: no usar |
| **66,3 % y 43,1 % de celdas** | Celdas donde el IC95 (bootstrap de años independiente) de Δ no incluye 0 y Δ < 0 (media / t99) | `bootstrap_summary.csv` | No es «66 % del territorio» (es 66 % de celdas, no ponderado por área) ni una prueba de campo: celdas vecinas no son independientes |
| **Tabla 3** | Pool de todas las rachas de todas las celdas de la ventana: n, media, mediana, p90, p99, máx., % ≥ 20 d | `_calcular_tabla3_ventanas.py` → `tabla3_estadisticos_ventanas.csv` | n_spells **no** es tamaño muestral efectivo (celdas correlacionadas); el máximo es ruidoso |
| **Sesgo relativo −16,7 / −21,2 / −22,4 / −14,7 %** | (media ALADIN − media CR2MET)/media CR2MET del pool, criterio i | `sesgo_relativo_criterios_ventanas.csv` | Es sobre el pool, no el promedio de celdas |
| **RR(D) y N_hist, N_fut** | Cociente de conteos de rachas con duración ≥ D, inicio mar–nov, regiones administrativas; pool de celdas | `_run_pregunta5_aladin.py` → `outputs_pregunta5_aladin/results.json` | No es razón de probabilidades ni de fracciones: al ser conteos, una caída del número total de rachas se refleja en RR de D chico (RR(5) < 1 en Coquimbo y O'Higgins) |
| **IC95 del RR** | Percentiles 2,5–97,5 de 1 000 réplicas: 35 años del histórico y 35 del futuro sorteados con reemplazo, **independientes**; se descartan y cuentan réplicas sin N_hist* | `results.json` | Mide variabilidad de muestreo de años, **no** incertidumbre del modelo ni del escenario |
| **Réplicas descartadas** | A D = 20: 0. Los Lagos: 128/1 000 a D = 30 y todas desde D = 35. La Araucanía: 2, 39, 101 a D = 40, 50, 60 | `results.json` («Replicas descartadas») | Por eso la curva de Los Lagos termina en D = 30 |
| **Frecuencia seca / ρ** | Frecuencia seca = fracción de días secos (mar–nov); ρ = P(seco\|seco ayer) − P(seco\|húmedo ayer), promedio sobre celdas y meses | `contraste_persistencia_regiones.csv` | ρ describe agrupamiento de **1.er orden**; no es memoria larga |
| **−6,4 % de días húmedos (dominio), −12,7 / −12,7 / −10,1 % (Maipo, Maule, Biobío)** | Cambio relativo de la fracción de días húmedos con R₀ fijo, SSP5-8.5 vs histórico | `_calcular_frecuencia_humeda_ventanas.py` → `frecuencia_humeda_ventanas.csv` | **En Loa sube +1,4 %** (2,965 → 3,006 %): el argumento de la frecuencia no explica Loa. Ver §5 |
| **61 de 3 587 celdas excluidas** | Menos de 30 rachas en algún producto; todas entre 18,4°S y 22,5°S | caché de mapas (`REGISTRO_RESULTADOS.md` §5.4) | El Δ del dominio no las incluye |
| **85–97 % y 81–96 %** | Fracción de la precipitación anual de mar–nov por región (CR2MET / ALADIN) | `estacionalidad_marnov_regiones.csv` | Justifica el recorte mar–nov de los RR, no de las PDFs |

**Derivadas, sin script propio (no usar en texto sin agregarlo a un script):** n_spells futuro/histórico por ventana: Loa −3,4 %, Maipo −11,7 %, Maule −11,2 %, Biobío −7,2 % (cociente de columnas de `tabla3_estadisticos_ventanas.csv`).

---

## 4. Cómo se responde cada bloque de Resultados (lo que el lector debe llevarse)

1. **§3.1.1 (CR2MET).** Solo establece el gradiente y que la cola de Loa y la de Biobío no son conmensurables. Conclusión: cualquier evaluación del modelo debe respetar el gradiente.
2. **§3.1.2 (mapas, Fig. 4).** Responde «¿reproduce el patrón y la magnitud?»: patrón sí, magnitud no (Δ negativo generalizado, más fuerte en la transición norte–centro). Cierra con el criterio: 1 mm ⇒ −27/−135 d; R₀ global ⇒ −8/−46; R₀ local ⇒ −7/−36. El sesgo residual se interpreta como limitación del RCM **con la salvedad** de que en Maipo y Maule parte es frecuencia residual.
3. **§3.1.3 (PDFs y Tabla 3).** Responde «¿dónde falla más?»: hay que decir si se habla de días o de porcentaje. Días: Loa. Porcentaje: Maipo–Maule (p99 −31 % en Maule).
4. **§3.2.1 (RR).** Responde P2a con la tabla de las cuatro regiones y la matización de sensibilidades (SSP3-7.0 y 1 mm). La forma de RR(D) (RR(5) < 1, creciente con D) es el hallazgo central: el cambio está en la cola, no en un desplazamiento uniforme.
5. **§3.2.2 (PDFs futuro).** Responde P2b con Tabla 3. Debe mencionar que Loa cambia poco en términos relativos **sin atribuirlo a «ya son largas»** (esa explicación no tiene respaldo; ver §5).
6. **Discusión §4.1.** Aquí entra P2c (frecuencia vs agrupamiento) y el contraste con Latoja et al. (2025): signo opuesto del cambio de n_spells y de duración, sin «reorganización», con mecanismos explícitamente no diagnosticados.

---

## 5. Hallazgos que el paper aún no resuelve (clasificados)

### BLOQUEANTE
Ninguno abierto en cifras. Los dos bloqueantes anteriores (§2.3 último párrafo y §3.1.3) están reescritos en el texto actual y coinciden con `frecuencia_humeda_ventanas.csv` y `sesgo_relativo_criterios_ventanas.csv`.

### MAYOR
1. **«El secado futuro … no consiste únicamente en una reducción de la precipitación acumulada» (cierre de §4.3 y párrafo de §4.2).** El paper **no analiza el acumulado futuro** (el párrafo de «−10 a −20 %» y la Fig. 8 se eliminaron). Sin resultado propio, la frase no está respaldada. Además choca con P2c: el aumento de rachas largas es *compatible* con la caída de días húmedos. La variante «el cambio en la duración no se deduce del signo del cambio en la precipitación total» tampoco está demostrada aquí. Decisión: eliminar la afirmación o calcular el cambio de acumulado con un script.
2. **Loa y el argumento de la frecuencia.** Con R₀ fijo, la frecuencia de días húmedos en Loa **sube** +1,4 % (y con SSP3-7.0 +22 %, sobre una base de 3 %), mientras que n_spells baja −3,4 % y la fracción ≥ 20 d sube. El texto dice que en Loa el cambio es pequeño «porque las rachas ya son prolongadas»: no hay un resultado que lo sostenga. Decir solo «el cambio relativo es menor en Loa» o `[PENDIENTE]`.
3. **Cobertura del Markov de 1.er orden.** El «31 de 31» es cierto (cuenta de `solo_frec_dentro_IC95_obs`, N_hist ≥ 100), pero en regiones húmedas el modelo Markov no reproduce el histórico observado: a D = 20, cobertura 0,70 en Los Lagos (423 vs 601 rachas) y 0,89 en La Araucanía. Los IC del observado son anchos (La Araucanía D = 30: RR obs. 3,47 frente a 2,02 «solo frecuencia»). Decir «dentro del IC95» es correcto; **no decir «explica»**. Declarar la cobertura junto al resultado.
4. **«Bajo SSP5-8.5 domina la respuesta forzada»** (Discusión, párrafo del forzante): sin cita ni resultado, con un solo miembro. Eliminar o respaldar.
5. **«La metodología es transferible a otros RCMs del dominio CHP»** (§4.2): no se probó. Eliminar o marcar como perspectiva.
6. **Petrova et al. (2024):** «el alargamiento proyectado por ALADIN podría ser conservador» se apoya en un resumen. Criterio vigente del repo: *si no está sostenido, sale*. La sugerencia pendiente la deja con `[VERIFICAR]`; se puede mantener solo hasta leer el paper.
7. **Citas sin leer / ausentes de la lista:** Stechmann y Neelin (2014), Sun et al. (2006), Chen et al. (2021); Polade (2014, 2017) respaldan días secos, no rachas más largas; Martinez-Villalobos y Neelin (2018) trata acumulaciones. Detalle en `VERIFICACION_CITAS.md`.
8. **Definición de «misma persistencia/organización temporal»:** el texto usa «organización temporal» y «agrupamiento» para lo que solo se midió con ρ de 1.er orden. Usar «persistencia día a día (1.er orden)».

### MENOR (forma y referencias cruzadas, verificado contra el texto del Doc)
1. §3.1.3 dice «Figura 6; Tabla 3» para las PDFs: son la **Figura 5**.
2. Pie de la Fig. 5: «El futuro se discute en §4.2»; el futuro por cuenca está en **§3.2.2**.
3. §2.1: «se discute en §5.3»; las limitaciones están en **§4.3**.
4. Tabla 3: «CR2MET **1**histórico» en Maule.
5. §3.2.2 conserva `[PENDIENTE: agregar esta fracción a la Tabla 3]`, pero la columna «Rachas ≥ 20 d (%)» **ya existe**: aceptar la sugerencia que lo quita.
6. Intro (párr. 12): «los modelos globales proyectan para climas mediterráneos, más días secos y rachas más largas» duplica la idea de la frase anterior.
7. Referencias: numeradas como «5» y fuera de orden alfabético (Herrera-Marín antes de Garreaud; Latoja tras Leung); Leung con puntos suspensivos.
8. Párrafos de §2.5 y §2.6 con restos de edición (espacios, «.»).

---

## 6. Las tres preguntas más duras (de comisión) y la respuesta que hoy da el paper

1. **«¿Por qué calibrar con la frecuencia de días húmedos?»** *(tu comentario sin responder en el Doc)*. Porque la duración de una racha seca la determina primero cuántos días son secos; sin igualar esa frecuencia lo que se mide en las rachas es el exceso de lluvia débil de ALADIN. Calibrar con la duración misma sería ajustar lo que se evalúa (razón de tu respuesta en el Doc, cuyo final no leí completo: revisarla). Respuesta casi lista; falta pegarla corta en §2.3.
2. **«¿Por qué dar el sesgo con R₀ global si el local lo reduce más?»** Local: Δ −6,8 / −36,0 d; global: −8,2 / −46,3 d (diferencia pequeña frente a −27 / −135 d con 1 mm). El global es **un solo umbral dependiente del modelo, el mismo que se aplica al futuro**; el local impone la igualdad celda a celda y es una sensibilidad. Costo reconocido: el global deja +4,2 y +3,3 pp de frecuencia en Maipo y Maule. Decisión de diseño: se mantiene, pero **justificar la elección en una frase de §2.3**, no en un comentario. Esto lo debe confirmar el profesor.
3. **«¿Por qué afirman que más rachas largas si dos de las cuatro regiones no son significativas y las sensibilidades cambian la conclusión?»** El paper no afirma «aumento en las cuatro regiones»: dice significativo en el análisis principal en Coquimbo y La Araucanía, con la salvedad de SSP3-7.0 y de 1 mm (Tablas S3 y S5). La respuesta de la tesis debe repetir exactamente esa formulación.

Otras previsibles: *¿es un solo miembro? sí, y el IC es de muestreo de años; ¿por qué interpolación lineal y no conservativa? Tabla S4: Δ cambia < 0,4 d (media) y < 3 d (t99); ¿por qué censuradas excluidas? porque su duración real es desconocida, a costa de perder las más largas (cuantificado).*

---

## 7. Sugerencias pendientes en Discusión (las cinco que hay en el Doc) — recomendación

| # | Dónde | Recomendación |
|---|---|---|
| 1 | §3.2.2, «(Tabla 3)» y quitar `[PENDIENTE…]` | **Aceptar.** Verificado: la columna existe. |
| 2 | §4.1 párr. 1: Sun, Chen `[VERIFICAR]`, frase de igualar la frecuencia, Petrova | Aceptar la frase nueva sobre frecuencia desigual por ventana (respaldada por `frecuencia_humeda_ventanas.csv`). Las citas siguen `[VERIFICAR]`. La frase de «conservador» queda `[VERIFICAR]` o se elimina. |
| 3 | §4.1 párr. 2: «signo opuesto a Latoja» y bloque Markov/ρ | **Aceptar**, pero agregar la salvedad de Loa (frecuencia no baja) y la cobertura del Markov (§5 MAYOR 2–3), y completar la Tabla S6 en el suplemento antes de citarla. Verificar «menos precipitación por racha húmeda» contra Latoja (no revisado hoy). |
| 4 | §4.3: celdas, censuradas, R₀ no físico, Leung | Aceptar. Leung queda `[VERIFICAR]`. |
| 5 | Cierre: «rachas secas más prolongadas, de magnitud compatible…» | Aceptar, **y revisar** el inicio del mismo párrafo («no consiste únicamente en una reducción de la precipitación acumulada»; ver MAYOR 1). |

---

## 8. Qué falta decidir con el profesor (no inventar respuesta)

1. ¿Figuras 2 y 3 al suplemento? (depende del espacio).
2. ¿Se acepta el cierre «compatible con menor frecuencia de días húmedos; sin evidencia de cambio de persistencia» en lugar de «reorganización»? Es lo que sostienen los datos.
3. ¿Se elimina la afirmación sobre acumulado futuro o se agrega un script que lo calcule?
4. ¿Se declara en el texto principal que el RR no es robusto a SSP3-7.0 ni a 1 mm? (hoy ya está en §3.2.1).
5. Metadatos de ALADIN (escenario solo por nombre de archivo; recorte a 45,9°S): preguntar a Goubanova.
6. Tabla S6 y cambios de texto del suplemento (`CAMBIOS_AL_PAPER.md`, sección «Suplemento»).

---

## 9. Qué se lleva a la tesis (sin dudas ni correcciones después)

- **Se hereda tal cual (con adaptación):** las dos preguntas, los métodos de §2, las tablas 2–4, las cifras de §3 de este archivo y la formulación cautelosa de P2.
- **Se escribe de nuevo para la tesis:** introducción, marco teórico, discusión y conclusiones (capítulos 1–2 y 6–7), con cada sección de marco teórico cerrando con «dónde se usa».
- **Se corrige primero en el paper:** los MAYOR 1–5 y los MENOR 1–5 de §5, y la Tabla S6.
- **No se reintroduce:** macrozonas, 14 regiones, «reorganización temporal», tendencias de la Fig. 7, la Fig. 8 antigua, el «−10 a −20 %».
