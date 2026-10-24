# Resumen para entender lo hecho (evaluación histórica, pasos 0 a 4.6)

Documento interno, escrito el 2026-10-23. **Cada número trae entre paréntesis qué es, de dónde sale (archivo en `salidas/`) y qué significa.** Es un resumen de lo calculado; no es texto del paper.

---

## 0. Cómo leer los números (lo más importante)

**Un intervalo entre corchetes `[a; b]`** es el rango donde, con un 95 % de confianza, está el valor verdadero. Sale de repetir el cálculo 1 000 veces remuestreando años (bootstrap). Para saber si algo es «real» o «puede ser azar», hay que mirar **si el intervalo contiene el valor de referencia**:

| Qué estamos midiendo | Valor de referencia | Regla |
|---|---|---|
| Una **diferencia** (ALADIN − CR2MET), en días, en % o en puntos | **0** (0 = «son iguales») | Si el intervalo **no contiene el 0**, la diferencia es real. Si lo contiene, **no se distingue del azar**. |
| Un **cociente** (risk ratio, en la fase del futuro) | **1** (1 = «no cambia») | Si el intervalo no contiene el 1, el cambio es real. |

**Tres ejemplos con números nuestros**
- Duración media en Valparaíso: **−2,32 d [−3,49; −1,16]**. Todo el intervalo está bajo 0 → ALADIN tiene rachas **realmente** más cortas.
- Duración media en La Araucanía: **−0,03 d [−0,24; +0,20]**. El intervalo cruza el 0 → **no hay diferencia distinguible**.
- p99 en Metropolitana: **+10,82 d [−0,22; +21,66]**. El intervalo cruza el 0 por muy poco → **no se puede afirmar** que haya diferencia (aunque el valor puntual parece grande).

**Un signo menos** en una diferencia ALADIN − CR2MET significa que **ALADIN es más corto** que CR2MET.

---

## 1. Qué datos usamos

- **CR2MET v2.5** (observación grillada, 0,05° ≈ 5 km): lluvia diaria, la referencia. Se interpola linealmente a la grilla del modelo.
- **ALADIN CHP12** (modelo regional, ≈ 12–13 km, forzado por CNRM-ESM2-1): lluvia diaria, histórico y futuro (SSP5-8.5 y SSP3-7.0).
- **Período histórico:** 1980–2014 (12 784 días, 35 años). **Futuro:** 2040–2074 (12 784 días).
- **Dominio:** 972 celdas de ALADIN dentro de siete regiones (`paso_0_1_celdas_por_region.csv`):

| Región | Celdas |
|---|---|
| Valparaíso | 108 |
| Metropolitana | 100 |
| O'Higgins | 110 |
| Maule | 184 |
| Ñuble | 87 |
| Biobío | 160 |
| La Araucanía | 223 |
| **Total** | **972** (32,06° a 39,71° S) |

- **Chequeos de los datos** (`paso_1_1_chequeos.txt`, `paso_1_2_chequeos.txt`): 0 huecos (NaN) en las 972 celdas; los dos productos tienen exactamente las mismas 12 784 fechas; la interpolación de CR2MET suma 0,55 puntos de días húmedos frente a tomar el vecino más cercano (27,19 % frente a 26,64 %).
- **Los días no se comparan uno a uno** (ALADIN está forzado por un modelo global, no por el tiempo real: correlación diaria ≈ 0,18). Se comparan **estadísticos** (duración media, p99, etc.).

---

## 2. Las dos preguntas del trabajo

1. **Pregunta 1 (pasado):** una vez igualada la frecuencia de días húmedos, ¿reproduce ALADIN la distribución histórica de duraciones de rachas secas?
2. **Pregunta 2 (futuro):** bajo SSP5-8.5, ¿cambia la probabilidad de rachas largas? (**Aún no calculada.**)

Este resumen cubre solo la pregunta 1.

---

## 3. Lo que hicimos, paso a paso, con sus números

### 3.1 Qué es un «día húmedo» y cómo se calibró (pasos 2.1 a 2.5)
- **Día húmedo en CR2MET = lluvia ≥ 0,1 mm** (instrucción del profesor). Seco = menos de 0,1 mm.
- **Fracción de días húmedos de CR2MET: 27,19 %** (3 378 099 de 12 426 048 celda-días; `paso_2_1_chequeos.txt`). Significa: de cada 100 días, unos 27 llegan a 0,1 mm o más. Por región: Valparaíso 13,76 %, Metropolitana 18,95 %, O'Higgins 19,37 %, Maule 22,97 %, Ñuble 26,21 %, Biobío 32,32 %, La Araucanía 41,42 %.
- **ALADIN con el mismo umbral de 0,1 mm tiene 48,91 %** de días húmedos (`paso_2_2_chequeos.txt`): 21,7 puntos más (1,8 veces). Llueve «demasiado seguido».
- **Calibración:** se busca el umbral de ALADIN que da el mismo 27,19 %. Resultado: **R₀ = 5,742 mm** (`paso_2_3_R0.json`; verificado con un segundo método que da 5,74199 mm). Significa: en ALADIN un día cuenta como húmedo si llueve **5,742 mm o más**. R₀ es un truco estadístico para igualar frecuencias, **no un umbral físico** (la mediana de los días húmedos de ALADIN con R₀ es 20,8 mm/día).
- **Tres definiciones comparadas:** **i** (principal) ALADIN con R₀ global; **ii** 0,1 mm en ambos (sin calibrar); **iii** R₀ distinto en cada celda (`paso_2_5_R0_local_por_celda.csv`: de 1,5 a 20,0 mm, mediana 6,14).

### 3.2 Las rachas secas (paso 3.1)
- **Racha seca:** secuencia de días secos seguidos en una celda. Su año es el año del día en que **empieza**.
- **Censurada:** la que toca el primer o último día del período (no se sabe cuánto dura). **Se excluye.** Son 0,07–0,15 % de las rachas.
- **Resultado** (todas las celdas juntas, rachas completas; `paso_3_1_resumen.csv`):

| | Rachas completas | Media | Mediana | p99 | Máximo |
|---|---|---|---|---|---|
| CR2MET (0,1 mm) | 1 276 196 | 7,07 d | 4 d | 50 d | 265 d |
| ALADIN criterio i | 1 394 364 | 6,47 d | 3 d | 52 d | 243 d |

 («Media»: duración promedio de una racha; «mediana»: la mitad dura menos; «p99»: solo el 1 % más largo de las rachas dura más que ese valor; «máximo»: la racha más larga.)
- **Chequeo:** la suma de las duraciones coincide exactamente con el número de días secos, y otro método independiente da las mismas rachas.
- **Una limitación a declarar:** en el futuro SSP5-8.5, 132 rachas censuradas son más largas que el p99 de las completas de su región (Maule 41, Ñuble 26, Biobío 19, La Araucanía 43, Metropolitana 2, O'Higgins 1; `paso_3_1_censuradas_por_region.csv`). La cola futura puede estar **subestimada**.

### 3.3 Una celda de ejemplo (paso 3.3)
- Celda cercana a Santiago: CR2MET 940 rachas, media 11,72 d, p99 84,6 d; ALADIN 1 123 rachas, media 9,46 d, p99 83,6 d. Diferencia en esa celda: media **−2,26 d**, p99 **−1,05 d**.

### 3.4 Por celda: media y p99 (pasos 4.1 y 4.2)
- **Se calcula en cada una de las 972 celdas y se promedia** (`paso_4_1_chequeos.txt`):

| | CR2MET | ALADIN | Δ promedio (ALADIN − CR2MET) | Δ mediano |
|---|---|---|---|---|
| Duración media | 7,90 d | 7,02 d | **−0,88 d** (≈ −11 %) | −0,62 d |
| p99 | 53,2 d | 55,1 d | **+1,92 d** | +0,42 d |

- **¿Es real la diferencia en cada celda?** (`paso_4_2_chequeos.txt`; bootstrap de años, 1 000 réplicas):

| | ALADIN más corto | Sin diferencia clara | ALADIN más largo |
|---|---|---|---|
| Duración media | **61 %** de las celdas | 32 % | 7 % |
| p99 | 5 % | **82 %** | 13 % |

 (Se lee: en 61 de cada 100 celdas, el intervalo del Δ de la media queda por debajo de 0. En el p99, en 82 de cada 100 celdas el intervalo cruza el 0, así que no se puede afirmar nada.)
- **Por qué el p99 es incierto:** se apoya en solo el 1 % más largo de las rachas de cada celda (cada celda tiene entre 549 y 1 930 rachas, es decir, entre unas 5 y unas 19 rachas por encima del p99; cuenta aproximada); su intervalo mide 16 días en la mediana.

### 3.5 Por región (paso 4.3, `paso_4_3_sesgo_por_region.csv`)
Δ = ALADIN − CR2MET, promedio de las diferencias por celda, con su intervalo (en días):

| Región | Δ duración media | ¿Real? | Δ p99 | ¿Real? |
|---|---|---|---|---|
| Valparaíso | −2,32 [−3,49; −1,16] | Sí | +5,03 [−6,61; +17,31] | No |
| Metropolitana | −0,94 [−1,67; −0,18] | Sí | +10,82 [−0,22; +21,66] | No (cruza 0 por poco) |
| O'Higgins | −1,23 [−1,94; −0,49] | Sí | +3,88 [−8,00; +16,08] | No |
| Maule | −1,09 [−1,66; −0,62] | Sí | +0,44 [−6,70; +7,52] | No |
| Ñuble | −0,88 [−1,32; −0,49] | Sí | −2,44 [−8,55; +2,90] | No |
| Biobío | −0,56 [−0,85; −0,26] | Sí | −1,79 [−5,21; +1,77] | No |
| La Araucanía | −0,03 [−0,24; +0,20] | **No** | +1,06 [−1,74; +3,37] | No |
| **Dominio** | **−0,88 [−1,35; −0,43]** | Sí | +1,92 [−3,05; +6,72] | No |

(«¿Real?» = el intervalo **no** contiene el 0.)
- **Correlación espacial** entre las 972 celdas de CR2MET y de ALADIN (`paso_4_3b_chequeos.txt`): **0,968** (duración media) y **0,950** (p99). Se lee: ALADIN reproduce muy bien **dónde** las rachas son largas y dónde cortas (1 = idéntico; 0 = sin relación).

### 3.6 La distribución completa por región (paso 4.4, `paso_4_4_tabla3_historico.csv` y `paso_4_4_diferencias_ic.csv`)
- Juntando todas las rachas de una región (de 94 000 a 374 000 por región) se compara la **forma** de la distribución. Las curvas de CR2MET y ALADIN se superponen entre 4 y ≈ 50 días; ALADIN tiene **más rachas de 1–3 días** y **menos rachas largas**.
- Cifras del dominio, con la diferencia relativa ALADIN frente a CR2MET:

| Estadístico | CR2MET | ALADIN | Diferencia [IC95] | ¿Real? |
|---|---|---|---|---|
| Duración media | 7,07 d | 6,47 d | −8,5 % [−12,8; −3,9] | Sí |
| p99 | 50 d | 52 d | +4,0 % [−9,1; +17,4] | No |
| % de rachas ≥ 20 días | 6,78 % | 5,72 % | −1,06 puntos [−1,85; −0,32] | Sí |

 («% de rachas ≥ 20 días» = de cada 100 rachas, cuántas duran 20 días o más.)
- **% de rachas ≥ 20 d por región** (diferencia en puntos): Valparaíso −4,77 [−6,92; −2,63]; Metropolitana −1,76 [−3,16; −0,37]; O'Higgins −2,28 [−3,65; −0,87]; Maule −1,89 [−3,04; −0,76]; Ñuble −1,30 [−2,49; −0,17] (todas reales); Biobío −0,23 [−1,09; +0,58] y La Araucanía +0,41 [−0,14; +0,94] (no reales).
- **Ojo:** el p99 (el 1 % más largo) no muestra diferencia, pero la proporción de rachas ≥ 20 días (el ≈ 7 % más largo) sí. El 20 días está en una zona con muchos más datos que el p99.

### 3.7 ¿La calibración iguala la frecuencia en cada región? (paso 4.5, `paso_4_5_chequeos.txt`)
Con R₀ = 5,742 mm: diferencia de frecuencia de días húmedos ALADIN − CR2MET, en **puntos porcentuales**:

| Zona | CR2MET | ALADIN | Diferencia [IC95] | ¿Real? |
|---|---|---|---|---|
| Dominio | 27,19 | 27,19 | −0,00 [−1,52; +1,51] | No (igualado por construcción) |
| Valparaíso | 13,76 | 15,53 | +1,78 [+0,34; +3,04] | Sí |
| Metropolitana | 18,95 | 20,46 | +1,52 [−0,12; +2,98] | No |
| O'Higgins | 19,37 | 20,59 | +1,22 [−0,36; +2,65] | No |
| Maule | 22,97 | 24,67 | +1,70 [+0,05; +3,30] | Sí |
| Ñuble | 26,21 | 27,50 | +1,28 [−0,44; +3,04] | No |
| Biobío | 32,32 | 32,09 | −0,23 [−1,92; +1,54] | No |
| **La Araucanía** | 41,42 | 37,53 | **−3,88 [−5,69; −2,06]** | **Sí** |

Se lee: en el dominio entero, la frecuencia coincide (es la calibración). Dentro de las regiones, ALADIN queda algo más húmedo en el norte (hasta +1,8 puntos) y **más seco en La Araucanía (−3,9 puntos)**. Es una limitación: el criterio i no iguala la frecuencia en cada región.

### 3.8 ¿Cuánto depende todo de cómo se defina el día húmedo? (paso 4.6, `paso_4_6_sensibilidades.csv`)
Dominio, Δ = ALADIN − CR2MET, con intervalo:

| Versión | Δ duración media (días) | Δ p99 (días) |
|---|---|---|
| **i** principal (R₀ global) | −0,88 [−1,35; −0,43] (−11 %) | +1,92 [−3,05; +6,72] |
| **ii** 0,1 mm en ambos, sin calibrar | **−4,02** [−4,36; −3,69] (−51 %) | **−33,08** [−36,40; −29,59] (−62 %) |
| **iii** R₀ local por celda | −0,64 [−1,12; −0,18] (−8 %) | **+5,16** [+0,12; +10,14] |
| **iv** regrillado (CR2MET promediado por área) | −0,82 [−1,27; −0,39] (−11 %) | +2,23 [−2,65; +6,83] |

Se lee:
- **Sin calibrar (ii)** el sesgo es enorme: las rachas son la mitad de largas. Por eso la calibración es indispensable.
- **Entre las versiones calibradas (i, iii, iv)** la media se mantiene: ALADIN acorta entre un 8 y un 11 %. Es un resultado **robusto**.
- **El regrillado (iv) casi no importa.**
- **Con R₀ local (iii)**, el p99 de ALADIN sale **más largo** que el de CR2MET (+5,16 d, intervalo sobre 0). En **ninguna** versión calibrada ALADIN **subestima** la cola.
- La correlación espacial con R₀ local sube a 0,988 (media) y 0,966 (p99).

---

## 4. Respuesta a la pregunta 1 (lo sostenible)

> «Tras igualar la frecuencia de días húmedos, ALADIN reproduce el patrón espacial (correlación ≈ 0,97) y la forma general de la distribución de duraciones de las rachas secas. Tiene una duración media 8–11 % menor, con más rachas cortas y menos rachas largas (≥ 20 días). En el extremo de la cola (p99) no se distingue de CR2MET, y en ninguna definición calibrada la subestima.»

| Se puede afirmar (con respaldo) | No se puede afirmar |
|---|---|
| Patrón espacial reproducido (r = 0,97 y 0,95) | «ALADIN reproduce la cola» (sin diferencia clara no es «iguales»; los intervalos del p99 son anchos) |
| Duración media algo menor (−8 a −11 %), real en 6 de 7 regiones | «ALADIN subestima la cola» (los datos no lo respaldan) |
| Menos rachas largas (≥ 20 d) y más cortas (1–3 d) | «La frecuencia está igualada en cada región» (La Araucanía: −3,9 puntos) |
| El resultado de la media aguanta tres definiciones calibradas | «Es independiente de la definición» (sin calibrar es −51 %) |
| No hay subestimación de la cola en ninguna versión calibrada | Nada sobre estacionalidad, variabilidad entre años ni otros modelos |

---

## 5. Limitaciones que deben ir junto a la conclusión

1. **Un modelo, un miembro, un período de 35 años.** Los intervalos miden variabilidad entre años, no la incertidumbre del modelo.
2. **Comparaciones múltiples:** miramos varias métricas, siete regiones y cuatro versiones, con intervalos del 95 % sin ajuste. Una significancia aislada puede ser azar; hay que apoyarse en lo que se repite.
3. **CR2MET es un producto interpolado de estaciones**; el umbral de 0,1 mm está cerca de su ruido; su incertidumbre no está en los intervalos.
4. **R₀ es estadístico, no físico.**
5. **La convención de «día»** (ventana de 24 h) de cada producto no está documentada.
6. **El p99 por celda** se apoya en pocas rachas largas.

## 6. Qué falta

- **Estacionalidad de la evaluación histórica** (marzo–noviembre frente a verano): pendiente de decisión.
- **Pregunta 2 (futuro):** rachas futuras, risk ratios, sensibilidades (SSP3-7.0 y criterio ii), mecanismo (frecuencia frente a agrupamiento, Markov).
- **Conexión con Stechmann y Neelin (2014):** hay que leer el trabajo (aún no leído).
- **Texto del paper:** no se escribe hasta terminar los pasos.

## 7. Glosario mínimo

- **Celda:** casilla del modelo (≈ 13 km). **Región:** conjunto de celdas dentro de una región administrativa.
- **Racha seca (dry spell):** días secos seguidos. **Duración:** cuántos días dura.
- **Media / mediana / p99:** promedio / valor del medio / valor que supera solo el 1 % más largo.
- **Δ (delta):** ALADIN − CR2MET. **Δ < 0:** ALADIN más corto.
- **Bootstrap:** repetir el cálculo 1 000 veces con años sorteados para medir cuánto podría variar el resultado.
- **IC95 / intervalo:** rango plausible al 95 %. Si contiene el 0 (en una diferencia), no hay diferencia clara.
- **Punto porcentual (pp):** diferencia entre dos porcentajes (27 % frente a 25 % = 2 puntos).
- **Calibrar:** ajustar el umbral de ALADIN para que tenga la misma frecuencia de días húmedos que CR2MET.
