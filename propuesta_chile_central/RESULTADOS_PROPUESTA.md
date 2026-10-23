# Resultados de la propuesta «Chile central» (uso interno; no es texto del paper)

Calculado el **2026-10-22** con las unidades fijadas en `REGLA.md` (Valparaíso, Metropolitana, O'Higgins y Maule; las mismas cuatro para las PDFs y los risk ratios). Mapa script → salida: `REGISTRO.md`. Todas las cifras de este archivo salen de las salidas de esta carpeta; las marcadas «derivada» son cocientes de dos cifras de una salida, sin script propio.

## 0. Control de que el cambio no rompió nada
- **O'Higgins** estaba en el conjunto anterior. Con el mismo código da **exactamente** lo mismo en los conteos y el RR: 7 015 → 8 096 rachas ≥ 20 d, RR 1,15 [0,95; 1,40]; y en la frecuencia seca (73,6 → 76,9 %), la fracción mar–nov (97,3 % / 95,8 %) y Δρ puntual (+0,008). Los **IC95 de la persistencia** difieren en el tercer decimal (antes [−0,024; 0,038], ahora [−0,022; 0,038]) porque el bootstrap de ese script usa una sola secuencia aleatoria para todas las regiones y O'Higgins pasó de ser la 2.ª a la 3.ª; no es un error.
- **Verificación independiente** (`_verificacion_independiente.py`: `scipy.ndimage.label` y máscara con `shapely.intersects_xy`): diferencia **0** en los conteos de RR (10 valores de D × 4 regiones, histórico y futuro) y menor que 0,0005 (redondeo del CSV) en la Tabla 3 (n, media, mediana, p90, p99, máx., n ≥ 20; 3 productos × 4 regiones).
- **Réplicas descartadas** del bootstrap de los RR: 0 en todas las regiones y variantes (en el conjunto anterior Los Lagos y La Araucanía perdían réplicas desde D = 30–40).

## 1. Lo que cubren las unidades (descripción, no resultado)
| Región | Celdas | Latitud de las celdas | Precipitación anual CR2MET | Días húmedos 1 mm (CR2MET) | % de la precipitación en mar–nov (CR2MET / ALADIN) |
|---|---|---|---|---|---|
| Valparaíso | 108 | 32,1°–33,9°S | 457 mm | 11,3 % | 97,1 / 95,9 |
| Metropolitana | 100 | 33,1°–34,3°S | 728 mm | 16,0 % | 96,2 / 95,4 |
| O'Higgins | 110 | 34,0°–35,0°S | 949 mm | 17,5 % | 97,3 / 95,8 |
| Maule | 184 | 34,8°–36,4°S | 1 294 mm | 21,2 % | 96,1 / 95,1 |

El gradiente de precipitación es de un factor ≈ 2,8 (antes, 11). El recorte marzo–noviembre de los RR queda respaldado en las cuatro regiones (≥ 95 %).

## 2. Evaluación histórica (1980–2014, criterio i)
**Tabla 3 (pool anual por región; días):**
| Región | Serie | n | Media | Mediana | p90 | p99 | Máx. | Rachas ≥ 20 d (%) |
|---|---|---|---|---|---|---|---|---|
| Valparaíso | CR2MET | 80 682 | 15,1 | 7 | 32 | 155 | 299 | 18,3 |
| | ALADIN hist | 110 518 | 10,4 | 5 | 21 | 124 | 243 | 10,9 |
| | ALADIN fut | 96 223 | 12,2 | 5 | 24 | 142 | 261 | 13,6 |
| Metropolitana | CR2MET | 99 198 | 10,8 | 5 | 23 | 103 | 281 | 12,5 |
| | ALADIN hist | 120 703 | 8,3 | 4 | 17 | 79 | 184 | 8,3 |
| | ALADIN fut | 106 738 | 9,7 | 4 | 20 | 109 | 221 | 10,2 |
| O'Higgins | CR2MET | 110 205 | 10,5 | 5 | 22 | 117 | 250 | 11,6 |
| | ALADIN hist | 133 363 | 8,3 | 4 | 18 | 77 | 194 | 8,8 |
| | ALADIN fut | 117 741 | 9,6 | 4 | 20 | 98 | 224 | 10,6 |
| Maule | CR2MET | 210 284 | 8,8 | 4 | 19 | 76 | 206 | 9,7 |
| | ALADIN hist | 250 908 | 7,0 | 4 | 16 | 55 | 160 | 7,2 |
| | ALADIN fut | 223 841 | 8,1 | 4 | 18 | 71 | 188 | 8,9 |

- **Sesgo relativo de ALADIN (pool), media / p99:** Valparaíso −30,9 / −20,0 %; Metropolitana −22,8 / −23,3 %; O'Higgins −21,0 / −34,2 %; Maule −20,4 / −27,6 %. ALADIN subestima la media y la cola en las cuatro regiones.
- **Δ promedio por celda (mapas de dominio completo, media / t99, días):** Valparaíso −5,0 / −31,6; Metropolitana −2,9 / −21,3; O'Higgins −2,5 / −31,8; Maule −2,0 / −23,1 (todas las celdas de las cuatro regiones entran: ≥ 30 rachas).
- **Frecuencia de días húmedos con R₀ global (5,285 mm):** queda **por encima** de CR2MET en las cuatro regiones: +4,7 (Valparaíso), +5,1 (Metropolitana), +3,6 (O'Higgins), +4,0 pp (Maule). Parte del sesgo de persistencia con el criterio i es ese exceso residual.
- **Con R₀ local (criterio iii)**, sesgo relativo de la media: −6,5 / −1,2 / −7,2 / −9,2 %; del p99: **+9,0 / +21,4** / −17,1 / −13,2 %. Es decir, con la frecuencia igualada celda a celda, la cola de ALADIN en Valparaíso y Metropolitana **supera** a la de CR2MET (el déficit de cola del criterio i era, en esas dos regiones, en buena parte frecuencia residual); en O'Higgins y Maule el déficit de cola persiste.
- **Con 1 mm en ambos (criterio ii):** −52,8 / −44,4 / −40,4 / −37,2 % (media) y −58,7 / −52,4 / −57,3 / −52,6 % (p99).
- **Rachas censuradas excluidas:** 0,13–0,26 % por serie y región. Las que superan el p99 de las completas: ninguna en CR2MET ni en ALADIN histórico; en ALADIN futuro, Valparaíso 0, Metropolitana 2, O'Higgins 1 y **Maule 36** (de 360 censuradas): la cola futura de Maule está truncada.
- **Precipitación anual ALADIN / CR2MET (derivada):** 3,6; 3,1; 2,4 y 2,1 veces (Valparaíso, Metropolitana, O'Higgins, Maule). R₀ sigue siendo un umbral estadístico.
- **Regrillado (lineal frente a promedio por área):** el pool de CR2MET cambia como máximo 0,06 d en la media; el p90 no cambia y el p99 solo cambia en O'Higgins (117 → 114).

## 3. Proyección (SSP5-8.5, 2040–2074), risk ratios con inicio mar–nov
**RR(20 d) [IC95], conteos:**
| Región | Principal (R₀ = 5,285) | 1 mm (S4) | SSP3-7.0 (S6) |
|---|---|---|---|
| Valparaíso | **1,17 [1,01; 1,36]** (8 557 → 10 022) | **1,26 [1,03; 1,52]** (5 551 → 6 972) | 1,15 [0,98; 1,33] |
| Metropolitana | 1,18 [0,98; 1,43] (6 248 → 7 373) | 1,23 [0,99; 1,54] (3 773 → 4 639) | 1,12 [0,95; 1,33] |
| O'Higgins | 1,15 [0,95; 1,40] (7 015 → 8 096) | 1,17 [0,96; 1,47] (4 370 → 5 132) | 1,15 [0,97; 1,37] |
| Maule | 1,19 [0,95; 1,49] (9 137 → 10 855) | **1,47 [1,12; 2,00]** (4 507 → 6 612) | 1,14 [0,93; 1,41] |

- Los 12 RR(20) (4 regiones × 3 variantes) son **> 1** (1,12 a 1,47). Con significancia (IC95 por encima de 1): Valparaíso en el análisis principal y con 1 mm; Maule solo con 1 mm; **ninguna región con SSP3-7.0**. Ninguna región es significativa en las tres variantes.
- **Forma de la curva (principal):** RR(5) = 0,93 [0,87; 0,98], 0,93 [0,88; 0,98], 0,93 [0,88; 0,98] y 0,95 [0,90; 1,01] (menos rachas cortas), y RR crece con D. Con IC95 sobre 1: Valparaíso D = 20, 30 y 35; Metropolitana D = 30, 35 y 60; O'Higgins D = 60 (1,49 [1,01; 2,22]); Maule D = 35 y 40. Son 40 pares región × D evaluados: la curva se interpreta en conjunto, no por punto.
- **Distribución (Tabla 3):** n_spells disminuye en las cuatro regiones: −12,9, −11,6, −11,7 y −10,8 % (derivadas); la media sube +17,1, +16,0, +16,3 y +16,2 % y el p99 +14,5, +38,0, +27,3 y +29,1 % (derivadas); la fracción de rachas ≥ 20 d pasa de 10,9 a 13,6 %, de 8,3 a 10,2 %, de 8,8 a 10,6 % y de 7,2 a 8,9 %.
- **Días húmedos con R₀ fijo (futuro frente a histórico):** −14,4 % (Valparaíso), −12,2 % (Metropolitana), −13,1 % (O'Higgins) y −12,2 % (Maule); en el dominio, −6,4 %.

## 4. ¿Frecuencia o agrupamiento?
- **Frecuencia seca (mar–nov), histórico → futuro:** 80,0 → 82,7 %; 74,0 → 76,9 %; 73,6 → 76,9 %; 68,7 → 72,5 %. Δ = +2,6 [1,0; 4,3], +2,8 [0,8; 4,5], +3,3 [1,4; 5,0], +3,8 [1,6; 5,8] pp: sube en las cuatro (IC95 sin 0).
- **Persistencia de 1.er orden ρ:** Δρ = −0,001 [−0,032; 0,029], +0,000 [−0,030; 0,030], +0,008 [−0,022; 0,038], +0,011 [−0,018; 0,040]: el IC95 incluye 0 en las cuatro; precisión ≈ ±0,03.
- **Referencia Markov (solo cambia la frecuencia):** el RR esperado cae dentro del IC95 del observado en **38 de 40** casos región × D con N_hist ≥ 100 (en el conjunto anterior eran 31 de 31). Los dos casos fuera son Metropolitana D = 40 (esperado 1,49; IC95 observado [0,94; 1,45]) y Maule D = 25 (esperado 1,52; IC95 [0,85; 1,49]), en ambos por **encima** del observado.
- A D = 20, esperado solo por frecuencia → observado: Valparaíso 1,16 → 1,17; Metropolitana 1,23 → 1,18; O'Higgins 1,25 → 1,15; Maule 1,40 → 1,19. **En las tres regiones del sur la frecuencia sola predice un aumento algo mayor que el observado** (el conjunto anterior no mostraba esto). La cobertura histórica del Markov (rachas simuladas / observadas) es 0,92–1,01 a D = 20 (antes, 0,70–0,89 en las regiones húmedas): la referencia es más fiable aquí.

## 5. Qué se puede concluir (y qué no)
**Sostenido por los datos:**
1. ALADIN reproduce el patrón y subestima la duración y la cola en las cuatro regiones (media −20 a −31 %, p99 −20 a −34 % con el criterio i). Con R₀ local el déficit de cola se atenúa o se invierte en Valparaíso y Metropolitana.
2. Bajo SSP5-8.5 hay menos rachas (≈ −11 a −13 %) y más rachas largas: RR(20) entre 1,15 y 1,19 en las cuatro regiones, con RR(5) < 1 y RR creciente con D. La significancia a D = 20 es solo de Valparaíso y no resiste SSP3-7.0.
3. El aumento de rachas largas es de magnitud compatible con la menor frecuencia de días húmedos (−12 a −14 %); la persistencia de 1.er orden no cambia de forma detectable; no hay evidencia de agrupamiento adicional (potencia limitada, Markov de 1.er orden).

**No sostenido / con salvedad:**
- No se puede afirmar «aumento significativo en las cuatro regiones» (solo Valparaíso a D = 20).
- Las cuatro regiones son contiguas y comparten forzante a gran escala: la homogeneidad de los RR (1,15–1,19) no equivale a cuatro estimaciones independientes.
- Un solo modelo, un solo miembro, un solo escenario principal; el bootstrap de años mide variabilidad de muestreo, no la incertidumbre de la señal.

## 6. Comparación con el conjunto anterior (solo para ti)
| Aspecto | Conjunto anterior | Propuesta |
|---|---|---|
| Unidades | 4 cajas (Loa, Maipo, Maule, Biobío) y 4 regiones (Coquimbo, O'Higgins, La Araucanía, Los Lagos) | 4 regiones centrales para ambos análisis |
| Precipitación anual CR2MET | 247–2 722 mm | 457–1 294 mm |
| Sesgo relativo de la media (criterio i) | −14,7 a −22,4 % | −20,4 a −30,9 % |
| Sesgo relativo del p99 (criterio i) | −21,1 a −31,3 % | −20,0 a −34,2 % |
| Exceso de frecuencia húmeda con R₀ global | +4,1 (Maipo), +3,3 (Maule), +0,9 (Biobío); −0,2 (Loa) | +3,6 a +5,1 pp en las cuatro |
| RR(20) | 1,12 · 1,15 · 1,72 · 2,15 (significativos: Coquimbo y La Araucanía) | 1,17 · 1,18 · 1,15 · 1,19 (significativa: Valparaíso) |
| Robustez de la significancia | Coquimbo y La Araucanía: no resisten SSP3-7.0; Los Lagos con 1 mm = 0,25 | Valparaíso no resiste SSP3-7.0; Maule solo significativa con 1 mm |
| Réplicas descartadas | Los Lagos y La Araucanía a D ≥ 30–40 | Ninguna |
| Markov frente a observado | 31 de 31 dentro del IC95 | 38 de 40 |
| Cola truncada por censuradas (ALADIN fut., sobre el p99) | Loa 22, Maule 39, Biobío 21, Maipo 12 | Maule 36, Metropolitana 2, O'Higgins 1 |
| Frase «ALADIN llueve 2–4 veces lo de CR2MET» | falsa (1,8–4,4) | 2,1–3,6: verdadera con estas regiones |

La conclusión cualitativa central se mantiene (menos rachas, más rachas largas, compatible con la menor frecuencia de días húmedos). Cambia **dónde** se ve significancia, y desaparecen los RR extremos y los intervalos inestables del sur húmedo.

## 7. Si se adopta, qué hay que cambiar en el paper
- **Título y alcance:** el título aprobado habla de «gradiente climático»; con estas unidades corresponde «Chile central» (los mapas de dominio completo siguen cubriendo 17,7°–45,9°S).
- **Texto y figuras:** §2.1 (unidades y Tabla 1), Figura 1, Tabla 3, Figura 5, Tabla 4, Figura 6; §3.1.2–3.1.3 (cifras por región; el Δ nacional no cambia); §3.2; Discusión (31 de 31 → 38 de 40; frases sobre Loa y el norte árido; límites de censuradas y de precipitación anual) y todas las cifras de los párrafos de limitaciones.
- **Suplemento:** S2 y S4 (regiones), S3, S5, S6, Figuras S1, S3, S4.
- **Lo que no cambia:** calibración de R₀, mapas, Tabla S1, sensibilidad al mínimo de rachas.
- Antes de proponer texto: auditar de nuevo cada cifra contra estas salidas, como se hizo con el paper actual.

## 8. Límites de esta propuesta
- El gradiente climático de las unidades se acorta (≈ 3 veces en precipitación): se pierden el extremo árido (Loa) y el húmedo (Los Lagos) como unidades, aunque siguen en los mapas de dominio completo.
- Cuatro regiones contiguas: poca independencia espacial.
- El conjunto se fijó conociendo los resultados del anterior (declarado en `REGLA.md`).
- R₀ global deja +3,6 a +5,1 pp de frecuencia húmeda en estas regiones; el criterio i mezcla ese exceso con el sesgo de agrupamiento.
- La cola futura de Maule está truncada por 36 rachas censuradas que superan el p99 de las completas: el alargamiento proyectado de la cola en Maule puede estar subestimado.
- Aplica a esta propuesta lo mismo que a todo el trabajo: un RCM, un miembro, un escenario principal.

## 9. Qué no se hizo
- No se revisó el contenido de ninguna cita ni se escribió texto para el paper.
- No se probó ninguna variante de unidades (REGLA.md, compromiso 1).
- La Figura 1 se revisó a ojo; el resto de las figuras también. No se comparó figura a figura con la anterior.
