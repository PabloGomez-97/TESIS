# Plan de ajustes al paper («Introducción v1»)

Doc revisado: `1QG5xHTIcqvRqTvF8k3DVCsJ9If_iT5YmWFmIj3CDdJE`, con sus 65 comentarios (Cristian Martínez y Pablo Gómez), leído el 2026-10-21. **El paper no se modificó**; esta es la lista para aplicar a mano. Todo número remite a un archivo registrado en `REGISTRO_RESULTADOS.md`. «#n» = comentario n; la tabla completa con el texto de cada uno y su estado está en `COMENTARIOS_PAPER.md`.

Criterio (decisión del 2026-10-21): **lo que los datos propios no sostienen sale del texto.** Gravedad: **BLOQUEANTE** (invalida una afirmación), **MAYOR** (un evaluador lo atacaría), **MENOR** (forma).

Verificado y sin cambios: R₀ = 5,285 mm y F = 20,94 / 30,68 % (+9,7 pp); las 12 filas de la Tabla 3; los conteos y RR de la Tabla 4 (Coquimbo 25 367 → 28 458, 1,12 [1,01; 1,25]; O'Higgins 7 015 → 8 096, 1,15 [0,95; 1,40]; La Araucanía 2 397 → 4 118, 1,72 [1,10; 2,88]; Los Lagos 601 → 1 291, 2,15 [0,90; 8,50]); los RR por D citados en §4.1 (Coquimbo D = 5: 0,94; D = 30: 1,26 [1,10; 1,43]; La Araucanía D = 30: 3,47 [1,29; 11,8]; O'Higgins D = 60: 1,49 [1,01; 2,22]); Δ = −8,2 d y −46,3 d; 43 % de celdas con t99 significativamente menor; Δ con 1 mm (−27 d, −135 d); censuradas < 1,1 % (Loa) y < 0,2 % (resto); 21–23 bins efectivos (Loa 23, Maipo 22, Maule 21, Biobío 21); 3 526 celdas comunes.

---

## 1. «Reorganización temporal»: verificada, no hay evidencia → se elimina

Se contrastó de forma directa (`_contraste_persistencia_regiones.py`; ALADIN, R₀ = 5,285 mm; marzo–noviembre; promedio sobre celdas y meses; IC95 por bootstrap de años independiente en cada período, 1 000 réplicas). El agrupamiento día a día lo describe ρ = P(seco | seco ayer) − P(seco | húmedo ayer):

| Región | Frecuencia seca, hist → fut | Δ frecuencia seca [IC95] | ρ, hist → fut | Δρ [IC95] |
|---|---|---|---|---|
| Coquimbo | 86,4 → 88,4 % | +2,1 pp [0,7; 3,4] | 0,307 → 0,288 | −0,019 [−0,049; 0,014] |
| O'Higgins | 73,6 → 76,9 % | +3,3 pp [1,3; 4,9] | 0,303 → 0,311 | +0,008 [−0,024; 0,038] |
| La Araucanía | 55,5 → 59,4 % | +3,9 pp [1,8; 5,8] | 0,341 → 0,344 | +0,004 [−0,021; 0,026] |
| Los Lagos | 44,6 → 47,0 % | +2,4 pp [0,6; 4,3] | 0,356 → 0,377 | +0,021 [−0,002; 0,044] |

- La frecuencia de días secos **sí** sube en las cuatro regiones (IC95 sin 0); ρ **no** cambia de forma detectable en ninguna (IC95 incluye 0).
- El RR esperado solo por el cambio de frecuencia (Markov de 1.er orden con ρ histórico fijo) cae dentro del IC95 del RR observado en **31 de 31** casos región × D con N_hist ≥ 100 (`descomposicion_markov_rr_regiones.csv`); a D = 20: Coquimbo 1,08 (obs. 1,12), O'Higgins 1,25 (1,15), La Araucanía 1,78 (1,72), Los Lagos 1,71 (2,15).
- Con R₀ fijo la frecuencia de días húmedos baja 6,4 % en el dominio (20,94 → 19,61 %) y 10–13 % en Maipo, Maule y Biobío (`frecuencia_humeda_ventanas.csv`).
- Límites: ρ es de primer orden (no mide memoria larga); el IC95 de Δρ mide ≈ ±0,03; un solo miembro y un solo escenario principal. **No se puede descartar un cambio pequeño de agrupamiento; sí se puede afirmar que no hay evidencia de él.**

**Dónde quitarlo y con qué reemplazarlo:**

| Dónde | Hoy dice | Reemplazar por |
|---|---|---|
| Intro, último párrafo | «cómo se reorganiza temporalmente la persistencia de los períodos secos» | «cómo cambia la distribución de duraciones de las rachas secas, en particular la frecuencia de rachas largas, y en qué medida ese cambio está asociado a la menor frecuencia de días húmedos» |
| Intro, párrafo del contraste con Latoja (#4) | «si se reorganizará hacia menos interrupciones y una mayor frecuencia de rachas muy largas» | Mantener la tensión como pregunta abierta, sin «reorganización»: «si el secado futuro repetirá la mayor intermitencia documentada o se expresará como menos rachas y más largas» (n_spells y duración sí están medidos) |
| §4.2 | «hay menos interrupciones y, por tanto, rachas más largas» | «n_spells disminuye en las cuatro ventanas, y la media y el p99 aumentan en Maipo, Maule y Biobío» (Tabla 3) |
| §5.1, párrafo 2 | «Esta reorganización es opuesta a la que Latoja…»; «se expresa en menos interrupciones de los períodos secos» | «Latoja et al. encuentran, en la serie observada, más rachas y más cortas; ALADIN proyecta menos rachas y más largas. Con R₀ fijo la frecuencia de días húmedos de ALADIN disminuye 6 % y el aumento de rachas largas es de magnitud compatible con esa caída; no se detecta un cambio en la persistencia día a día (Δρ, IC95 incluye 0). Los mecanismos no se diagnostican aquí.» |
| §5.2, §5.3 y último párrafo | «el trabajo caracteriza la reorganización de las rachas secas»; «una reorganización temporal hacia menos interrupciones» | «caracteriza el cambio en la distribución de duraciones»; «un aumento de rachas largas compatible con una menor frecuencia de días húmedos» |
| Resumen, títulos, cualquier otra | «reorganización», «reorganiza» (9 apariciones en Intro, §4.2, §5.1, §5.2 y §5.3) | quitar; la mención a Zolina et al. (2013) en la Intro es de la literatura y puede quedar atribuida |

Añadir la tabla de arriba al suplemento (Tabla S6) y una frase en §4.2. Esto responde a #29 con las dos comparaciones que pide el profesor (ver §2, Discusión).

---

## 2. Cambios por sección

### Introducción
1. **#4:** ver §1.
2. **#19 y #33, alcance:** «Chile continental» aparece 7 veces (Intro, §2.1, título y pie de la Fig. 1, §3, §5). Las celdas de Chile van de **17,7°S a 45,9°S** (verificado; coincide con tu nota en #19; «~18°S y ~44°S» no coincide con los datos). Escribir «Chile entre 17,7°S y 45,9°S». La Fig. 1 se regeneró con ese título.
3. **#31, #41, #17, síntesis bibliográfica:** el párrafo existe, pero sus citas no están verificadas (ver «Referencias»). Marcar `[VERIFICAR]` en lo que dependa de ellas hasta leerlas.
4. **#51:** «RCM» aparece por primera vez en la Tabla 1 sin definirse; definir en la primera mención del texto.

### 2. Métodos
1. **#45:** quitar «Al tener ambas ventanas la misma longitud, los conteos de rachas pueden compararse directamente» en §2.1 y su repetición en §2.6.
2. **#39:** §2.1, «se eligieron como casos de estudio» → «se eligieron para ilustrar el gradiente norte–sur» (tu respuesta en el comentario).
3. **#35:** §2.2, quitar «Dentro del dominio, ninguna de las series contiene días sin dato…» (tu decisión en el comentario). El hecho está verificado y registrado.
4. **#34, simplificar la calibración:** la motivación de R₀ está en §2.2 (párrafos «La comparación histórica…» y «Por ello el análisis principal adopta el criterio i…»), otra vez en §2.3 y otra en §2.7. Dejar en §2.2 solo las definiciones de día húmedo/seco y de racha; concentrar la justificación y la Tabla 2 en §2.3; dejar §2.7 solo como descripción de las pruebas. Figuras 2 y 3 al suplemento «si falta espacio» (tu nota): decidir con la extensión final.
5. **#5 (MAYOR):** §2.4, quitar «siguiendo la convención de PDFs… (Martinez-Villalobos y Neelin, 2018; Martinez-Villalobos et al., 2022)». Es lo que anotaste («describo cómo las construyo, no es necesario citar») y además la cita de 2018 trata acumulaciones, no rachas (`VERIFICACION_CITAS.md`). Lo mismo con la cita de 2018 en §2.5 (pool regional) y en la Intro.
6. **#42 (MAYOR):** §2.5, párrafo 2: «usando el mismo sorteo para CR2MET y ALADIN (bootstrap pareado)» → «se sortean 35 años con reemplazo de forma independiente para cada producto» (el código ya lo hace, `_run_pregunta7_bootstrap.py:191–192`). Las cifras vigentes son las del bootstrap independiente: **66,3 %** de celdas con ALADIN significativamente más corto en la media (el Doc dice 65 %) y 43,1 % en t99.
7. **#47:** §2.5, quitar «cabe esperar que alrededor del 5 % resulte significativo por azar…» (estadística básica; además la frase está duplicada).
8. **#28, documentación sin análisis nuevo:**
   - §2.5: «Δ se calcula solo en las celdas con al menos 30 rachas **en ambos productos** (3 526 de 3 587); la media y el t99 espaciales de cada producto que se reportan se promedian sobre esas mismas 3 526 celdas; los mapas de cada producto muestran sus celdas válidas (3 549 y 3 550)». Las 61 celdas excluidas están entre 18,4°S y 22,5°S.
   - §2.6: «RR(D) no está definido si N_hist(t ≥ D) = 0. En el bootstrap se descartan y se cuentan las réplicas sin rachas históricas (N_hist* = 0). A D = 20 no se descarta ninguna; en Los Lagos se descartan 128 de 1 000 réplicas a D = 30 y todas desde D = 35; en La Araucanía, 2, 39 y 101 a D = 40, 50 y 60» (`results.json`, campo «Replicas descartadas»).
9. **#37 (opcional):** justificar la interpolación lineal en una frase con la Tabla S4: promediar por área en vez de interpolar cambia Δ en menos de 0,4 d (media) y menos de 3 d (t99).
10. **BLOQUEANTE, §2.3, último párrafo:** la frase «las diferencias que persisten reflejan la organización temporal… y no solo su exceso de días húmedos» (duplicada y mal pegada) no está sostenida: R₀ global iguala la frecuencia solo en el dominio. Días húmedos (%), CR2MET 1 mm / ALADIN R₀: Loa 3,20 / 2,96; **Maipo 16,12 / 20,27**; **Maule 19,89 / 23,18**; Biobío 30,74 / 31,64. Con R₀ local el sesgo relativo de la media es Loa −23,0 %, Maipo −3,5 %, Maule −13,5 %, Biobío −12,6 % (`sesgo_relativo_criterios_ventanas.csv`). Reescribir: «parte del sesgo persiste cuando la frecuencia se iguala celda a celda (Loa, Maule, Biobío); en Maipo casi desaparece».
11. **§2.1 y Tabla 1 (MAYOR):** escribir lo que dicen los archivos (CNRM-ALADIN 6.4.1, forzado por CNRM-ESM2-1 r1i1p1f2; CR2MET v2.5, Boisier et al., creado el 29-ago-2022) y no apoyar la versión en Nabat et al. (2020), que describiría ALADIN63. Dejar fuera lo no documentado (`PROCEDENCIA_DATOS.md`).
12. **§2.6, «85–97 %» de precipitación mar–nov (MAYOR, no tenía origen):** CR2MET 85,0–97,3 %; ALADIN 81,4–95,8 % (`estacionalidad_marnov_regiones.csv`). Escribir ambos, o solo el de ALADIN, que es el que entra en los RR.
13. **§2.6, «D = 20 ≈ p90»:** el p90 de la Tabla 3 es de todo el año y de las ventanas; los RR usan inicio mar–nov y regiones. Decirlo.
14. **#61:** §2.7 cita «Tablas S1–S3, Figuras S1 y S3» y §2.1 «Material Suplementario, S6»; el suplemento tiene Tablas S1–S5 y Figuras S1–S4 en secciones S1–S6. Alinear la numeración.

### 3. Resultados históricos
1. **§3.1.2 (MAYOR):** 42,1 y 34,0 d (t99 221,1 y 175,3) son promedios de cada producto sobre celdas distintas; Δ sale de las 3 526 comunes y 42,1 − 34,0 ≠ 8,2. Usar, sobre las celdas comunes: **41,0 y 32,8 d; t99 215,8 y 169,6** (`sesgo_por_ventana.csv`, «Todo el dominio»), con Δ = −8,2 y −46,3 d sin cambio. «65 %» → «66 %».
2. **§3.1.3 (BLOQUEANTE):** el título («buen acuerdo en el centro-sur y mayor sesgo de cola en el norte»), el párrafo 1 y el cierre («reproduce mejor la forma… en Maipo, Maule y Biobío») dependen de usar días absolutos. Sesgo relativo del pool (ALADIN − CR2MET)/CR2MET, criterio i (`tabla3_estadisticos_ventanas.csv`):

   | Ventana | media | p99 |
   |---|---|---|
   | Loa | −16,7 % | −21,1 % |
   | Maipo | −21,2 % | −26,9 % |
   | Maule | −22,4 % | **−31,3 %** |
   | Biobío | −14,7 % | −22,7 % |

   Reescribir con esta tabla y con la de §2.10; decir si se habla de días o de porcentaje; quitar «buen acuerdo».
3. **§3.1.3, última frase del párrafo 1:** «aunque la frecuencia de wet days ya está emparejada» es falso por ventana (§2.10). Cambiar por «en Maipo y Maule la frecuencia con R₀ global queda 3–4 pp sobre la de CR2MET».
4. **Tabla 3:** las 12 filas están verificadas; no cambiar. Opcional: añadir columnas de sesgo relativo y de fracción de rachas ≥ 20 d.

### 4. Proyecciones
1. **§4, párrafo inicial y Figura 8 (MAYOR):** «−10 a −20 %» y «un aumento de la duración media… de varios días» no tienen script ni salida; la Fig. 8 es anterior a la regla de rachas censuradas y tiene la maquetación rota. **Eliminar el párrafo y la figura** (no responde a ninguna de las dos preguntas del paper). Si se quisiera conservar, hay que escribir un script nuevo antes de citar cifras.
2. **§4.1, robustez (MAYOR):** el texto no muestra que ninguna región «significativa» lo sea en las tres variantes. RR a D = 20 (`outputs_pregunta5_aladin*/results.json`):

   | Región | R₀ = 5,285 (principal) | 1 mm (S4) | SSP3-7.0 (S6) |
   |---|---|---|---|
   | Coquimbo | 1,12 [1,01; 1,25] | 1,21 [1,06; 1,39] | 1,09 [0,98; 1,24] |
   | O'Higgins | 1,15 [0,95; 1,40] | 1,17 [0,96; 1,47] | 1,15 [0,97; 1,37] |
   | La Araucanía | 1,72 [1,10; 2,88] | 3,97 [1,46; 17,0] | 1,53 [0,99; 2,47] |
   | Los Lagos | 2,15 [0,90; 8,50] | **0,25** [0,01; 28,1] (107 → 27 rachas) | 2,73 [1,13; 11,1] |

   Reescribir «aumento significativo en Coquimbo y La Araucanía» como «significativo en el análisis principal; con SSP3-7.0 los IC cruzan 1 (Coquimbo, La Araucanía); con 1 mm, Los Lagos no es estimable». El límite inferior de Coquimbo a D = 20 es 1,0075 (marginal) y a D = 25 el IC cruza 1 (0,988): «sostenido entre 30 y 60» necesita ese matiz. Las Tablas S3 y S5 del suplemento ya contienen estos números.
3. **§4.2 (Biobío, «la probabilidad de spells ≥ 20 días sube»):** ahora tiene salida. Fracción de rachas ≥ 20 d, ALADIN hist → fut: Loa 53,7 → 56,4 %; Maipo 8,8 → 10,8 %; Maule 7,9 → 9,6 %; Biobío 4,1 → 5,4 % (`tabla3_estadisticos_ventanas.csv`). #58: usar uno o dos ejemplos, no narrar la tabla.
4. **#61, numeración:** las figuras del Doc son 1, 2, 3, 5, 6, 8, 9. Tras eliminar la Fig. 8: Fig. 1 `fig02_dominio_cuencas_regiones.png`; 2 `fig_esquema_dryspell.png`; 3 `fig03_calibracion_R0_ftau.png`; 4 `fig05_mapas_persistencia_criterio_i.png`; 5 `fig42_pdf_cuencas_loglog_panel.png`; 6 `fig08_rr_curves_aladin.png` (si 2 y 3 pasan al suplemento, 1–4). En la figura de PDFs añadir las letras (a)–(d); en la de RR decir que la curva de Los Lagos termina en D = 30 por RR indefinido.

### 5. Discusión (#29, #43)
El profesor pide dos comparaciones cortas con la literatura en vez de más resumen de resultados:
1. **Latoja et al. (2025) frente a ALADIN:** Latoja encuentra, 1960–2021 en 32°–40°S, más rachas y más cortas; ALADIN proyecta menos rachas y más largas. Decirlo como diferencia en el signo del cambio de n_spells y de la duración, sin «reorganización» (§1). Mecanismos y forzantes aquí, y marcados como no diagnosticados. Corregir la descripción de Latoja: «al sur de 36°S» vale para días secos; el largo medio es significativo solo en 35°–38°S, y ellos cuentan rachas por semestre/año hidrológico (no comparable sin aclararlo).
2. **Petrova et al. (2024) frente al sesgo histórico de cola:** conectar con el déficit de t99 (−21 a −31 % por ventana). La frase «el alargamiento proyectado por ALADIN podría ser conservador» se apoya en una relación de modelos globales y de la racha seca anual más larga que solo se leyó en resumen (`VERIFICACION_CITAS.md`): marcar `[VERIFICAR]` o quitar.
3. **Añadir (MAYOR, no está en el paper):** (a) R₀ = 5,285 mm es un umbral estadístico, no físico: ALADIN llueve 2–4 veces lo de CR2MET (Coquimbo 1 090 frente a 247 mm/año; O'Higgins 2 288 / 949; La Araucanía 3 635 / 2 048; Los Lagos 5 016 / 2 722; `estacionalidad_marnov_regiones.csv`). (b) Leung et al. (2022) encuentra rachas más largas en los modelos que en las observaciones (signo opuesto al de este trabajo; detalle de su suplemento sin leer). (c) Un solo RCM, un solo miembro, un solo escenario principal; el bootstrap de años no mide incertidumbre de la señal. Declararlas junto a las conclusiones, no solo en §5.3.
4. **§5.3:** añadir que 61 celdas (18,4–22,5°S) no entran en los mapas y que las rachas censuradas excluidas son las más largas (en Loa, 22–40 por producto superan el p99 de las completas).

### Referencias (MAYOR)
- **Citadas y ausentes de la lista:** Stechmann y Neelin (2014) (Intro), Sun et al. (2006) y Chen et al. (2021) (§5.1). Hay que traerlas con DOI y leerlas; no se escriben de memoria.
- **Mal atribuidas:** Martinez-Villalobos y Neelin (2018) trata acumulaciones (ver §2.5); Polade et al. (2014, 2017), citadas en la Intro para «más días secos y dry spells más largos», solo respaldan días secos.
- **Sin leer:** casi todas las citas de Intro y Discusión (Garreaud, Boisier, Giorgi, Sillmann, Zhang, Trenberth, Pendergrass y Knutti, Wainwright, Lana, Araya-Osses, Salazar, Bozkurt, Báez-Villanueva, Stephens; Herrera-Marín en «log-normal y Pareto»). Verificadas con lectura: Latoja (PDF completo), Leung 2022 §3.b, Stechmann y Neelin (modelo teórico, parcial), Maraun (resumen).
- **MENOR:** «Leung, L. R., … y Zhou» truncada con puntos suspensivos; orden alfabético roto (Herrera-Marín antes de Garreaud; Latoja después de Leung).

### Suplemento (permiso: solo tablas; ya hecho)
Hecho el 2026-10-21 en las tablas: **S1** (porcentajes de celdas significativas, ahora del bootstrap independiente: 66,3 / 7,8 / 43,1 / 3,7 / 97,4 / 93,3 / 50,0) y **S4** (Δ duración media, criterio iii, promedio por área: −6,6 en lugar de −6,7). Verificadas sin cambios: S2, S3 (RR con 1 mm), S5 (SSP3-7.0) y el resto de S4. **Cambios de texto que quedan para ti** (no se tocó el texto):
1. S1, párrafo inicial: «bootstrap pareado por años de §2.5» → «bootstrap independiente».
2. Pie de la Tabla S1: las medias de CR2MET y ALADIN ya se calculan sobre las mismas celdas que Δ (columna «Celdas»; actualizado el 2026-10-21, ver `REGISTRO_RESULTADOS.md`). Si aceptas la sugerencia del pie que dice lo contrario («promedios sobre las celdas válidas de cada producto»), cámbiala por: «Todos los valores se calculan sobre las celdas comunes de la última columna (con al menos 30 rachas en ambos productos)».
3. S1, texto: «el sesgo se reduce a menos de un tercio» vale para la media (−8,2 frente a −27,2) pero no para t99 (46,3 frente a 134,9, 34 %); escribir «a cerca de un tercio».
4. Párrafo inicial del suplemento: «máscara de Chile continental» → latitudes; y la lista «(Tabla S2, Figura S1)… (Tabla S3, Figura S3)» no cubre S5 ni S6: completar.
5. S3 (Figura S2): «los valores más altos se concentran en la zona centro-norte (aproximadamente 28–37°S)»; el cuartil superior de R₀ local está entre ~25°S y ~36,5°S (mediana 30,1°S).
6. Añadir la Tabla S6 de §1 (frecuencia seca y ρ) si se acepta eliminar «reorganización».

---

## 3. Estado de los 65 comentarios (verificado contra el texto actual; detalle en `COMENTARIOS_PAPER.md`)

- **Atendidos y verificados:** #1 (censuradas), #2 y #46 (RR como razón de conteos), #6 (bootstrap por año de inicio, percentiles 2,5–97,5), #7 y #10 (jerarquía y orden: el último párrafo de la Intro pone primero el cambio futuro), #8 (bins), #13, #50 y #52 (sensibilidad: 3 frases en §3.1.2 y Fig. 4 al suplemento; falta el arreglo de cifras de §2), #36 (SSP3-7.0 hecho; falta declarar el resultado en §4.1), #59 (R₀ por bisección), #65 («Métodos» con subsección «Datos»).
- **El pasaje comentado ya no existe** (eliminado o reescrito): #14, #18, #20, #21, #22, #26, #30, #31, #38, #40, #41, #44, #48, #49, #57, #58, #64, y #32, #54, #55 (tendencias de la Fig. 7: el script se eliminó y las cifras 0,035 % y 0,115 % estaban escritas a mano en el código, **no verificables**: no deben reaparecer), y #56, #62 (subsección 4.1.3 y Fig. 10). En #31 y #41 hay síntesis bibliográfica, pero sus citas siguen sin verificar.
- **Citas pedidas:** #24, #27, #43 y #60 (Martinez-Villalobos et al. 2022, Leung et al. 2022, Petrova) están citadas; falta leerlas.
- **Pendientes (qué falta, en §2):** #4, #5, #19, #28, #29, #33, #34, #35, #37, #39, #42, #45, #47, #51, #53 (voz activa: parcial), #61, #63 (traducir al inglés al final).
- **Notas propias sin acción de texto:** #3, #9, #11, #12 (sin ancla: no sé a qué figura se refiere), #15, #16, #17, #23, #25.

---

## 4. Preguntas para el profesor
1. Eliminar «reorganización» (§1): los comentarios #4 y #29 la proponen como gancho; los datos solo sostienen «compatible con menor frecuencia de días húmedos; sin evidencia de cambio de persistencia». ¿Acepta la formulación o quiere otro contraste?
2. ¿Figuras 2 y 3 al suplemento? (depende de la extensión).
3. ¿Se elimina la Figura 8 (mapas de precipitación y duración) o se rehace con script?
4. Declarar en el texto principal que el RR no es robusto a SSP3-7.0 (Coquimbo, La Araucanía) ni a 1 mm (Los Lagos).
5. Metadatos de ALADIN (escenario solo identificable por el nombre de archivo; recorte a 45,9°S): preguntar a Goubanova (`PROCEDENCIA_DATOS.md`).
