# Guion de auto-ayuda — Disertación (qué decir en cada diapositiva)

**Cómo usar este documento:** léelo en paralelo a las láminas (`diapositivas.md`). Cada sección tiene **tiempo sugerido**, **frase de apertura**, **puntos a vocalizar** y **transición** a la siguiente slide. Duración total orientativa: **25–35 minutos** + preguntas.

---

## Diapositiva 1 — Portada (≈30 s)

**Apertura:** «Buenos días/tardes. Presento los resultados de mi tesis sobre precipitación y periodos secos en Chile, comparando CR2MET como referencia con la simulación regional ALADIN CHP12.»

**No leas el título palabra por palabra.** Presenta tu nombre, institución y el periodo central (1980–2014).

**Transición:** «Comienzo con por qué este problema importa en Chile.»

---

## Diapositiva 2 — Motivación (≈2 min)

**Apertura:** «Chile ha vivido una megasequía prolongada; para estudiarla necesitamos productos espacializados confiables.»

**Desarrolla:**
- CR2MET es la referencia habitual en estudios nacionales; ALADIN aporta resolución de modelo regional.
- El error común: aplicar el mismo umbral de 1 mm/día a ambos productos y concluir que “llueve más” en el modelo — parte del exceso es **definición y intensidad**, no solo “más agua real”.
- La tesis no es solo un mapa de diferencias: es una **cadena** que va del marco teórico de dry spells a calibración y tendencias.

**Transición:** «Con esto en mente, planteo los objetivos específicos.»

---

## Diapositiva 3 — Objetivos (≈1,5 min)

**Apertura:** «La tesis persigue cinco objetivos encadenados.»

**Vocaliza cada número del slide** en una frase corta. Insiste en que el objetivo 4 (τ*) es el **puente metodológico** que habilita comparar dry spells de forma justa, citando Martinez-Villalobos 2022.

**Transición:** «Antes de los resultados, fijo el dominio y los datos.»

---

## Diapositiva 4 — Datos y dominio (≈2 min)

**Apertura:** «Todo lo que verán después usa un dominio común sobre la grilla de ALADIN.»

**Desarrolla:**
- ~3 587 celdas en Chile continental (máscara Natural Earth).
- CR2MET se interpola linealmente a los centros de celda de ALADIN; ALADIN queda en su grilla nativa — así comparamos **la misma geometría**.
- Periodo histórico común 1980–2014 para CR2MET vs ALADIN; P1/P2 solo donde corresponde (CR2MET hasta 2021 en P5 y P4).
- Unidades: conversión explícita de ALADIN a mm/día.

**Si preguntan por “peras y manzanas”:** «Evitamos comparar resoluciones distintas sin regrillar; el costo es suavizar CR2MET localmente.»

**Transición:** «El orden lógico de los análisis es este.»

---

## Diapositiva 5 — Cadena analítica (≈1 min)

**Apertura:** «Esta figura es la columna vertebral de la presentación.»

**Recorre el flujo:** P3 muestra sesgo → P5 muestra cambio en la referencia → P4 pregunta por eventos largos en CR2MET → P6 calibra → P7 compara dry spells ya calibrados → P8 busca tendencia temporal en ALADIN.

**Transición:** «El trabajo previo en la tesis parte del modelo conceptual.»

---

## Diapositiva 6 — Pregunta 1 (≈2 min)

**Apertura:** «La pregunta 1 es teórica: ¿cómo se distribuye la duración de un periodo seco bajo Stechmann y Neelin 2014?»

**Desarrolla:**
- SDE de humedad sin lluvia hasta cruzar umbral b.
- Ecuación 6 da la PDF analítica; simulamos con Euler–Maruyama.
- Caso con piso en humedad: cola más corta — la atmósfera no admite humedad negativa ilimitada.

**No entres en detalle numérico de parámetros** salvo que el profesor lo pida.

**Transición:** «La pregunta 2 extiende el modelo con precipitación tipo rampa.»

---

## Diapositiva 7 — Pregunta 2 (≈1 min)

**Apertura:** «Aquí la precipitación no es un interruptor, sino una rampa en humedad.»

**Mensaje:** cambia la forma de la distribución de duraciones; conecta la teoría con la necesidad de definir bien umbrales en datos reales.

**Transición:** «Pasamos a los datos observacionales y de modelo: pregunta 3.»

---

## Diapositiva 8 — Pregunta 3 (≈3 min) — **núcleo**

**Apertura:** «La pregunta 3 es: si usamos el mismo umbral de 1 mm, ¿cuánto se separan CR2MET y ALADIN entre 1980 y 2014?»

**Cifras obligatorias:**
- Wet days: 20,9 % vs 30,7 %.
- Media todos los días: 2,60 vs 5,42 mm/día.
- Media en wet days: 8,66 vs 13,55 mm/día lluvioso.

**Interpretación:** ALADIN tiene más días por encima de 1 mm y lluvia más intensa cuando llueve — **sesgo de frecuencia e intensidad**.

**Señala los mapas** si los tienes en la lámina.

**Transición:** «Separado del sesgo modelo–referencia, en CR2MET puro preguntamos si el clima cambió entre dos ventanas.»

---

## Diapositiva 9 — Pregunta 5 (≈2,5 min)

**Apertura:** «La pregunta 5 no compara modelos: mide el cambio en CR2MET entre 1980–2000 y 2001–2021.»

**Cifras:** wet days 37,1 % → 36,2 %; media diaria −0,21 mm/d; media en wet days −0,44 mm/d.

**Espacial:** centro-sur se seca; no es homogéneo — hay píxeles con aumento.

**Mensaje:** la referencia sí muestra **sequedización leve** en el siglo XXI, coherente con la megasequía.

**Transición:** «Una pregunta natural es si los periodos secos muy largos también aumentaron; eso es la pregunta 4.»

---

## Diapositiva 10 — Pregunta 4 (≈3 min)

**Apertura:** «La pregunta 4 usa el esquema de Martinez-Villalobos: risk ratio de probabilidad de dry spells ≥ D días entre P1 y P2.»

**Metodología en voz alta:**
- Día seco en CR2MET: menos de 1 mm.
- Rachas que inician entre marzo y noviembre.
- Bootstrap: remuestreamos años de inicio; IC95 si no cruza 1 → no significativo.

**Resultado principal:** en 20 días, Coquimbo tiene RR ~1,14 pero el intervalo **incluye 1** — no podemos afirmar aumento significativo. Repite que con 0,1 mm Coquimbo sube a ~1,17 pero el IC **sigue incluyendo 1**.

**Cuidado:** no digas “casi significativo” como conclusión fuerte; di «evidencia insuficiente con IC95».

**Transición:** «Dado el sesgo de la pregunta 3, calibramos el umbral en ALADIN: pregunta 6.»

---

## Diapositiva 11 — Pregunta 6 (≈3 min)

**Apertura:** «La pregunta 6 busca el τ* en ALADIN que iguala la fracción integrada de wet days de CR2MET.»

**Define fracción integrada:** promedio espacial del porcentaje de días húmedos por celda — no un solo conteo nacional.

**Cifras:** para CR2MET 1 mm (20,94 % integrada), ALADIN necesita **5,285 mm** para igualar. Para 0,1 mm, τ* ≈ 3,67 mm.

**Opción B:** en adelante, día seco CR2MET &lt; 1 mm y ALADIN &lt; 5,285 mm.

**Transición:** «Con esa calibración comparamos la climatología de dry spells: pregunta 7.»

---

## Diapositiva 12 — Pregunta 7 (≈3 min)

**Apertura:** «La pregunta 7 es: con umbrales calibrados, ¿quién tiene rachas más largas, CR2MET o ALADIN?»

**Cifras:** duración media 42,5 vs 35,1 días (−7,9 d); t99 221,8 vs 178,2 días (−44,7 d).

**Interpretación:** ALADIN **acorta** las rachas, sobre todo en extremos — subestima persistencia de sequía respecto a la referencia, incluso después de igualar frecuencia de días húmedos.

**Mapas:** comenta un patrón norte–sur si lo tienes visible.

**Transición:** «La pregunta 7 es climatología del periodo completo; la 8 pregunta si hay tendencia año a año.»

---

## Diapositiva 13 — Pregunta 8 (≈2,5 min)

**Apertura:** «La pregunta 8 estima tendencias lineales en ALADIN, 1980–2014, y marca dónde p &lt; 0,05.»

**Cifras:** pendiente media duración −0,027 d/década; solo ~0,035 % de celdas significativas. t99 ~0 d/dec; ~0,12 % significativas.

**Clarifica:** −0,027 **no** significa sequía acortándose de forma detectable — es casi cero y no significativo.

**Conclusión:** no hay tendencia espacial robusta en dry spells en este histórico.

**Transición:** «Cierro con la síntesis.»

---

## Diapositiva 14 — Síntesis (≈2 min)

**Apertura:** «En seis mensajes resume la tesis.»

Lee los seis puntos del slide **como conclusiones**, no como métodos. Cierra el hilo: referencia se seca (P5); modelo es más húmedo y rachas más cortas (P3, P7); eventos ≥20 d no cambian sig. (P4); no hay tendencia ALADIN (P8); τ* es indispensable (P6).

---

## Diapositiva 15 — Limitaciones (≈1,5 min)

**Tono honesto:** dominio común, interpolación, ventana temporal, potencia estadística en colas raras.

**Si el profesor es metodológico:** ofrece repetir P4 con umbrales documentados como sensibilidad, no como búsqueda de significancia.

---

## Diapositiva 16 — Trabajo futuro (≈1 min)

**Breve:** proyecciones, τ* regional, validación in situ, vínculo con P1–P2.

---

## Diapositiva 17 — Cierre (≈30 s)

**Apertura:** «Para cerrar: comparar CR2MET y ALADIN requiere calibrar umbrales y distinguir cambio en la referencia, sesgo del modelo y ausencia de tendencia en dry spells históricos.»

«Gracias; quedo atento a sus preguntas.»

---

## Preguntas frecuentes (chuleta rápida)

| Pregunta probable | Respuesta corta |
|-------------------|-----------------|
| ¿Por qué grilla ALADIN? | Misma geometría para ambos productos; CR2MET interpolado a centros de celda. |
| ¿Qué es fracción integrada? | Promedio espacial del % de días húmedos por píxel. |
| ¿Por qué 5,285 mm? | τ* que iguala esa fracción cuando CR2MET usa 1 mm. |
| ¿RR &gt; 1 sin significancia? | IC95 incluye 1; no rechazamos hipótesis de no cambio. |
| ¿P5 contradice P8? | P5 es cambio P1→P2 en CR2MET; P8 es tendencia lineal 1980–2014 en ALADIN — productos y métodos distintos. |
| ¿−0,027 d/dec es importante? | No: magnitud minúscula y casi ningún píxel significativo. |

---

## Checklist antes de presentar

- [ ] Exportar mapas clave de cada notebook (PNG 300 dpi).
- [ ] Verificar que las láminas P3–P8 usan las mismas cifras que las tablas de los notebooks.
- [ ] Tener a mano `guion_autoayuda.md` en tablet o segundo monitor.
- [ ] Ensayo cronometrado: 30 min + 10 min preguntas.
