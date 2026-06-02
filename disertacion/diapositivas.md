# Presentación de disertación — Tesis precipitación y dry spells (Chile)

**Uso:** copia cada bloque `## Diapositiva N` a una lámina en Canva/PowerPoint.  
**Figuras sugeridas:** exporta los PNG de los notebooks indicados entre paréntesis.

---

## Diapositiva 1 — Portada

**Título:** Evaluación de precipitación y periodos secos en Chile: CR2MET vs ALADIN CHP12 (1980–2014)

**Subtítulo:** Comparación en dominio común, calibración de umbrales y análisis de dry spells

**Pie:** [Tu nombre] · [Programa / Universidad] · [Fecha]

---

## Diapositiva 2 — Motivación

- La megasequía y el cambio climático exigen productos gridded confiables para planificación hídrica y estudios de extremos.
- **CR2MET** se usa como referencia observacional en Chile; **ALADIN CHP12** aporta simulación regional de alta resolución.
- Comparar ambos con el **mismo umbral fijo** (p. ej. 1 mm/día) puede sesgar conclusiones: el modelo puede “ver” más días húmedos por intensidad distinta.
- La tesis articula: (i) diagnóstico de sesgos, (ii) cambios en CR2MET, (iii) calibración metodológica (Martinez-Villalobos et al. 2022), (iv) dry spells y tendencias.

---

## Diapositiva 3 — Objetivos

1. **Caracterizar** diferencias CR2MET–ALADIN en wet days e intensidad (dominio común, 1980–2014).
2. **Cuantificar** cambios en CR2MET entre periodos P1 (1980–2000) y P2 (2001–2021).
3. **Evaluar** si la probabilidad de dry spells largos (≥20 d) cambió regionalmente (risk ratio + bootstrap).
4. **Calibrar** umbral dependiente del modelo (τ*) para comparar métricas de forma equitativa.
5. **Comparar** climatología de dry spells y **tendencias** lineales en ALADIN histórico.

---

## Diapositiva 4 — Datos y dominio común


| Producto               | Rol                                   | Periodo principal                |
| ---------------------- | ------------------------------------- | -------------------------------- |
| CR2MET v2.5            | Referencia                            | 1980–2014 / P1–P2 según análisis |
| ALADIN CHP12 histórico | Modelo regional                       | 1980–2014                        |
| Máscara Chile          | Natural Earth sobre **grilla ALADIN** | ~3 587 celdas                    |


**Regrillado:** CR2MET (lat/lon 1D) → interpolación **lineal** a centros de celda ALADIN; ALADIN en grilla nativa.

**Unidades:** ALADIN kg m⁻² s⁻¹ × 86 400 → mm/día; CR2MET ya en mm/día.

*(Figura opcional: esquema dominio + mapa máscara desde `pruea.ipynb`)*

---

## Diapositiva 5 — Cadena analítica de la tesis

```mermaid
flowchart LR
  P3[P3 Sesgo mismo τ] --> P5[P5 Δ CR2MET P1/P2]
  P3 --> P6[P6 Calibrar τ*]
  P5 --> P4[P4 RR dry spells]
  P6 --> P7[P7 Climatología dry spells]
  P7 --> P8[P8 Tendencias ALADIN]
```



**Mensaje:** primero se demuestra el sesgo; luego se separa señal climática en referencia vs modelo; la calibración habilita comparar dry spells; las tendencias cierran el periodo histórico.

---

## Diapositiva 6 — Marco teórico (Pregunta 1)

**Pregunta:** ¿Cómo se distribuye la **duración de periodos secos** bajo el modelo estocástico de Stechmann & Neelin (2014)?

- SDE de humedad con precipitación nula → tiempo de primer paso hacia umbral de lluvia **b**.
- Solución analítica: **Ecuación 6** (distribución tipo Gaussiana inversa).
- Simulación **Euler–Maruyama**: caso libre vs caso con piso físico en humedad.

**Resultado clave:** el piso físico acorta la cola de dry spells muy largos respecto al caso teórico irrestricto — coherente con limitaciones físicas de la humedad.

*(Figura: `pregunta1-stechmann.ipynb` — PDF analítica vs histograma simulado)*

---

## Diapositiva 7 — Extensión con rampa (Pregunta 2)

**Pregunta:** ¿Cambia la distribución si la precipitación sigue una **rampa** suave en humedad (no un umbral abrupto)?

- Parámetros: pendiente α, recarga E*, difusión D₀².
- Comparación área bajo histograma vs solución de referencia.

**Resultado clave:** la rampa modifica la forma de la distribución de duraciones; prepara el puente conceptual hacia umbrales y eventos definidos en datos gridded.

*(Figura: `pregunta2-stechmann.ipynb`)*

---

## Diapositiva 8 — Pregunta 3: CR2MET vs ALADIN (mismo τ = 1 mm)

**Pregunta 3 responde:** ¿En qué medida difieren CR2MET y ALADIN en **frecuencia de días húmedos** e **intensidad** sobre el mismo dominio y periodo **1980–2014**, usando wet day ≥ **1 mm/día**?


| Métrica (media espacial Chile) | CR2MET        | ALADIN         | Δ       |
| ------------------------------ | ------------- | -------------- | ------- |
| Fracción wet days              | **20,9 %**    | **30,7 %**     | +9,8 pp |
| Precip. media (todos los días) | **2,60** mm/d | **5,42** mm/d  | +2,82   |
| Precip. media (solo wet days)  | **8,66** mm/d | **13,55** mm/d | +4,89   |


**Conclusión:** con el mismo 1 mm, ALADIN es más “lluvioso” en frecuencia e intensidad — no son intercambiables sin calibración.

*(Mapas: `pruea.ipynb` — fracción wet days, media todos los días, media wet days)*

---

## Diapositiva 9 — Pregunta 5: cambio en CR2MET (P1 vs P2)

**Pregunta 5 responde:** ¿Cómo cambió la climatología de precipitación en **CR2MET** entre **P1 (1980–2000)** y **P2 (2001–2021)** a escala Chile (~30 000 celdas)?


| Métrica (media nacional)       | P1         | P2            | Δ           |
| ------------------------------ | ---------- | ------------- | ----------- |
| Fracción wet days (≥1 mm)      | **37,1 %** | **36,2 %**    | **−0,9 pp** |
| Precip. media (todos los días) | **3,85**   | **3,63** mm/d | **−0,21**   |
| Precip. media (wet days)       | **8,84**   | **8,40** mm/d | **−0,44**   |


**Patrón espacial:** centro-sur con Δ negativos (sequedización); ~20–26 % de píxeles con Δ > 0.

**Conclusión:** la referencia CR2MET sí muestra **sequedización moderada** en el siglo XXI.

*(Mapas Δ: `mapas_delta_pr.ipynb`)*

---

## Diapositiva 10 — Pregunta 4: risk ratio de dry spells largos

**Pregunta 4 responde:** ¿Cambió la probabilidad de dry spells con duración ≥ **D** días entre **P1 y P2** en cuatro regiones, usando día seco CR2MET **< 1 mm**?

- **RR = P₂/P₁**; IC95 por **bootstrap** (1 000 réplicas, remuestreo de años de inicio de racha).
- Rachas: inicio **marzo–noviembre**; regiones: Coquimbo, O'Higgins, La Araucanía, Los Lagos.

**Resultado (τ = 1 mm, umbral 20 d):**


| Región         | RR@20d   | IC95      | Significativo     |
| -------------- | -------- | --------- | ----------------- |
| Coquimbo       | ~1,14    | 0,96–1,36 | No (IC incluye 1) |
| Otras regiones | ~1,0–1,1 | amplios   | No                |


**Sensibilidad (τ = 0,1 mm):** Coquimbo RR ≈ **1,17**, IC **0,99–1,41** — aún no excluye 1.

**Conclusión:** no hay evidencia robusta de aumento **regional significativo** de spells ≥20 d en el marco MH18.

*(Figura: `pregunta5.ipynb` — curvas RR vs umbral)*

---

## Diapositiva 11 — Pregunta 6: calibración τ* (wet days)

**Pregunta 6 responde:** ¿Qué umbral en ALADIN (**τ***) iguala la **fracción integrada de wet days** de CR2MET?

**Definición:** fracción integrada = promedio espacial sobre Chile de (% días con pr ≥ τ) por celda, 1980–2014, 3 587 celdas.


| τ referencia CR2MET | F integrada CR2MET | F ALADIN (mismo τ) | *τ ALADIN**  |
| ------------------- | ------------------ | ------------------ | ------------ |
| 0,1 mm              | 23,47 %            | 40,66 %            | **3,67 mm**  |
| 1,0 mm              | 20,94 %            | 30,68 %            | **5,285 mm** |


**Referencia:** Martinez-Villalobos et al. (2022), §3b — umbral dependiente del modelo.

**Opción B (P7–P8):** día seco CR2MET < 1 mm; ALADIN < **5,285 mm**.

*(Figura: `pregunta6_umbral_wetdays.ipynb` — curva F(τ) y τ*)*

---

## Diapositiva 12 — Pregunta 7: climatología de dry spells

**Pregunta 7 responde:** Con umbrales calibrados (Opción B), ¿Cómo difieren **duración media** y **extremos (t99)** de dry spells entre ALADIN y CR2MET en **1980–2014**?


| Métrica (media espacial) | CR2MET      | ALADIN      | Δ (ALADIN−CR2MET) |
| ------------------------ | ----------- | ----------- | ----------------- |
| Duración media           | **42,5 d**  | **35,1 d**  | **−7,9 d**        |
| t99 (p99)                | **221,8 d** | **178,2 d** | **−44,7 d**       |


**Conclusión:** tras igualar frecuencia de wet days, ALADIN produce rachas **más cortas**, especialmente en la cola extrema — subestima persistencia de sequía respecto a CR2MET.

*(Mapas: `pregunta7_mapas_dryspells.ipynb`)*

---

## Diapositiva 13 — Pregunta 8: tendencias en ALADIN

**Pregunta 8 responde:** ¿Existen **tendencias lineales significativas** (p < 0,05) en duración de dry spells en ALADIN durante **1980–2014**?


| Métrica        | Pendiente media (d/década) | % celdas p<0,05 |
| -------------- | -------------------------- | --------------- |
| Duración media | **−0,027**                 | **0,035 %**     |
| t99            | **+0,001**                 | **0,12 %**      |


**Conclusión:** pendientes cercanas a cero y casi ninguna celda significativa → **sin tendencia robusta** a escala Chile en el histórico ALADIN.

*(Mapas tendencia + significancia: `pregunta8_tendencias_dryspells_aladin.ipynb`)*

---

## Diapositiva 14 — Síntesis de resultados

1. **Sesgo estructural (P3):** ALADIN más húmedo con τ = 1 mm.
2. **Señal en referencia (P5):** CR2MET se seca levemente P2 vs P1.
3. **Eventos largos (P4):** RR de spells ≥20 d **no significativo** regionalmente.
4. **Calibración (P6):** τ* ≈ **5,3 mm** en ALADIN para equiparar wet days de CR2MET a 1 mm.
5. **Persistencia (P7):** ALADIN acorta dry spells vs CR2MET con umbrales calibrados.
6. **Tendencias (P8):** sin cambio lineal detectable en 1980–2014.

**Hilo narrativo:** el modelo y la referencia no son intercambiables; la sequedización aparece en CR2MET; ALADIN no reproduce la persistencia de sequía ni muestra tendencia histórica fuerte en dry spells.

---

## Diapositiva 15 — Limitaciones

- Dominio y periodo común restringen comparación con estudios en grilla CR2MET nativa (~30k celdas).
- Interpolación lineal CR2MET → ALADIN suaviza contrastes locales.
- Umbrales y estación de rachas (mar–nov) condicionan P4 y P7.
- Bootstrap P4: incertidumbre amplia en zonas húmedas del sur.
- Tendencias P8: una sola ventana 35 años; posible baja potencia para extremos raros.

---

## Diapositiva 16 — Trabajo futuro

- Extender análisis a proyecciones ALADIN y escenarios SSP.
- Repetir P4–P8 con τ* regional (no solo nacional).
- Validar con estaciones in situ en cuencas clave.
- Acoplar hallazgos con modelo conceptual (P1–P2) para interpretar colas de distribución.

---

## Diapositiva 17 — Cierre

**Mensaje final:** La tesis muestra que comparar CR2MET y ALADIN exige **calibrar umbrales** y **separar** cambios en la referencia, sesgo del modelo y ausencia de tendencia en dry spells históricos.

**Gracias — preguntas**

---

## Anexo (respaldo, 1 diapositiva opcional)

**Notebooks:** `pregunta1-stechmann.ipynb`, `pregunta2-stechmann.ipynb`, `pruea.ipynb`, `mapas_delta_pr.ipynb`, `pregunta5.ipynb`, `pregunta6_umbral_wetdays.ipynb`, `pregunta7_mapas_dryspells.ipynb`, `pregunta8_tendencias_dryspells_aladin.ipynb`

**Referencia principal:** Martinez-Villalobos, C. A., et al. (2022). *J. Climate*, 35(12), JCLI-D-21-0590.