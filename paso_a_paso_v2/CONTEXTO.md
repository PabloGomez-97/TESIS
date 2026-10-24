# Contexto guardado (2026-10-22) — por qué empezamos de nuevo con el dominio «Chile central»

Esta carpeta **no modifica nada fuera de ella**. Los datos crudos se leerán mediante enlaces simbólicos (`pr/`, `pr1/`) cuando se llegue al paso 0.2. Todo lo que se cita abajo como «resultado previo» viene de otra carpeta (`propuesta_chile_central/`) y **se vuelve a calcular aquí, paso a paso**; no se arrastra.

## 1. Decisión y estado
- **Decisión:** el dominio de estudio pasa a ser **la unión de las cuatro regiones Valparaíso, Metropolitana, O'Higgins y Maule** (no una caja de latitud–longitud, no todo Chile, no la franja 32°–40°S de Latoja).
- **Aprobada por el profesor guía** (fecha exacta: por registrar). Motivo: alinear el dominio con el título del proyecto marco («… en Chile central») y con el alcance del trabajo.
- **Modo de trabajo nuevo:** un paso a la vez. Cada paso se explica antes, se ejecuta, se mira el resultado, y solo entonces se pasa al siguiente. Objetivo: entender cada decisión y poder defenderla, no producir rápido.
- **Texto del paper/tesis:** no se escribe hasta terminar los pasos. Las versiones v2 de los Docs (Introducción v2 y Material suplementario v2) describen el diseño de dominio nacional + 4 regiones y quedarán obsoletas en lo que dependa del dominio.

## 2. Historia de las unidades (documento interno; no pasa al texto)
1. **Conjunto original:** ventanas Loa, Maipo, Maule, Biobío (PDFs) y regiones Coquimbo, O'Higgins, La Araucanía, Los Lagos (risk ratios). Lo eligió un asistente de código en una etapa exploratoria, a partir de la petición del profesor de «4 cuencas distintas». No hay criterio escrito.
2. **Propuesta «Chile central» (22-oct):** las mismas cuatro regiones centrales para PDFs y risk ratios, con calibración y mapas nacionales (17,7°–45,9°S, 3 587 celdas). Regla escrita antes de calcular (`propuesta_chile_central/REGLA.md`). Se calculó una vez.
3. **Esta carpeta:** el dominio completo también se acota a las cuatro regiones. Es la tercera definición de unidades. Por decisión del usuario no se declara en el texto (`REGLA.md`, `CLAUDE.md` §9); se hace con una regla escrita antes de calcular, un solo cálculo y reporte de lo que salga.

## 3. Qué cambia y qué no al acotar el dominio
- **Cambia:** la calibración de R₀ (hoy 5,285 mm, calibrada con 3 587 celdas de todo Chile) se rehace en el nuevo dominio y **dará otro valor** (el mapa de R₀ local es más alto en Chile central). Cambian los mapas, la Tabla S1 (que era nacional), las sensibilidades de dominio completo y todo lo que dependa de ellos. Se pierde el gradiente norte–sur como evidencia.
- **No cambia (método):** períodos 1980–2014 y 2040–2074 (SSP5-8.5; SSP3-7.0 solo sensibilidad de RR); rachas censuradas excluidas; bootstrap de años independiente por producto y período (1 000 réplicas, semilla 42); RR(D) = N_fut(t ≥ D)/N_hist(t ≥ D) con inicio mar–nov y D = 5…60; Markov de 1.er orden (R = 40, semilla 7); contraste de persistencia (semilla 11); CR2MET interpolado linealmente a la grilla de ALADIN.
- **Nota interna:** recalibrar R₀ en Chile central probablemente mejorará la concordancia de frecuencia; es consecuencia del recorte. La regla se escribe antes de calcular.

## 4. Referencia: resultados del conjunto anterior (propuesta del 22-oct; solo para comparar después)
Todos con R₀ = 5,285 mm; **se recalcularán aquí**.
- Precipitación anual CR2MET por región: Valparaíso 457, Metropolitana 728, O'Higgins 949, Maule 1 294 mm (factor ≈ 2,8). Celdas: 108, 100, 110, 184 (502 en total).
- Sesgo relativo de ALADIN (pool anual, media / p99): Valparaíso −30,9 / −20,0 %; Metropolitana −22,8 / −23,3 %; O'Higgins −21,0 / −34,2 %; Maule −20,4 / −27,6 %.
- Exceso de frecuencia húmeda con R₀ global: +4,7, +5,1, +3,6 y +4,0 pp.
- RR(20 d) [IC95]: 1,17 [1,01; 1,36], 1,18 [0,98; 1,43], 1,15 [0,95; 1,40], 1,19 [0,95; 1,49]. Solo Valparaíso significativo; ninguna región lo es con SSP3-7.0.
- Frecuencia seca mar–nov sube +2,6 a +3,8 pp; Δρ con IC95 que incluye 0; Markov: RR esperado dentro del IC95 en 38 de 40 casos.
- Acotación de lo anterior: un solo modelo, un miembro, un escenario principal; regiones contiguas (no son estimaciones independientes); cola futura de Maule truncada por 36 censuradas.

## 5. Conceptos explicados en la conversación (para repasar)
- **Criterio i / ii / iii:** i = R₀ calibrado único (ALADIN) y 0,1 mm (CR2MET); ii = 0,1 mm en ambos; iii = R₀ local por celda.
- **Calibración de R₀:** buscar por bisección el umbral de ALADIN que da la misma fracción de días húmedos que CR2MET con 0,1 mm.
- **Media y p99, en días y en %:** el % es la diferencia relativa a CR2MET (ej. 10,4 frente a 15,1 días = −31 %). El Δ por celda (días) se promedia sobre celdas; el sesgo relativo se calcula sobre el pool de rachas de la región. Son dos medidas del mismo defecto.
- **Celdas comunes vs celdas propias:** el promedio espacial cambia según el conjunto de celdas (3 526 comunes frente a 3 549/3 550/3 587 propias). Nunca mezclar conjuntos en una misma comparación.
- **Significativo ≠ grande:** una diferencia de pocos días puede ser significativa si hay muchas rachas.
- **Markov de 1.er orden:** modelo de «moneda de juguete» que solo recuerda el día de ayer. Sirve para ver si el aumento de rachas largas se explica solo por tener más días secos. Resultado previo: sí, de forma compatible; no se probó orden 2 o 3. Limitación: potencia baja (Δρ ≈ ±0,03).
- **Latoja et al. (2025):** CR2MET, 1960–2021, franja 32°–40°S, ≈ 7 076 celdas de 0,05°; todo su análisis es en esa franja; los resultados se describen por tramos de latitud (días secos al sur de 36°S; número de rachas al sur de 34°S; largo medio solo en 35°–38°S). Cuentan rachas por semestre y año hidrológico, no rachas completas. Detalle de método **no releído** esta sesión (fuente: `VERIFICACION_CITAS.md`).
- **Resolución:** CR2MET 0,05° (≈ 5 km), ALADIN ≈ 12 km. Interpolamos CR2MET a ALADIN, no al revés.

## 6. Hallazgos de control de calidad de la sesión (no repetir los errores)
- Un gráfico de autoayuda mostraba 15,5 y 34 días: eran promedios sobre las celdas propias de cada producto; la Tabla S1 usa celdas comunes (14,9 y 32,8). Regla: un solo conjunto de celdas por comparación, dicho en el pie.
- Las cifras de titular nacionales (−8,2 y −46,3 días; 66 % y 43 % de celdas) no describen Chile central; las lidera el norte árido.
- Hay cifras de v1/v2 que dependen de las unidades anteriores (Coquimbo, La Araucanía, Los Lagos, Loa, Maipo, Biobío). No deben reaparecer.

## 7. Dónde están los archivos anteriores (solo lectura)
- Propuesta del 22-oct: `propuesta_chile_central/` (scripts, salidas, `REGISTRO.md`, `RESULTADOS_PROPUESTA.md`, `v2/`, `figuras_para_pegar/`).
- Análisis nacional del repo: `*.py` de la raíz, `resultados_tesis/`, `REGISTRO_RESULTADOS.md`, `paper_figuras/`.
- Datos crudos: `pr/` (CR2MET v2.5) y `pr1/` (ALADIN CHP12); hashes en `resultados_tesis/sha256_datos_crudos.txt`; procedencia en `PROCEDENCIA_DATOS.md`.
- Citas: `VERIFICACION_CITAS.md`.

## 8. Instrucciones del profesor guía del 2026-10-23 (reunión posterior a los pasos 0.1–2.5 de `chile_central_paso_a_paso/`)
- **Umbral de CR2MET: 0,1 mm**, «para reducir el R₀ de ALADIN sustancialmente».
- **Añadir Ñuble, Biobío y La Araucanía** («aunque sea del sur»).
- **Conectar con Stechmann y Neelin (2014):** fracción de tiempo húmedo (Ec. 33) y cutoff de duración de rachas secas t_L (Ec. 12): «una menor fracción de tiempo húmedo puede estar asociada a un aumento de t_L, aunque no necesariamente». Es algo a tener en consideración; el trabajo no está leído ni verificado.
- **Decisión:** se reinicia desde la definición del dominio en esta carpeta; `chile_central_paso_a_paso/` queda intacta (trabajo previo, solo referencia interna). Los scripts de esa carpeta se reutilizan como plantilla.

- Reservas: ver `CLAUDE.md` §9.1.

## 9. Estado de los pasos en esta carpeta
Ninguno ejecutado.
