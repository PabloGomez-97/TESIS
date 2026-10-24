# Plan de pasos — Chile central (dominio = unión de Valparaíso, Metropolitana, O'Higgins y Maule)

Orden de trabajo: **un paso a la vez**. En cada paso: (1) explico la idea, (2) tú me dices si la entendiste, (3) lo calculo, (4) miramos el resultado, (5) lo registro. No avanzo al siguiente sin tu OK. «Antes» = lo que hicimos en el análisis nacional / propuesta del 22-oct.

Leyenda de la última columna: **=** igual que antes · **≠** cambia al acotar el dominio · **nuevo** no existía como paso separado.

## Fase 0 — Preparar el terreno
| # | Qué hacemos | Para qué | Entrada → salida | Idea que hay que entender | Antes |
|---|---|---|---|---|---|
| 0.1 | Definir el dominio: qué celdas de ALADIN pertenecen a las 4 regiones (centro de la celda dentro del polígono) | Todo lo que sigue se calcula solo en esas celdas | Grilla ALADIN + polígonos Natural Earth → máscara (≈ 502 celdas) y mapa | Qué es una celda, qué es una máscara, por qué el centro y no el área | ≠ (antes 3 587 celdas) |
| 0.2 | Abrir y revisar los datos: unidades (mm/día), calendario, períodos, huecos, hashes | No calcular sobre datos mal leídos | `pr/` (CR2MET v2.5) y `pr1/` (ALADIN CHP12) → ficha de datos | Qué contiene cada archivo y qué no está documentado (`PROCEDENCIA_DATOS.md`) | = |

## Fase 1 — Hacer comparables los dos productos
| # | Qué hacemos | Para qué | Entrada → salida | Idea que hay que entender | Antes |
|---|---|---|---|---|---|
| 1.1 | **Interpolación lineal (bilineal) de CR2MET a la grilla de ALADIN** (de 0,05° ≈ 5 km a ≈ 12 km) | Comparar celda a celda sin mezclar resoluciones | CR2MET diario + grilla ALADIN → CR2MET diario en las celdas de ALADIN | Cómo se estima un valor en el centro de una celda con los 4 vecinos; por qué interpolamos CR2MET a ALADIN y no al revés; se interpola antes de enmascarar para no perder datos en el borde | = (se hace en todo CR2MET y luego se enmascara) |
| 1.2 | Alinear períodos y calendarios: 1980–2014 (histórico) y 2040–2074 (futuro) | Que cada año y cada día coincidan | Series de 1.1 y ALADIN → series diarias alineadas | Qué pasa con años bisiestos y con calendarios distintos | = |
| 1.3 | Ver un día y una celda: mapa y serie de lluvia de CR2MET frente a ALADIN | Tener una imagen concreta de qué comparamos | Series de 1.2 → 1–2 figuras de inspección (no van al paper) | Que ALADIN llueve «más fino» y más seguido | nuevo |

## Fase 2 — Qué es un día húmedo y cómo calibrar el umbral
| # | Qué hacemos | Para qué | Entrada → salida | Idea que hay que entender | Antes |
|---|---|---|---|---|---|
| 2.1 | Fracción de días húmedos de CR2MET con 1 mm en el dominio | Fijar la «verdad» observada | CR2MET 1980–2014 → F_CR2MET | Qué es una fracción de días húmedos | ≠ (antes 20,94 % nacional) |
| 2.2 | Fracción de ALADIN con 1 mm | Ver cuánto «exceso de días húmedos» tiene el modelo | ALADIN 1980–2014 → F_ALADIN(1 mm) | Por qué un modelo con llovizna constante da demasiados días húmedos | ≠ |
| 2.3 | **Calibrar R₀ por bisección** en el dominio: el umbral de ALADIN que da la misma fracción que CR2MET | Que «día húmedo» signifique lo mismo en los dos | F_CR2MET y ALADIN → R₀ nuevo | Cómo funciona la bisección; por qué R₀ es estadístico y no físico; **R₀ nuevo ≠ 5,285 mm** | ≠ (decisión central) |
| 2.4 | Figura de calibración: F(R₀) y dónde se cruza con F_CR2MET | Mostrar que R₀ está bien elegido | Curva de 2.3 → Figura (equivalente a la antigua Fig. 3) | Leer una curva de calibración | ≠ |
| 2.5 | Definir los 3 criterios: i (R₀ calibrado), ii (1 mm en ambos), iii (R₀ local por celda) y calcular R₀ local | Sensibilidades del sesgo | 2.3 y datos → mapa y tabla de R₀ | Qué cambia cada criterio y qué responde | ≠ |

## Fase 3 — Contar rachas
| # | Qué hacemos | Para qué | Entrada → salida | Idea que hay que entender | Antes |
|---|---|---|---|---|---|
| 3.1 | Marcar cada día como seco/húmedo y extraer las rachas secas por celda (año = año de inicio) | Es la materia prima de todo | Series de 1.2 + umbrales de 2 → lista de rachas por celda | Qué es una racha, qué es una racha censurada | = |
| 3.2 | Excluir censuradas (tocan el primer o el último día del período) y cuantificarlas | No inventar duraciones | Rachas de 3.1 → rachas completas + tabla de censuradas | Por qué no se sabe cuánto dura una racha cortada | = |
| 3.3 | Esquema de la definición operativa (figura conceptual) | Que el lector entienda la definición | — → figura (equivalente a la antigua Fig. 2) | — | = |
| 3.4 | Caso de ejemplo: una celda, su lista de rachas, su media y su p99 a mano | Entender los números antes de ver mapas | Una celda → tabla pequeña | Qué es media y qué es p99 (percentil 99) | nuevo |

## Fase 4 — Evaluación histórica (¿ALADIN reproduce la persistencia?)
| # | Qué hacemos | Para qué | Entrada → salida | Idea que hay que entender | Antes |
|---|---|---|---|---|---|
| 4.1 | Estadísticos por celda (media y p99), con mínimo de 30 rachas; mapas de CR2MET, ALADIN y diferencia Δ | Ver dónde acierta y dónde falla el modelo | Rachas de 3.2 → 6 mapas (equivalente a la Fig. 4) | Qué es una estadística «espacial» (una por celda) | ≠ |
| 4.2 | Bootstrap por celda: ¿la diferencia es real o azar? % de celdas con ALADIN más corto/más largo | Distinguir diferencia clara de ruido | 4.1 + remuestreo de años → mapa de significancia y %s | Qué es el bootstrap y el IC95; significativo ≠ grande | ≠ |
| 4.3 | Resumen del sesgo: Δ medio/mediano y sesgo relativo (%) por región y para el dominio | Cuánto acorta ALADIN las rachas | 4.1 → tabla y figura de barras | Diferencia entre días y %; qué conjunto de celdas se usa | ≠ |
| 4.4 | PDFs de duración por región (pool de rachas), Tabla 3 y figura log–log | Ver forma de la distribución y de la cola | Rachas de 3.2 → Tabla 3 + figura (equivalente a Fig. 5) | Qué es un «pool» y por qué se junta; por qué log–log | ≠ |
| 4.5 | Frecuencia de días húmedos por región con R₀ nuevo | Ver cuánto exceso queda dentro de cada región | 2.3 y datos → tabla | Por qué la calibración global no iguala cada región | ≠ |
| 4.6 | Sensibilidades históricas: criterios ii y iii; mínimo de rachas (30/50/100/200); regrillado lineal frente a promedio por área | Mostrar robustez | Pasos previos → Tablas S1–S2, Fig. S1 | Qué cambia con cada alternativa y por qué no se usó otra (p. ej. *quantile mapping*) | ≠ |

## Fase 5 — Proyección (SSP5-8.5, 2040–2074)
| # | Qué hacemos | Para qué | Entrada → salida | Idea que hay que entender | Antes |
|---|---|---|---|---|---|
| 5.1 | Rachas futuras de ALADIN con el **mismo R₀** del histórico; PDFs futuras | Comparar futuro con pasado en el mismo modelo | ALADIN 2040–2074 → rachas y estadísticos | Por qué el futuro se compara con ALADIN histórico y no con CR2MET | ≠ |
| 5.2 | **Risk ratio** RR(D) = N_fut(t ≥ D)/N_hist(t ≥ D), inicio mar–nov, D = 5…60, bootstrap independiente | Cuánto más (o menos) probable es una racha larga | Rachas de 3.2 y 5.1 → Tabla 4 y figura de curvas | Qué es un cociente de riesgos; qué es un IC95 que cruza 1 | ≠ |
| 5.3 | Sensibilidades: 1 mm en ambos y escenario SSP3-7.0 | Ver si la conclusión resiste | 5.2 con otros umbrales/escenario → figuras S | Qué conclusión sobrevive a cada variante | ≠ |

## Fase 6 — ¿Menos días húmedos o más agrupamiento?
| # | Qué hacemos | Para qué | Entrada → salida | Idea que hay que entender | Antes |
|---|---|---|---|---|---|
| 6.1 | Frecuencia húmeda futura frente a histórica con R₀ fijo | Ver si hay menos días húmedos en el futuro | 5.1 y 4.5 → tabla | Por qué menos días húmedos alargan las rachas por sí solo | ≠ |
| 6.2 | Persistencia de 1.er orden ρ y su contraste histórico–futuro | Ver si los días secos se pegan más | Series diarias → Δρ con IC95 | Qué mide ρ y por qué nuestra prueba es poco sensible (±0,03) | ≠ |
| 6.3 | Referencia Markov de 1.er orden: RR esperado solo por frecuencia | Comparar con el RR observado | Series y parámetros → tabla de descomposición | La «moneda de juguete» que recuerda solo ayer | ≠ |

## Fase 7 — Control de calidad y registro
| # | Qué hacemos | Para qué | Entrada → salida | Idea que hay que entender | Antes |
|---|---|---|---|---|---|
| 7.1 | Verificación independiente de conteos y estadísticos con otro código | Descartar errores de implementación | Rachas de 3.2 → diferencia 0 con otro método | Por qué un resultado impreso no es evidencia hasta verificarlo | = |
| 7.2 | `REGISTRO.md` (script → salida → figura/tabla) y hashes de las salidas | Trazabilidad | Todo lo anterior → registro | Qué significa que una cifra «tenga origen» | = |
| 7.3 | Lista de cifras, figuras y tablas finales; **recién entonces** el texto (Introducción v3 y suplemento v3) | Redactar sin inventar | Registro → textos | — | ≠ |

## Decisiones que necesito de ti antes del paso 1
1. **¿Confirmas la `REGLA.md`?** En especial: R₀ se recalibra en el dominio nuevo y se reporta el valor que salga (no se intenta recuperar 5,285 mm).
2. **Las cuatro regiones siguen como subunidades** para PDFs y risk ratios (el dominio es su unión).
3. **¿Con qué paso partimos?** Propongo 0.1 (el dominio y su máscara) y luego 1.1 (interpolación), que es el ejemplo que diste.
4. **Ritmo:** un paso por vez, y entre uno y otro me dices «entendido» o preguntas.

Quedan fuera de este plan, salvo que el profesor lo pida: Markov de orden 2 o 3, cualquier otro dominio, cualquier otro RCM.
