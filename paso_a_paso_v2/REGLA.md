# Regla del análisis «paso a paso v2» — VIGENTE desde 2026-10-23 (confirmada por el usuario)

**Estado:** escrita el 2026-10-23 **antes de ejecutar ningún script de esta carpeta** y confirmada por el usuario ese mismo día, con las decisiones de abajo. Reemplaza a `chile_central_paso_a_paso/REGLA.md` (que queda intacta) tras las instrucciones del profesor guía del 2026-10-23. Solo se edita para añadir fechas y errores corregidos, con nota.

## Dominio
La unión de **siete regiones administrativas** (Natural Earth `admin_1_states_provinces`, 1:10 m), de norte a sur: Valparaíso, Región Metropolitana de Santiago, Libertador General Bernardo O'Higgins, Maule, Ñuble, Bío-Bío y La Araucanía (≈ 32° a ≈ 39,7°S). Una celda de ALADIN pertenece al dominio si su **centro** cae dentro de alguno de los siete polígonos o sobre su borde. Sin cajas de latitud–longitud. (Comprobado el 2026-10-23: el mapa de Natural Earth del repo trae Ñuble, Bío-Bío y La Araucanía como regiones separadas.)

Subunidades: las mismas siete regiones, para las PDFs, la Tabla 3 y los risk ratios (confirmado por el usuario el 2026-10-23).

## Instrucciones del profesor guía (2026-10-23)
1. **Umbral de día húmedo de CR2MET = 0,1 mm.** Un día es húmedo si pr ≥ 0,1 mm y seco si pr < 0,1 mm.
2. **Añadir Ñuble, Biobío y La Araucanía** al dominio y a las subunidades.
3. **Conexión con Stechmann y Neelin (2014):** fracción de tiempo húmedo (Ec. 33) y cutoff de duración de rachas secas t_L (Ec. 12); una menor fracción de tiempo húmedo puede asociarse a un aumento de t_L, aunque no necesariamente. Se tiene presente; **no se calcula nada de esto hasta leer el trabajo y confirmar con el profesor qué se calcularía** (`[VERIFICAR]`; no se citan ecuaciones de memoria).

## Qué no cambia
Períodos 1980–2014 y 2040–2074 (SSP5-8.5; SSP3-7.0 solo sensibilidad); rachas censuradas excluidas; bootstrap de años independiente por producto y período (1 000 réplicas, semilla 42); RR(D) con inicio mar–nov y D = 5…60; Markov de 1.er orden (R = 40, semilla 7); contraste de persistencia (semilla 11); interpolación lineal (bilineal) de CR2MET a la grilla de ALADIN; dia seco si pr < umbral.

## Qué cambia
- **R₀ se calibra de nuevo** sobre el dominio nuevo, por bisección, para igualar la fracción de días húmedos de **CR2MET con 0,1 mm** (1980–2014). Su valor será distinto de los anteriores. Se reporta el que salga.
- **Criterios de umbral** (confirmados): i = CR2MET 0,1 mm y ALADIN R₀ global (principal); ii = **0,1 mm en ambos** (sin calibrar); iii = R₀ local por celda.  El umbral de CR2MET es 0,1 mm en todo el análisis y no se compara con otros umbrales.
- Los mapas, la tabla de sesgo por criterio y las sensibilidades se rehacen sobre el dominio nuevo.

## Compromisos
1. **Un solo cálculo con esta definición.** No se prueban variantes de dominio, de umbral de CR2MET ni de unidades por iniciativa propia; se reporta lo que salga, incluidas las ausencias de significancia y el valor de R₀. El dominio y el umbral solo cambian por **instrucción expresa del profesor**, con fecha y motivo en `CONTEXTO.md`.
2. Los errores de código se corrigen; las unidades y el umbral no se vuelven a elegir según los resultados.
3. **Un paso a la vez:** cada paso se explica, se ejecuta, se revisa con el usuario y se registra antes del siguiente.
4. Cada cifra, tabla y figura queda ligada a un script y a una salida en `REGISTRO.md`; los hashes se regeneran.
5. Un solo conjunto de celdas por comparación, indicado en el pie (celdas comunes salvo que se diga otra cosa).
6. No se edita nada fuera de esta carpeta. No hay commits ni instalación de paquetes.
7. No se escribe texto del paper ni de la tesis hasta terminar los pasos.
8. **Verificar siempre, ahora con 0,1 mm:** la sensibilidad de la fracción de días húmedos a la interpolación (bilineal frente a vecino más cercano) y a valores pequeños de lluvia de CR2MET, porque 0,1 mm está cerca del ruido del producto.

## Trabajos anteriores: no se declaran
Por decisión del usuario (2026-10-22), el paper, la tesis y el suplemento no declaran ni mencionan los trabajos anteriores (conjuntos de ventanas y regiones antiguos, la propuesta del 22-oct, el dominio nacional, el dominio de cuatro regiones, sus resultados ni comparaciones con ellos). No se redacta la historia de cómo se llegó al dominio ni al umbral; solo el motivo aprobado por el profesor. `CONTEXTO.md` conserva esa historia como documento interno. Límite que se mantiene: el texto no afirma nada falso sobre el diseño, y las preguntas directas se responden con la verdad.


## Notas de actualización
- 2026-10-23: por decisión del usuario se **elimina el mínimo de rachas por celda** (antes: 30). Motivo: ninguna celda del dominio tiene menos de 549 rachas completas, así que el mínimo no actuaba y solo generaba preguntas. Los estadísticos por celda usan todas sus rachas completas; se informa el número de rachas por celda. Se rehicieron los pasos 3.3 y 4.1.
