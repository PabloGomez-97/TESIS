# Regla del análisis «Chile central, paso a paso» — VIGENTE desde 2026-10-22 (confirmada por el usuario)

**Estado:** escrita el 2026-10-22 **antes de ejecutar ningún script de esta carpeta** y confirmada por el usuario ese mismo día (con la modificación de la sección «Trabajos anteriores: no se declaran»). Desde ahora no se edita salvo para añadir fechas y errores corregidos (con nota).

## Dominio
La unión de las cuatro regiones administrativas (Natural Earth `admin_1_states_provinces`, 1:10 m): Valparaíso, Metropolitana de Santiago, Libertador General Bernardo O'Higgins y Maule. Una celda de ALADIN pertenece al dominio si su **centro** cae dentro de alguno de los cuatro polígonos. Sin cajas de latitud–longitud.

Subunidades: las mismas cuatro regiones, para las PDFs, la Tabla 3 y los risk ratios.

## Motivo (de alcance, no de resultados)
Alinear el dominio con el proyecto marco («… en Chile central») y con la pregunta del trabajo; aprobado por el profesor guía.
## Qué no cambia
Períodos 1980–2014 y 2040–2074 (SSP5-8.5; SSP3-7.0 solo sensibilidad); CR2MET con 1 mm; rachas censuradas excluidas; mínimo de 30 rachas por celda para estadísticas por celda; bootstrap de años independiente por producto y período (1 000 réplicas, semilla 42); RR(D) con inicio mar–nov y D = 5…60; Markov de 1.er orden (R = 40, semilla 7); contraste de persistencia (semilla 11); interpolación lineal de CR2MET a la grilla de ALADIN.

## Qué cambia
- **R₀ se calibra de nuevo sobre el dominio nuevo** (celdas de las cuatro regiones, 1980–2014), por bisección, para igualar la fracción de días húmedos de CR2MET con 1 mm. Su valor será distinto de 5,285 mm. El valor de 5,285 mm solo se conserva como comparación.
- Los criterios ii (1 mm en ambos) y iii (R₀ local por celda) se mantienen como sensibilidades.
- Los mapas, la tabla de sesgo por criterio y las sensibilidades de dominio completo se rehacen sobre el dominio nuevo.

## Compromisos
1. **Un solo cálculo con este dominio.** No se prueban variantes de dominio ni de unidades. Se reporta lo que salga, incluidas las ausencias de significancia y el valor de R₀ que resulte.
2. Los errores de código se corrigen; **el dominio no se vuelve a elegir** en función de los resultados.
3. **Un paso a la vez:** cada paso se explica, se ejecuta, se revisa con el usuario y se registra antes del siguiente.
4. Cada cifra, tabla y figura queda ligada a un script y a una salida en `REGISTRO.md`; los hashes de las salidas se regeneran.
5. Un conjunto de celdas por comparación, indicado en el pie de cada figura y tabla (celdas comunes salvo que se diga otra cosa).
6. No se edita nada fuera de esta carpeta (ni scripts, ni salidas, ni registros, ni los Docs). No hay commits ni instalación de paquetes.
7. No se escribe texto del paper ni de la tesis hasta terminar los pasos.

## Trabajos anteriores: no se declaran
Por decisión del usuario (2026-10-22), **el paper, la tesis y el suplemento no declaran ni mencionan los trabajos anteriores**: los conjuntos de ventanas y regiones antiguos, la propuesta del 22-oct, el dominio nacional como alternativa, sus resultados y cualquier comparación con ellos. Tampoco se redacta la historia de cómo se llegó al dominio; solo el motivo de alcance aprobado. `CONTEXTO.md` conserva esa historia como documento interno. Esta regla también está en `CLAUDE.md` §9.

Límite que se mantiene: el texto no afirma nada falso sobre el diseño (por ejemplo, que el dominio se fijó antes de conocer cualquier resultado), y las preguntas directas se responden con la verdad.
