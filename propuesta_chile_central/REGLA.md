# Regla de la propuesta «Chile central» (escrita antes de calcular)

Fecha y hora de escritura: **2026-10-22 17:3x (hora de Chile)**, antes de ejecutar ningún script de esta carpeta. Estado del repo en ese momento: commit `9f5df39` más cambios sin confirmar de la sesión.

## Estatus
Esta carpeta es una **propuesta alternativa**. Nada fuera de ella se modifica. Si el profesor no acepta el cambio, se elimina la carpeta y el trabajo continúa con el conjunto anterior. Los datos crudos y las cachés del análisis de dominio completo se leen mediante enlaces simbólicos (solo lectura).

## Unidades
Cuatro regiones administrativas (Natural Earth `admin_1_states_provinces`, 1:10 m, ya usadas en el repo), de norte a sur:

1. Valparaíso
2. Región Metropolitana de Santiago
3. Libertador General Bernardo O'Higgins
4. Maule

**Se usan las mismas cuatro unidades para las PDFs/Tabla 3 (antes «ventanas») y para los risk ratios (antes «regiones»).** No hay cajas de latitud–longitud.

## Motivo (de alcance, no de resultados)
- Alinear el trabajo con el foco «Chile central» del proyecto marco y con el dominio del antecedente más cercano (Latoja et al. 2025: 32°–40°S).
- Unidades oficiales, sin trazado a mano, y un solo conjunto para las dos preguntas del trabajo.
- El dominio del análisis de mapas, la calibración de R₀ y la comparación histórica celda a celda siguen siendo los de todo Chile (17,7°–45,9°S) y **no cambian**.

## Qué no cambia (mismas reglas que el análisis anterior)
Períodos 1980–2014 y 2040–2074 (SSP5-8.5; SSP3-7.0 solo sensibilidad de los RR); R₀ = 5,285 mm en ALADIN y 1 mm en CR2MET; rachas censuradas excluidas; bootstrap de años independiente por producto/período (1 000 réplicas, semilla 42); RR = N_fut(t ≥ D)/N_hist(t ≥ D) con inicio marzo–noviembre y D = 5…60 d; Markov de 1.er orden (R = 40, semilla 7) y contraste de persistencia (semilla 11); CR2MET interpolado linealmente a la grilla de ALADIN; sensibilidades de umbral (criterios ii y iii) y de regrillado.

## Compromisos
1. **Un solo cálculo con este conjunto.** No se prueban variantes de unidades. Se reporta lo que salga, incluidas ausencias de significancia.
2. Los errores de código se corrigen; **las unidades no se vuelven a elegir** en función de los resultados.
3. No se edita nada fuera de esta carpeta (ni scripts, ni salidas, ni registros, ni los Docs). No hay commits.
4. Se verifica con código independiente y se audita el texto que dependa de estas cifras antes de proponer cambios al paper.

## Contexto que debe poder decirse con honestidad
El conjunto anterior (ventanas Loa, Maipo, Maule y Biobío; regiones Coquimbo, O'Higgins, La Araucanía y Los Lagos) se propuso en una etapa exploratoria por un asistente de código, a partir de la petición del profesor de «4 cuencas distintas», y sus resultados se conocían cuando se decidió este conjunto. El cambio se decide por alineación de alcance con el proyecto, no por los resultados. El historial de git conserva el conjunto anterior.
