# Cambios MAYOR propuestos al paper (2026-10-22, aún NO aplicados en el Doc)

Cada fila es una sugerencia de edición para el Doc del paper. «Origen» remite a `RESUMEN_PAPER_PARA_TESIS.md` §5.

| # | Dónde | Texto actual (fragmento) | Propuesta | Origen / respaldo |
|---|---|---|---|---|
| 1a | Intro, último párrafo | «Así abordamos no solo cuánto cambia la precipitación, sino cómo cambia…» | «Así abordamos cómo cambia…» | El paper no analiza el cambio de acumulado |
| 1b | §3.2, primera frase | «…rachas prolongadas —no solo el acumulado anual.» | «…rachas prolongadas.» | Idem |
| 1c | Conclusión (último párrafo) | «…no consiste únicamente en una reducción de la precipitación acumulada. El modelo proyecta…» | «Los resultados muestran que, hacia 2040–2074 y bajo SSP5-8.5, el modelo proyecta…» | Idem |
| 1d | Conclusión, última frase | «…dimensión del cambio hidroclimático que queda parcialmente oculta en la precipitación media y en índices…» | «Caracterizar la distribución de duraciones aporta una dimensión que un índice basado en un único extremo anual no recoge.» | Idem |
| 1e | §4.2, párrafo de gestión hídrica | «…y no se deduce del cambio en la precipitación total. Un aumento… condicionan el déficit de humedad del suelo, la recarga y la planificación del riego.» | Eliminar ambas oraciones; queda «el resultado central es que el cambio proyectado se concentra en las rachas largas.» | Sin resultado propio ni cita |
| 2a | Cierre de §3.2.2 (texto sugerido por Cristian Martínez en un comentario) | «El cambio relativo es pequeño en Loa, donde las rachas ya son prolongadas en el histórico, mientras…» | «El cambio relativo es pequeño en Loa, mientras…» | No hay resultado que sostenga la causa |
| 2b | §3.2.2, párrafo 1 | «en Loa el cambio relativo es menor porque las rachas ya son prolongadas» | «en Loa el cambio relativo de la media y del p99 es menor» | Idem (Tabla 3) |
| 2c | Discusión §4.1 | «…10–13 % en Maipo, Maule y Biobío, y el aumento…» | «…Biobío (en Loa no disminuye: +1,4 %, y el contraste de persistencia no se calculó para esa ventana), y el aumento…» | `frecuencia_humeda_ventanas.csv` (2,965 → 3,006 %) |
| 3 | Discusión §4.1 | «un modelo de Markov de primer orden que solo cambia la frecuencia reproduce el RR observado dentro de su IC95 en los 31 casos con N_hist ≥ 100» | «el RR que predice un modelo de Markov de primer orden que solo cambia la frecuencia cae dentro del IC95 del RR observado en los 31 casos con N_hist ≥ 100 (los IC son amplios, y a D = 20 ese modelo reproduce solo el 70 % y el 89 % de las rachas históricas de Los Lagos y La Araucanía)» | `descomposicion_markov_rr_regiones.csv` (cobertura 0,704 y 0,895) |
| 4 | Discusión, párrafo del forzante | «…, mientras que bajo SSP5-8.5 domina la respuesta forzada.» | «. Con un solo miembro, aquí no se separa la señal forzada de la variabilidad interna.» | Sin cita ni resultado |
| 5 | §4.2 | «La metodología, basada en PDFs…, es transferible a otros RCMs del dominio CHP.» | Eliminar (la extensión a otros RCMs ya está en «trabajo futuro») | No se probó |
| 6 | Discusión §4.1, párrafo 1, final | «Aunque esa relación se obtuvo con modelos globales… podría ser conservador [VERIFICAR…], especialmente en el norte árido, donde el acortamiento de la cola histórica es mayor.» | Eliminar | Solo se leyó el resumen de Petrova; además «mayor en el norte» es cierto en días pero no en porcentaje (p99: Loa −21,1 %, Maule −31,3 %) |
| 8a | Discusión §4.1 | «…del de la organización temporal.» | «…del de la persistencia día a día.» | Solo se midió persistencia de 1.er orden |
| 8b | Discusión §4.1 | «…cambio en el agrupamiento de los días secos, aunque…» | «…cambio en la persistencia día a día de primer orden, aunque…» | Idem |
| 8c | Intro, párrafo 12 | «…pero no la reorganización día a día de las rachas (Báez-Villanueva…)» | «…pero no la estructura día a día de las rachas…» | Plan de `CAMBIOS_AL_PAPER.md` §1 |

**No incluido (requiere leer las fuentes o decidir):** citas ausentes de la lista (Stechmann y Neelin 2014, Sun 2006, Chen 2021), Polade 2014/2017 mal atribuidas, Leung con autores truncados, y la Tabla S6 del suplemento (hoy `[PENDIENTE]` en la Discusión).
