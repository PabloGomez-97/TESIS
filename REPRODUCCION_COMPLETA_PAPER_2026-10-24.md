# Reproducción completa del paper desde NetCDF

**Fecha:** 24 de octubre de 2026, hora de Chile.  
**Resultado:** reproducción completa satisfactoria de la cadena científica vigente de la raíz.

## Resultado comprobado

Se ejecutaron **23 comandos correspondientes a los 21 scripts científicos**, incluidas las dos variantes adicionales de risk ratios. Todos terminaron con código de salida 0.

Las **68 salidas** regeneradas coinciden con las originales:

| Formato | Archivos | Resultado |
|---|---:|---|
| NPZ | 9 | Idénticos por SHA-256; incluye los seis bootstrap por celda y la caché de regrillado |
| CSV | 28 | Idénticos por SHA-256 |
| Pickle | 2 | Idénticos por SHA-256; todas las duraciones principales y de sensibilidad |
| JSON | 6 | Idénticos por SHA-256 |
| PNG | 23 | 22 idénticos por SHA-256; el restante tiene todos los píxeles RGBA idénticos |
| **Total** | **68** | **Sin diferencias en valores ni píxeles** |

El PNG con diferente hash es `paper_figuras/figT01_referencia_sin_memoria.png`, una figura auxiliar que no aparece en los dos Docs del paper. Su única diferencia es la etiqueta de software: el original declara Matplotlib 3.10.9 y el regenerado 3.10.7. Dimensiones, DPI, modo RGBA y todos los píxeles coinciden exactamente. No se utilizó tolerancia visual para aceptar esta imagen.

También pasaron **90 controles cruzados** sobre las salidas nuevas, con una implementación separada para recalcular los RR y sus intervalos desde conteos anuales.

## Cómo se hizo

Directorio de evidencia:

[TESIS_reproducciones/2026-10-24_001214](/Users/pablogomez/Documents/TESIS_reproducciones/2026-10-24_001214)

- Se creó `ejecucion/` sin salidas, pickles, NPZ ni otras cachés de análisis.
- Se copiaron los 21 scripts sin cambios.
- Solo se enlazaron los directorios de entrada `pr/` y `pr1/`.
- Se congelaron las 68 salidas existentes en `referencia/` antes de empezar.
- Se recalculó la interpolación de CR2MET desde los NetCDF mensuales, además de las cachés de todos los demás análisis.
- Se conservaron semillas, períodos, umbrales, número de réplicas y reglas de censura originales.
- Se ejecutó la sensibilidad de PDFs antes de Tabla 3, corrigiendo el orden inválido de la documentación anterior.
- Se registraron versiones, código, datos, geometrías, comandos, tiempos, logs y hashes.

Tiempo total de la ejecución auditada: **18 minutos y 36 segundos**, incluida preparación, comprobación de entradas y comparación final. El bootstrap de los tres criterios tomó aproximadamente 9 minutos y 51 segundos.

## Cadena regenerada

1. Calibración global de R₀ y Figura 3.
2. Mapas y bootstrap independiente por años de media y t99 para los tres criterios: 1 000 réplicas por celda.
3. Resúmenes espaciales, sesgos por ventana y Tabla S1.
4. Risk ratios principales y sensibilidades a 1 mm y SSP3-7.0, con todos sus intervalos y conteos de réplicas descartadas.
5. Simulaciones Markov y contraste bootstrap de persistencia de primer orden.
6. Duraciones y PDFs de las cuatro ventanas, históricas y futuras.
7. Sensibilidad histórica de PDFs, Tabla 3 y sesgos relativos.
8. Frecuencia húmeda, estacionalidad de precipitación y campo de umbrales locales.
9. CR2MET interpolado y promediado por centros de celda; sensibilidad al regrillado.
10. Diagnóstico de rachas censuradas.
11. Todas las figuras de la cadena, incluida la referencia T01 auxiliar.
12. Sensibilidad al mínimo de rachas por celda.

## Integridad y controles

| Comprobación | Resultado |
|---|---|
| NetCDF contra el registro de datos crudos | **680/680** hashes coinciden: 672 CR2MET y 8 ALADIN |
| Continuidad temporal de cada archivo | Series diarias continuas |
| Continuidad entre tramos de cada producto/escenario | Sin huecos ni solapamientos |
| NetCDF antes y después de ejecutar | Sin cambios |
| Los 21 scripts científicos originales | Sin cambios |
| Las 68 salidas originales | Sin cambios |
| Los 30 archivos cartográficos registrados | Sin cambios |
| Tabla 3 frente a duraciones nuevas | Coincide a la precisión de sus CSV |
| PDFs principales frente a sensibilidad histórica | Duraciones idénticas para las series comunes |
| RR e IC95 calculados por conteos anuales | Coinciden en los 120 puntos: 10 duraciones × 4 regiones × 3 análisis |
| Máscaras comunes, mapas y diferencias | Coherentes; 3 526 celdas en el análisis principal |
| Significancia y resúmenes de los seis bootstrap | Coherentes con sus intervalos y máscaras |
| Conteos de Markov frente a los RR | Coinciden |
| Diagnóstico de censura frente a las PDFs | Coincide el número de rachas completas |

La comparación de reproducción fue exacta. Las pequeñas tolerancias usadas en algunos de los 90 controles cruzados únicamente permiten comparar los estadísticos sin redondear con CSV/JSON redondeados, o mapas guardados en float32 con diferencias calculadas en float64. No se usaron para ocultar diferencias entre la reproducción y los originales.

## Evidencia verificable

- [Resumen automático](/Users/pablogomez/Documents/TESIS_reproducciones/2026-10-24_001214/resumen.json).
- [Comparación de los 68 archivos](/Users/pablogomez/Documents/TESIS_reproducciones/2026-10-24_001214/comparacion_salidas.json).
- [Los 90 controles de consistencia](/Users/pablogomez/Documents/TESIS_reproducciones/2026-10-24_001214/controles_consistencia.json).
- [Estado, comandos y duración de cada paso](/Users/pablogomez/Documents/TESIS_reproducciones/2026-10-24_001214/estado.json).
- [Inventario y continuidad de los NetCDF](/Users/pablogomez/Documents/TESIS_reproducciones/2026-10-24_001214/inventario_netcdf.json).
- [Versiones del entorno](/Users/pablogomez/Documents/TESIS_reproducciones/2026-10-24_001214/entorno.json).
- [Hashes de los scripts ejecutados](/Users/pablogomez/Documents/TESIS_reproducciones/2026-10-24_001214/scripts_sha256.json).

La carpeta también conserva todos los logs, las salidas nuevas, la referencia congelada, `pip_freeze.txt` y los manifiestos de entradas y geometrías antes y después de ejecutar.

## Alcance y pendientes

**Queda confirmada la reproducibilidad computacional de los resultados vigentes desde los NetCDF locales de entrada, sin reutilizar cachés de análisis.** Esto incluye las etapas que quedaron pendientes en la revisión del 23 de octubre: interpolación completa de CR2MET, bootstrap de mapas y simulaciones Markov.

No equivale a validar toda elección metodológica, reconstruir CR2MET desde sus estaciones originales, comprobar citas bibliográficas o resolver la procedencia externa de ALADIN. Las geometrías Natural Earth existentes se trataron como entradas identificadas por hash.

**Actualización posterior del mismo día:** la reproducción completa descrita en este informe conservó la aproximación original de meses de 30,4375 días. Posteriormente se corrigió `_calcular_estacionalidad_marnov_regiones.py` para sumar el calendario real y se regeneró únicamente su CSV. La corrección se comprobó con una segunda ejecución aislada, sin sobrescribir la evidencia de esta reproducción. Véase `REGISTRO_RESULTADOS.md`, §2, y la evidencia incremental en `~/Documents/TESIS_reproducciones/correccion_precipitacion_2026-10-24_003845/`. Las cifras corregidas deben reflejarse en el paper.

Tampoco se modificaron los Docs ni se reemplazaron sus imágenes. La diferencia de versión de la Figura 1 en el Doc y los ajustes editoriales del informe anterior son asuntos distintos de esta comprobación.

## Cómo repetirlo

El procedimiento y los programas quedan en [reproducibilidad/README.md](/Users/pablogomez/Documents/GitHub/TESIS/reproducibilidad/README.md).

```sh
~/.pyenv/shims/python3.11 /Users/pablogomez/Documents/GitHub/TESIS/reproducibilidad/ejecutar.py
```

Cada ejecución crea un directorio nuevo fuera del repositorio. No sustituye las salidas de trabajo.
