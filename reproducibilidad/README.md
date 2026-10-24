# Reproducción del paper desde datos crudos

Desde la raíz del repositorio:

```sh
~/.pyenv/shims/python3.11 reproducibilidad/ejecutar.py
```

El programa crea una carpeta nueva en `~/Documents/TESIS_reproducciones/`.
Se puede elegir otro directorio nuevo fuera del repositorio con `--destino /ruta/nueva`.
No instala paquetes ni modifica datos, scripts científicos o resultados de la raíz.

## Qué ejecuta

- Copia los 21 scripts científicos de la raíz, sin modificarlos.
- Conserva una copia de referencia de todos los CSV, JSON, NPZ, pickle y PNG de la cadena.
- Enlaza únicamente las carpetas de datos NetCDF `pr/` y `pr1/`.
- Empieza sin cachés ni salidas de análisis en `ejecucion/`.
- Verifica los hashes de los 680 NetCDF contra el registro de datos crudos.
- Ejecuta 23 comandos: los 21 scripts y las dos variantes adicionales del RR.
- Regenera la interpolación de CR2MET, calibración, bootstrap por celda de los tres criterios,
  RR de los tres análisis, simulaciones Markov, sensibilidades, tablas y figuras.
- Compara todas las salidas con la referencia: valores exactos y posiciones de NaN,
  claves/tipos de arreglos y píxeles RGBA exactos en las imágenes. No usa tolerancias para
  decidir si una salida se reprodujo.
- Verifica al terminar que los datos, scripts, geometrías y salidas originales no cambiaron.

La Tabla 3 se ejecuta después de la sensibilidad de PDFs: necesita ambos pickles.
El proceso incluye la figura T01 auxiliar, aunque no está en los Docs del paper.
Las geometrías de Natural Earth ya disponibles son entradas cartográficas: se registran sus
hashes. No son cachés de análisis meteorológico.

## Evidencia guardada

- `estado.json`: estado y tiempos de cada comando.
- `logs/`: salida completa y errores de cada script.
- `scripts_sha256.json`: código científico usado.
- `datos_sha256_antes.json`, `datos_sha256_despues.json`: integridad de entradas.
- `entorno.json`, `pip_freeze.txt`: Python, sistema y paquetes.
- `geometrias_sha256_*.json`: insumos cartográficos.
- `referencia/`: resultados originales congelados.
- `ejecucion/`: resultados calculados desde cero.
- `comparacion_salidas.json`, `resumen.json`: comparación y dictamen automático.

Un código de salida 0 indica reproducción exacta e integridad conservada. El código 2 indica
diferencias; un fallo de script detiene la cadena y queda registrado. No se ocultan diferencias
numéricas con redondeos o tolerancias.

## Controles cruzados

Para registrar unidades, atributos, calendarios y continuidad de los NetCDF:

```sh
~/.pyenv/shims/python3.11 reproducibilidad/inventariar_datos.py \
  --repo /ruta/al/repositorio \
  --salida /ruta/de/la/reproduccion/inventario_netcdf.json
```

Después de una ejecución completa:

```sh
~/.pyenv/shims/python3.11 reproducibilidad/verificar_consistencia.py \
  --ejecucion /ruta/de/la/reproduccion/ejecucion \
  --salida /ruta/de/la/reproduccion/controles_consistencia.json
```

Este segundo programa contrasta relaciones entre salidas regeneradas: Tabla 3 y duraciones,
PDF principal y sensibilidad, conteos regionales, RR e intervalos bootstrap recalculados con
conteos por año, mapas y máscaras comunes, significancia y sus resúmenes, conteos Markov y
diagnóstico de censura. Las tolerancias de estos controles corresponden al redondeo publicado
en CSV/JSON o a la precisión de almacenamiento de los mapas; no sustituyen la comparación
exacta de la reproducción.

## Alcance científico

Reproducir los resultados demuestra que salen del código y los datos identificados.
No demuestra por sí solo que todos los métodos sean correctos ni valida referencias,
procedencia externa o incertidumbre multimodelo.

La reproducción completa del 24 de octubre se hizo antes de corregir la aproximación de
estacionalidad con meses de 30,4375 días. Después se actualizó ese script para sumar los días
reales de cada mes y se regeneró su CSV, con una comprobación incremental aislada. Una nueva
ejecución de este procedimiento usa el script corregido y compara contra el CSV actualizado.
La evidencia de la reproducción anterior se conserva sin sobrescribirla.
