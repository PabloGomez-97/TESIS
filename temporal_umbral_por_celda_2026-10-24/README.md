# Experimento temporal: umbral de ALADIN por celda

Esta carpeta repite el análisis del **paper original de la raíz** con CR2MET a 1 mm/día y umbrales de ALADIN calibrados por celda en 1980–2014. Se aplica el mismo campo al futuro 2040–2074. No corresponde a los análisis de 0,1 mm ni de Chile central de otras carpetas.

La referencia global se congeló al iniciar el experimento. Los scripts, datos y resultados originales se conservan.

## Entradas principales

- [Informe comparativo](INFORME_COMPARATIVO.md): resultados, tablas, figuras, diferencias de conclusiones, limitaciones y controles.
- [Comparativas numéricas](comparativa/): CSV completos y figuras de comparación.
- [Resultados recalculados](ejecucion/): misma cadena del paper adaptada al campo local.
- [Referencia global congelada](referencia_global/).
- [Estado de ejecución](estado.json) y [logs](logs/).
- [Cambios de código](cambios_codigo.diff).

Los documentos `.md` de contexto copiados en esta carpeta conservan la descripción del paper original. El informe comparativo y este README describen el experimento local.

## Diferencia adicional explícita

El código local previo limitaba la búsqueda a 15 mm/día y dejaba 82 celdas sin umbral. Aquí se amplía ese límite solo en las celdas que lo necesitan. Los otros 3.505 umbrales se conservan exactamente. Se informa la cobertura y se comparan métricas sobre muestras propias y sobre la intersección de celdas válidas.

## Convenciones de archivos

El bootstrap conserva los criterios `A_opcion_B_integrado` (global), `B_mismo_umbral` (1 mm) y `C_umbral_local` (local). El análisis principal de esta carpeta usa **C**. Algunos nombres de archivo heredados aún incluyen `i_global` o `integrado` para conservar las dependencias de la cadena: sus contenidos principales son locales.

`calibracion_R0.json` conserva el diagnóstico global de referencia; `calibracion_local.json`, `calibracion_local_por_celda.csv` y `R0_local_map.npz` registran el ajuste local realmente aplicado. La figura de calibración principal se adapta al campo local; la global se conserva como referencia.

## Ejecución

Entorno científico existente, sin instalar paquetes:

```sh
~/.pyenv/shims/python3.11 ejecutar_local.py
```

El programa reanuda pasos completos según `estado.json`. Si se cambia un script o una regla, usar una ejecución nueva: la reanudación no invalida automáticamente dependencias.

Para repetir **desde cero**, sin reutilizar cachés del experimento:

```sh
~/.pyenv/shims/python3.11 repetir_desde_cero.py --destino /ruta/a/una/carpeta/nueva
```

Los NetCDF se enlazan desde el repositorio de origen registrado en `configuracion.json`. La referencia global permanece congelada para que el contraste sea el mismo. `preparar.py` conserva la preparación inicial, mientras que `repetir_desde_cero.py` utiliza la versión final de todos los scripts adaptados.
