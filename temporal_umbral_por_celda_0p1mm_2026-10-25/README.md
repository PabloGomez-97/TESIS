# Umbral local por celda: CR2MET a 0,1 mm/día

Experimento separado del 25 de octubre de 2026. Conserva todos los análisis anteriores y repite la cadena completa del paper original con CR2MET a 0,1 mm/día, recalibrando ALADIN por celda en 1980–2014. Cada umbral histórico permanece fijo en 2040–2074, bajo SSP5-8.5 y SSP3-7.0.

La comparación principal es **local con referencia 1 mm frente a local con referencia 0,1 mm**. No es global frente a local, ni una división por diez de los umbrales previos. Al cambiar la referencia cambian las rachas de CR2MET y las de ALADIN: los errores se interpretan respecto de su propia definición.

- [Informe detallado](INFORME_COMPARATIVO_LOCAL_0p1mm_VS_1mm.md).
- [Dieciocho CSV y cinco figuras comparativas](comparativa/).
- [Resultados y scripts nuevos](ejecucion/).
- [Referencia local anterior congelada](referencia_local_1mm/).
- [Controles globales anteriores de 0,1 mm](referencia_global_0p1mm/).
- [Estado](estado.json), [logs](logs/) y [diff de código](cambios_codigo.diff).

Se mantienen períodos, datos, máscaras, ventanas, regiones, semillas, 1 000 réplicas de bootstrap, reglas de censura, mínimo de rachas y selección de inicio marzo–noviembre para los RR. Se conserva la bisección local con ampliación del límite de 15 mm/día únicamente cuando sea necesario.

Los criterios A/B/C de las sensibilidades conservan sus identificadores: A es global auxiliar a 3,668 mm; B aplica 0,1 mm en ambos; C es el análisis local principal. Algunos nombres de archivo heredados contienen `i_global` o `integrado` por compatibilidad con la cadena; sus salidas principales contienen el resultado local. El campo aplicado se registra en `calibracion_local.json`, `calibracion_local_por_celda.csv` y `R0_local_map.npz`.

## Ejecución

```sh
~/.pyenv/shims/python3.11 ejecutar.py
```

Reanuda pasos completos registrados. Si se modifica una regla o un script, usar una ejecución nueva: no invalida dependencias automáticamente.

Para repetir desde cero en otra carpeta nueva, sin cachés de análisis:

```sh
~/.pyenv/shims/python3.11 repetir_desde_cero.py --destino /ruta/nueva
```

Los NetCDF se enlazan desde el repositorio de origen de `configuracion.json`. Las referencias anteriores se mantienen congeladas y se comprueban por SHA-256, junto con los 680 NetCDF. No se instalan paquetes ni se modifican los Docs del paper.

El informe incluye además una comparación de las cuatro configuraciones (global/local × 1/0,1 mm) para separar ambas decisiones metodológicas.

Se corrigió una diferencia de precisión de la caché auxiliar cerca de 0,1 mm; la primera ejecución se conserva en `revision_precision_float32/` y la explicación en [NOTA_PRECISION.md](NOTA_PRECISION.md). Los resultados principales no cambiaron por esa corrección.
