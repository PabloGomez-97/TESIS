# Corrección de precisión de los diagnósticos auxiliares

El control de censura encontró una racha completa adicional de CR2MET en Biobío (357 068 frente a 357 067) al clasificar la caché interpolada en float32 a 0,1 mm. La interpolación principal se clasifica sin ese redondeo.

Se conserva la primera ejecución auxiliar en `revision_precision_float32/`. Se regenera la caché de regrillado en float64, sin cambiar pesos, fechas, máscara, unidades ni algoritmos de rachas. Los campos locales auxiliares se almacenan en float32 como el campo principal. Se verifican el campo local y las métricas lineales frente al análisis principal. Se repiten regrillado, censura y figura T01.

La calibración principal, los bootstrap por celda, PDFs principales, Tabla 3, RR y diagnósticos de persistencia/Markov ya usaban la clasificación principal y no requieren cambios por esta corrección. Todos los análisis anteriores a esta carpeta se conservan intactos.
