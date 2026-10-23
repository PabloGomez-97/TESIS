# Auditoría paper ↔ scripts ↔ salidas ↔ figuras ↔ tablas

Fecha: 2026-10-22. Texto auditado: Doc del paper (revisión posterior a las sugerencias MAYOR) y Doc del Material suplementario. Todo lo marcado «✓» se comparó hoy contra el archivo de salida citado; «no verificado» significa que no se hizo.

## 1. Alcance

| Qué | Cómo | Estado |
|---|---|---|
| Código | Lectura completa de `_run_pregunta5_aladin.py`, `_run_pregunta7_bootstrap.py`, `_generar_pregunta9_pdf_loglog_cuencas.py`, `_calcular_tabla3_ventanas.py`, `_calcular_frecuencia_humeda_ventanas.py`, `_calcular_sesgo_ventanas.py`, `_contraste_persistencia_regiones.py`, `_referencia_markov_rr_regiones.py`, `_cuantificar_censuradas.py`, `_calcular_estacionalidad_marnov_regiones.py`, `_generar_fig03_calibracion_R0.py`, y las partes de dibujo de `_generar_fig05…`, `_generar_fig_rr_regiones.py`, `_generar_fig_pdfs_ventanas.py` | Hecho |
| No leído | `_test_regrillado_conservativo.py`, `_generar_figS1…`, `_generar_figS2…`, `_generar_figT01…`, `_generar_fig02…`, `_generar_fig_dominio…`, `_generar_tablaS1…` | No verificado (sus salidas sí se compararon con las tablas) |
| Ejecución | No se re-ejecutó ningún script. Sí se recalcularon desde datos crudos o cachés: calendario y unidades de ALADIN, NaN de CR2MET interpolado, bins por ventana, estadísticos S2, frecuencia húmeda en Maipo y Maule, R₀ local, tabla de censuradas, celdas excluidas | Hecho |
| Tablas | Tablas 1–4 del paper y S1–S6 del suplemento, contra CSV/JSON/pickles | Hecho |
| Figuras | Vistas: Fig. 1, 3, 4, 5 (nueva y vieja), 6, S1, S2. No vistas: Fig. 2 (esquema), S3, S4, T01 | Parcial |
| Cifras del texto | Cada cifra de §2, §3, §4 y suplemento rastreada a su salida (§4 de este archivo) | Hecho salvo la Introducción (literatura) |
| Citas | No se leyó ninguna fuente nueva | No verificado |

## 2. Lo que sí coincide (resumen)

- **Tabla 3: las 12 filas × 8 columnas** coinciden con `tabla3_estadisticos_ventanas.csv` y con las duraciones del pickle.
- **Tabla 4 y los RR por D citados en §3.2.1:** coinciden con `outputs_pregunta5_aladin/results.json`. La salida independiente `descomposicion_markov_rr_regiones.csv` reproduce los mismos N_hist y N_fut (misma implementación distinta: bucle vectorizado).
- **Tabla S1** coincide con `bootstrap_summary.csv`, `resumen_pregunta7.csv` y `tablaS1_sensibilidad_umbral_chile.csv`. Tabla S2: coincide con el pickle de sensibilidad (n, media, mediana, p90, p99, máx. de 16 filas). S3 y S5: coinciden con los `results.json` de 1 mm y SSP3-7.0. S4: coincide con `mapas_lineal_vs_area.csv`, `r0_lineal_vs_area.json` y `cuencas_cr2met_lineal_vs_area.csv`. S6 (sugerencia pendiente): coincide con `contraste_persistencia_regiones.csv` en las 12 celdas de la tabla.
- **Calibración:** F = 20,94 / 30,68 %, +9,7 pp, R₀ = 5,285 mm. Fig. 3 las muestra tal cual.
- **Bins de las PDFs:** 23, 22, 21 y 21 bins efectivos (Loa, Maipo, Maule, Biobío), recalculados con la función del script.
- **Número de celdas:** 3 587; ventanas 298/154/171/243; regiones 239/110/223/324; 3 526 comunes; 61 excluidas entre 18,4°S y 22,5°S.
- **Censuradas:** < 1,1 % en Loa y < 0,2 % en el resto (máximos 1,07 % y 0,19 %); 22–40 sobre el p99 en Loa.
- **Estacionalidad:** 85,0–97,3 % (CR2MET) y 81,4–95,8 % (ALADIN) de la precipitación en mar–nov.
- **Supuestos de los datos descartados como riesgo:** el calendario de ALADIN es estándar (hist 13 149 días en 1979–2014; futuro 9 132 y 13 148), pr está en kg m⁻² s⁻¹, y CR2MET interpolado a las 3 587 celdas **no tiene NaN** ni valores negativos. Las dos cosas que habrían alterado F y R₀ (NaN contados como secos y calendario con 29 de febrero distinto) no ocurren.

## 3. Hallazgos

### MAYOR

**A1. La Figura 5 del Doc es la figura antigua (resuelto el 2026-10-22: se decidió conservarla; `fig_pdfs_ventanas.png` y su script se eliminaron, respaldo en `~/Documents/respaldo_TESIS_fig_pdfs_ventanas_2026-10-22.zip`).** El Doc tiene 6 imágenes; las razones de aspecto coinciden con `fig42_pdf_cuencas_loglog_panel.png` (1,521; Doc 1,519), no con `fig_pdfs_ventanas.png` (1,31). La figura vieja no tiene letras (a)–(d) (el script define `labels` pero no los dibuja), no marca el p99 ni los bins con < 10 rachas, usa títulos en inglés/sin tildes («Duracion», «ssp585», «5.285») y una leyenda con n. `REGISTRO_RESULTADOS.md` dice que `fig_pdfs_ventanas.png` la reemplaza. Hay que sustituir la imagen y actualizar el pie (explicar círculos huecos y triángulos). *No pude abrir la imagen del Doc; la conclusión se basa en la razón de aspecto. Verifica a ojo.*

**A2. «ALADIN produce entre 2 y 4 veces la precipitación anual de CR2MET» es falso.** Con `estacionalidad_marnov_regiones.csv`: Coquimbo 4,41; O'Higgins 2,41; La Araucanía 1,77; Los Lagos 1,84. El rango es **1,8–4,4**. Está en §4.3 (límite 6) y en `CLAUDE.md` §8. Corregir a «entre 1,8 y 4,4 veces».

**A3. «Δ negativo de manera generalizada» (§3.1.2) contradice la Figura 4.** El panel (c) y (f) muestran una franja costera del norte (≈ 18–24°S) con Δ positivo de hasta +100 d (media) y +400 d (t99), y `bootstrap_summary.csv` da 7,8 % de las celdas (media) y 3,7 % (t99) con ALADIN significativamente **más largo**. Además la media de Δ (−8,2 d) es arrastrada por el norte: el Δ mediano es −2,2 d. Reescribir: «Δ es negativo en la mayor parte del dominio, con un núcleo de sesgos fuertes entre ~24° y 32°S y un sector costero del norte donde ALADIN es más largo».

**A4. El percentil 99 por celda con ≥ 30 rachas es casi el máximo.** `MIN_SPELLS_FOR_STATS = 30`: con 30–100 rachas el p99 (interpolación lineal) depende de 1 o 2 valores. Esto afecta los mapas de t99, Δ t99 y su significancia, justo donde hay menos rachas (norte). El umbral de 30 no está justificado en el paper. Opciones: elevar el mínimo para t99 (p. ej. ≥ 100, con script y regeneración de la Fig. 4 y Tabla S1), o declarar la limitación. *No cuantifiqué cuántas celdas tienen < 100 rachas (la caché no guarda conteos).*

**A5. Censuradas en el futuro de ALADIN (no declarado).** `rachas_censuradas_por_cuenca.csv`: rachas censuradas que superan el p99 de las completas: Loa 27 (CR2MET), 40 (ALADIN hist), 22 (ALADIN fut); **ALADIN futuro: Maipo 12, Maule 39, Biobío 21; histórico y CR2MET: 0**. Es decir, la cola futura queda más truncada que la histórica en Maule, Biobío y Maipo, y eso afecta Tabla 3 (p99 y máx. futuros) y los RR a D grandes. El texto menciona solo Loa. Declararlo junto a la conclusión de §3.2.2: la exclusión sesga el cambio **a la baja**.

### MENOR (cifras y formato)

- **M1.** «4,2 pp» (Maipo) es doble redondeo: el valor es 20,274 − 16,125 = **4,1497** (recalculado hoy). Escribir «≈ 4 pp» o «4,15». Aparece en §2.3 y §3.1.3. Maule 3,30 ✓.
- **M2.** §2.7: «Tablas S1–S5, Figuras S1–S4»; el suplemento ya tiene S1–S6 y secciones S1–S7. Alinear.
- **M3.** Pie de la Fig. S1 (suplemento): «en Loa, Maule y Biobío la cola de ALADIN sigue terminando antes que la de CR2MET» es falso para Biobío (máx. ALADIN 153 d frente a 109 d en CR2MET; la figura lo muestra). Vale para el p99 (35 frente a 44) y para Loa y Maule.
- **M4.** Pie de la Fig. 5: «subestima la cola en las cuatro ventanas (Tabla 3)» es cierto para el p99; en Biobío el máximo de ALADIN es mayor. Escribir «p99».
- **M5.** Fig. 1: razón de aspecto en el Doc 0,450 frente a 0,421 del archivo (≈ 7 % de estiramiento horizontal). Verificar a ojo.
- **M6.** Fig. 1: las ventanas Maipo y Biobío se solapan con las regiones O'Higgins y La Araucanía. §4.3 y el cierre de §3.2.2 hablan de «Chile central y centro-sur» combinando PDFs (ventanas) y RR (regiones), que son unidades distintas; aclararlo.
- **M7.** La caída de 6 % de días húmedos citada en la Discusión es la del dominio anual. Las regiones de los RR, en mar–nov (derivado de `contraste_persistencia_regiones.csv`, 1 − frecuencia seca): Coquimbo −15,1 %, O'Higgins −12,3 %, La Araucanía −8,7 %, Los Lagos −4,4 %. Decir cuál se usa; el de las regiones es el pertinente para los RR.
- **M8.** Bootstrap por celda (`_run_pregunta7_bootstrap.py`): las réplicas con < 30 rachas dan NaN y se descartan; si quedan < 50 réplicas, la celda se declara «no significativa» y sigue en el denominador. Eso no está en §2.5. Efecto: subestima el % de celdas significativas en el norte.
- **M9.** `REGISTRO_RESULTADOS.md` §4 usa la numeración antigua de figuras (Fig. 5 = mapas, Fig. 6 = PDFs, Fig. 9 = RR); el Doc usa 4 = mapas, 5 = PDFs, 6 = RR. Actualizar el registro.
- **M10.** Tabla S6, Los Lagos: Δρ = +0,021 con IC95 [−0,002; 0,044]; incluye 0 por 0,002. La conclusión «no cambia de forma detectable» es correcta; «no cambia» no.
- **M11.** Supuesto de «fracción de celdas significativas» sin corrección por multiplicidad ni por dependencia espacial; el paper no afirma significancia de campo, pero el 66 % / 43 % se leerá así.
- **M12.** `_run_pregunta5_aladin.py` aún escribe `fig08_rr_curves_aladin.png` (figura vieja, no la del Doc) y `fig42_…` sigue en `paper_figuras/`; mantener dos versiones de la misma figura invita a usar la equivocada. Renombrar o mover las obsoletas (con respaldo, según `CLAUDE.md` §6).

## 4. Matriz de cifras del texto

| Cifra en el texto | Dónde | Fuente | Estado |
|---|---|---|---|
| 3 587 celdas; 17,7–45,9°S | §2.1, Fig. 1 | `calibracion_R0.json`, caché de mapas | ✓ |
| Ventanas 298/154/171/243; regiones 239/110/223/324 | Fig. 1 | `sesgo_por_ventana.csv`, `contraste_persistencia_regiones.csv` | ✓ |
| F = 20,94 / 30,68 %; +9,7 pp; R₀ = 5,285 | §2.3, Fig. 3 | `calibracion_R0.json` | ✓ |
| < 1,1 % / < 0,2 % censuradas | §2.2 | `rachas_censuradas_por_cuenca.csv` | ✓ |
| 21–23 bins | §2.4 | recalculado | ✓ |
| 3 526 celdas comunes; 3 549 / 3 550 | §2.5, Fig. 4 | caché de mapas | ✓ |
| 85–97 % y 81–96 % | §2.6 | `estacionalidad_marnov_regiones.csv` | ✓ |
| D = 20 ≈ p90 | §2.6 | Tabla 3 (p90 = 14–23 d) | ✓ con salvedad escrita |
| «4,2 y 3,3 pp» | §2.3, §3.1.3 | `frecuencia_humeda_ventanas.csv` | **M1** (4,15) y 3,30 ✓ |
| Sesgo relativo −3,5 / −23 / −14 / −13 % (R₀ local) | §2.3, §3.1.3 | `sesgo_relativo_criterios_ventanas.csv` | ✓ |
| Loa 65,6 / 456 / 1 400; Biobío 6,2 / 44; Maipo 11,0 / 119; Maule 9,5 / 83 | §3.1.1 | Tabla 3 | ✓ |
| 41,0 / 32,8 d; 215,8 / 169,6 d; Δ −8,2 y −46,3 | §3.1.2 | `sesgo_por_ventana.csv`, `bootstrap_summary.csv` | ✓ |
| 66 % y 43 % | §3.1.2 | `bootstrap_summary.csv` (66,3 / 43,1) | ✓ |
| −27 d y −135 d (1 mm) | §3.1.2 | `bootstrap_summary.csv` | ✓ |
| −21/−22 %; −27/−31 %; −17/−21 %; −15/−23 % | §3.1.3 | `tabla3_estadisticos_ventanas.csv` | ✓ |
| 66 267 frente a 55 015; 1 419 frente a 1 156 | §3.1.3 | Tabla 3 | ✓ |
| «Δ negativo generalizado» | §3.1.2 | Fig. 4, `bootstrap_summary.csv` | **A3** |
| Tabla 4 (4 filas) | §3.2.1 | `results.json` | ✓ |
| RR(5) = 0,94; RR(30) = 1,26 [1,10; 1,43]; límite 1,0075; D = 25 cruza 1 | §3.2.1 | `descomposicion_markov_rr_regiones.csv` | ✓ |
| La Araucanía 3,47 [1,29; 11,8] (D = 30) | §3.2.1 | idem | ✓ |
| O'Higgins D = 60: 1,49 [1,01; 2,22] | §3.2.1 | idem | ✓ |
| SSP3-7.0: [0,98; 1,24] y [0,99; 2,47]; 1 mm Los Lagos 0,25 (107 → 27) | §3.2.1 | `results.json` ssp370 y 1mm | ✓ |
| Fracción ≥ 20 d: 4,1 → 5,4 % (Biobío) | §3.2.2 | Tabla 3 | ✓ |
| n_spells disminuye en las 4 ventanas | §3.2.2 | Tabla 3 | ✓ |
| −6 % dominio; −10 a −13 % (Maipo, Maule, Biobío); Loa +1,4 % | §4.1 | `frecuencia_humeda_ventanas.csv` | ✓ (−6,35; −12,7; −12,7; −10,1; +1,39) |
| 31 de 31 casos; cobertura 70 % y 89 % | §4.1 | `descomposicion_markov_rr_regiones.csv` | ✓ |
| Δρ con IC95 que incluye 0 en las 4 | §4.1 | `contraste_persistencia_regiones.csv` | ✓ (M10) |
| 61 celdas; 18,4–22,5°S | §4.3 | caché de mapas (−18,36 a −22,48) | ✓ |
| 22–40 censuradas sobre p99 en Loa | §4.3 | `rachas_censuradas_por_cuenca.csv` | ✓ (A5 para el resto) |
| 1 090/247; 2 288/949; 3 635/2 048; 5 016/2 722 mm | §4.3 | `estacionalidad_marnov_regiones.csv` | ✓ |
| «entre 2 y 4 veces» | §4.3 | idem | **A2** (1,8–4,4) |
| Suplemento: R₀ local mediana 6,4; p10 2,3; p90 11,1; 39 %; 82 píxeles | S3 | `R0_local_map.npz` | ✓ (39,3 % de los válidos) |
| «25–37°S» (cuartil superior de R₀ local) | S3 (sugerencia) | `R0_local_map.npz` (p10–p90 de latitud: 25,0–36,5°S) | ✓ |
| «Mediana de 6 celdas» (promedio por área) | S5 | `cr2met_lineal_vs_area.npz` (mediana 6, rango 4–11) | ✓ |
| 0,006 pp; < 0,4 d; < 3 d; 1 día en cajas | S5 | `*_lineal_vs_area` | ✓ |

## 5. Observaciones de código (sin efecto en las cifras vigentes, para la tesis)

1. **Mismas reglas en todos los scripts:** NaN no es seco, censura contra la ventana del período, inicio asignado al año de la racha. El Markov reimplementa la detección de rachas con otro algoritmo y reproduce exactamente N_hist y N_fut de `results.json`.
2. **Semillas:** el RR reutiliza la semilla 42 para cada D (sorteos coherentes entre D), el bootstrap de celdas usa 42 + índice, el contraste usa 11 y el Markov 7. Documentado en el registro.
3. **El IC del RR se calcula condicionado a N_hist* > 0** (las réplicas sin rachas históricas se descartan), lo que sesga el IC en las colas raras; ya se declara y se cuenta.
4. **El p99 por celda y el mínimo de 30 rachas** son decisiones discutibles (A4).
5. **Duplicación de funciones** entre `_run_pregunta7_bootstrap.py` y `_generar_pregunta9…` (máscara, bisección de R₀ local): mismos nombres y mismo comportamiento hoy, pero un cambio en uno no llega al otro. Riesgo de mantenimiento, no de resultado.
6. **Caché con borrado manual:** `pdf_durations_i_global.pkl` hay que borrarlo para recalcular (el propio registro lo indica); un cambio de regla que no lo borre dejaría Tabla 3 y Fig. 5 desactualizadas.
7. **Regiones con `contains or touches` en `admin_1`:** las celdas en el borde entre dos regiones podrían caer en ambas; `_referencia_markov_rr_regiones.py` lo comprueba con `assert` (sin solapes). Los scripts de RR y de estacionalidad no lo comprueban, pero usan la misma máscara.

## 6. Lo que sigue pendiente de verificar

- Contenido de las citas (ninguna se leyó hoy).
- Figuras no vistas (esquema, S3, S4, T01) y el código de las figuras S1, S2, T01, 1.
- Que `_test_regrillado_conservativo.py` calcule el promedio por área como dice §S5.
- Que la imagen de la Fig. 5 en el Doc sea la antigua (A1) y que la Fig. 1 no esté estirada (M5).
- Re-ejecución de la cadena completa tras cualquier cambio de regla (A4, A5), según `CLAUDE.md` §4.
