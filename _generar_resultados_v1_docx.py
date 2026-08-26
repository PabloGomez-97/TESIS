"""Genera Resultados v1 a partir del outline (secciones 3 y 4)."""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "paper_figuras"
OUT_REPO = ROOT / "Resultados_v1_dryspells_chile.docx"
OUT_DL = Path("/Users/pablogomez/Downloads/Resultados_v1_dryspells_chile.docx")


def set_run_font(run, size=12, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)


def add_h(doc, text, size=14):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_run_font(run, size=size, bold=True)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    return p


def add_body(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(8)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.5
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.first_line_indent = Inches(0.3)
    run = p.add_run(text)
    set_run_font(run, size=12)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after = Pt(12)
    pf.line_spacing = 1.15
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    set_run_font(run, size=10, italic=True)
    return p


def add_figure(doc, path, caption, width=Inches(6.1)):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(path), width=width)
    add_caption(doc, caption)


def shade_header(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.makeelement(
        qn("w:shd"),
        {qn("w:val"): "clear", qn("w:color"): "auto", qn("w:fill"): "E8E8E8"},
    )
    tcPr.append(shd)


def set_cell_text(cell, text, *, bold=False, size=8):
    cell.text = ""
    run = cell.paragraphs[0].add_run(text)
    set_run_font(run, size=size, bold=bold)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[j], h, bold=True, size=8)
        shade_header(table.rows[0].cells[j])
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            set_cell_text(table.rows[i + 1].cells[j], val, size=8)
    return table


def build():
    doc = Document()
    section = doc.sections[0]
    for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, m, Cm(2.5))
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    add_h(doc, "3. Evaluación histórica (1980–2014)")

    add_body(
        doc,
        "Esta sección responde la primera pregunta del estudio: una vez igualada la "
        "frecuencia de días húmedos (criterio i; R₀ = 5,285 mm día⁻¹ en ALADIN y "
        "R = 1 mm día⁻¹ en CR2MET), ¿reproduce ALADIN la distribución histórica de "
        "duraciones de dry spells? El hilo es climatología de referencia (§3.1), "
        "sesgo espacial residual (§3.2), forma de las PDFs por cuenca (§3.3) y "
        "ausencia de tendencia en el histórico del modelo (§3.4). El cambio futuro "
        "se trata en §4.",
    )

    add_h(doc, "3.1 Climatología de persistencia en CR2MET", size=12)

    add_body(
        doc,
        "CR2MET muestra un contraste norte–sur marcado en la persistencia de rachas "
        "secas, coherente con el gradiente climático de Chile y con diagnósticos "
        "recientes de dry spells en la zona central (Latoja et al., 2025; "
        "Herrera-Marín et al., 2026). En el norte "
        "árido la duración media supera con creces el mes: en la ventana del Loa el "
        "pool regional tiene media de 65,6 días y p99 de 456 días, con máximos por "
        "encima de 1 400 días. En el sur más húmedo las rachas son cortas y numerosas: "
        "en Biobío la media es 6,2 días y el p99, 44 días. Maipo y Maule ocupan el "
        "tramo intermedio (medias de 11,0 y 9,5 días; p99 de 119 y 83 días). Ese "
        "gradiente no es un artefacto de un índice anual: aparece tanto en la media "
        "como en la cola (t99) a escala de píxel (Figura 5, columnas CR2MET).",
    )

    add_body(
        doc,
        "Dos implicaciones siguen de inmediato. Primera: un mismo estadístico "
        "agregado —por ejemplo, el CDD— comprime climas distintos en un solo máximo "
        "por año (Zhang et al., 2011; Sillmann et al., 2013a), mientras que aquí la "
        "cola de Loa y la de Biobío no son conmensurables. Segunda: cualquier "
        "evaluación de ALADIN tiene que respetar ese contraste latitudinal; un sesgo "
        "aceptable en el sur húmedo puede ser grave en el desierto si se pierde la "
        "cola de spells de cientos de días.",
    )

    add_h(doc, "3.2 Sesgo espacial de persistencia: ALADIN versus CR2MET", size=12)

    add_body(
        doc,
        "Con el criterio i, ALADIN reproduce el patrón espacial —norte persistente, "
        "sur de rachas cortas— pero subestima la duración. Sobre Chile continental, "
        "la media espacial es 42,5 días en CR2MET y 35,1 días en ALADIN "
        "(Δ = −7,9 días). El sesgo es mayor en la cola: t99 espacial de 221,8 días "
        "frente a 178,2 días (Δ = −44,7 días). Los mapas de diferencia (Figura 5) "
        "muestran Δ negativo de manera generalizada, más intenso en la transición "
        "norte–centro, donde CR2MET resuelve spells extremadamente largos que ALADIN "
        "acorta. Alrededor de dos tercios de las celdas tienen ALADIN "
        "significativamente más corto en t99 (bootstrap por años, IC95 de Δ que no "
        "incluye cero).",
    )

    add_body(
        doc,
        "Ese residuo no es solo el umbral. Sin calibrar (criterio ii, 1 mm en ambos), "
        "Δ media cae a −27,5 días y Δ t99 a −135,4 días: el exceso de días húmedos "
        "de ALADIN fragmenta las rachas, en línea con el sesgo de llovizna de los "
        "modelos (Stephens et al., 2010; Petrova et al., 2024). La calibración local "
        "(criterio iii) reduce el sesgo de t99 a −35,3 días y el de la media a "
        "−6,7 días, pero no lo elimina. Igualar la frecuencia de wet days es "
        "necesario y no suficiente: persiste una limitación estructural del RCM en "
        "la persistencia seca, sobre todo en la cola. El análisis principal se queda "
        "con el criterio i porque corrige la definición de día húmedo sin introducir "
        "un campo heterogéneo de umbrales; el sesgo residual se interpreta después, "
        "no se oculta.",
    )

    add_figure(
        doc,
        FIG / "fig05_mapas_persistencia_combined.png",
        "Figura 5. Persistencia histórica de dry spells (1980–2014), criterio i. "
        "Fila superior: duración media (días). Fila inferior: percentil 99 de las "
        "duraciones, t99 (días). Columnas: CR2MET regrillado (R = 1 mm), ALADIN "
        "histórico (R₀ = 5,285 mm) y Δ = ALADIN − CR2MET. Ambas fuentes reproducen "
        "el gradiente norte–sur; ALADIN subestima media y cola, con Δ negativo más "
        "marcado en t99.",
        width=Inches(6.2),
    )

    add_h(doc, "3.3 PDFs por cuenca: acuerdo y diferencias regionales", size=12)

    add_body(
        doc,
        "Las PDFs empíricas del pool regional (criterio i; ∫ f(t) dt = n_spells) "
        "permiten ver a la vez la forma de la distribución y el número de eventos "
        "(Figura 6; Tabla 3). En escala log–log, CR2MET e histórico de ALADIN son "
        "aproximadamente paralelos en Maipo, Maule y Biobío: ALADIN captura la forma "
        "y subestima de manera moderada las duraciones (medias 8,6 frente a 11,0 días "
        "en Maipo; 7,3 frente a 9,5 en Maule; 5,3 frente a 6,2 en Biobío). En Loa el "
        "desacuerdo se concentra en la cola: CR2MET mantiene más probabilidad en "
        "spells de cientos de días (p99 = 456 días frente a 360 días en ALADIN; "
        "máximo 1 419 frente a 1 156). ALADIN produce, en todas las cuencas, más "
        "spells que CR2MET (p. ej. 66 267 frente a 55 015 en Loa), coherente con "
        "rachas algo más cortas aunque la frecuencia de wet days ya está emparejada. "
        "La masa en la cola, mayor que la de un proceso exponencial sin memoria, es "
        "coherente con los ajustes paramétricos de duración en CR2MET de Herrera-Marín "
        "et al. (2026); aquí no se impone una familia, de modo que el futuro pueda "
        "desplazar la cola sin asumir un modelo de persistencia.",
    )

    add_body(
        doc,
        "En conjunto, ALADIN es una referencia utilizable para el cambio futuro en "
        "cuencas centrales y meridionales, donde la forma de la PDF histórica se "
        "parece a CR2MET. En el norte árido y en la cola extrema el modelo es más "
        "corto. Ese sesgo no invalida la comparación histórico–futuro dentro de "
        "ALADIN (§4), pero implica que las probabilidades absolutas de spells muy "
        "largos en el norte deben leerse como una cota inferior respecto de CR2MET "
        "(Petrova et al., 2024).",
    )

    add_caption(
        doc,
        "Tabla 3. Estadísticos del pool regional de duraciones de dry spells "
        "(criterio i; 1980–2014 y 2040–2074). n_spells es el número de rachas en el "
        "pool; p90 y p99 están en días.",
    )
    add_table(
        doc,
        ["Cuenca", "Producto", "n_spells", "Media", "Mediana", "p90", "p99", "Máx."],
        [
            ["Loa", "CR2MET hist", "55 015", "65,6", "23", "185", "456", "1 419"],
            ["Loa", "ALADIN hist", "66 267", "54,6", "22", "143", "360", "1 156"],
            ["Loa", "ALADIN fut", "64 043", "56,6", "25", "151", "385", "1 853"],
            ["Maipo", "CR2MET hist", "150 209", "11,0", "5", "23", "119", "283"],
            ["Maipo", "ALADIN hist", "181 049", "8,6", "4", "18", "87", "194"],
            ["Maipo", "ALADIN fut", "159 887", "10,0", "4", "21", "112", "224"],
            ["Maule", "CR2MET hist", "184 462", "9,5", "5", "21", "83", "206"],
            ["Maule", "ALADIN hist", "228 077", "7,3", "4", "17", "57", "160"],
            ["Maule", "ALADIN fut", "202 463", "8,6", "4", "19", "76", "188"],
            ["Biobío", "CR2MET hist", "344 251", "6,2", "4", "14", "44", "109"],
            ["Biobío", "ALADIN hist", "399 130", "5,3", "3", "12", "34", "153"],
            ["Biobío", "ALADIN fut", "370 260", "6,0", "3", "14", "39", "120"],
        ],
    )
    doc.add_paragraph()

    add_figure(
        doc,
        FIG / "fig42_pdf_cuencas_loglog_panel.png",
        "Figura 6. PDFs empíricas de duración de dry spells (criterio i, escala "
        "log–log; ∫ f dt = n_spells). Negro: CR2MET 1980–2014. Azul: ALADIN histórico. "
        "Rojo discontinuo: ALADIN futuro (SSP5-8.5, 2040–2074). El histórico de ALADIN "
        "sigue la forma de CR2MET en Maipo, Maule y Biobío, y pierde cola en Loa. El "
        "futuro se discute en §4.2: n_spells disminuye y la cola derecha se extiende "
        "en las cuencas centrales y en Biobío.",
        width=Inches(6.2),
    )

    add_h(doc, "3.4 Tendencias en ALADIN histórico: resultado nulo", size=12)

    add_body(
        doc,
        "Antes de interpretar el futuro conviene preguntar si el propio histórico de "
        "ALADIN (1980–2014, R₀ = 5,285 mm) ya muestra una tendencia en persistencia. "
        "La pendiente media espacial de la duración media es −0,027 días por década; "
        "la de t99 es +0,001 días por década —ambas indistinguibles de cero a escala "
        "de Chile. Menos del 0,2 % de las celdas tiene tendencia significativa "
        "(p < 0,05): 0,035 % para la media y 0,115 % para t99 (Figura 7). No hay, "
        "en este diseño, una señal espacial robusta de alargamiento o acortamiento "
        "de dry spells en el histórico del RCM.",
    )

    add_body(
        doc,
        "Ese resultado nulo no contradice el secado observado en Chile (Boisier et al., "
        "2016, 2018; Garreaud et al., 2017, 2020): dice que ALADIN, en 1980–2014, no "
        "reproduce una tendencia detectable en la persistencia de rachas. Por eso el "
        "mensaje principal del trabajo no es una tendencia histórica del modelo, sino "
        "el contraste entre dos climas de 35 años —histórico versus SSP5-8.5 a "
        "mediados de siglo— dentro del mismo RCM.",
    )

    add_figure(
        doc,
        FIG / "fig33_tendencias_aladin_null_summary.png",
        "Figura 7. Ausencia de tendencia robusta en persistencia de dry spells en "
        "ALADIN histórico (1980–2014, R₀ = 5,285 mm). Izquierda: pendiente media "
        "espacial (días/década) de la duración media y de t99. Derecha: porcentaje de "
        "celdas con p < 0,05. Ambas pendientes son ~0 y la fracción significativa "
        "queda por debajo de 0,2 %.",
        width=Inches(6.0),
    )

    add_h(doc, "4. Proyecciones futuras (SSP5-8.5, 2040–2074)")

    add_body(
        doc,
        "La segunda pregunta es si, bajo SSP5-8.5, cambia la probabilidad de rachas "
        "prolongadas —no solo el acumulado anual. La comparación es ALADIN histórico "
        "versus ALADIN futuro, con el mismo R₀. A escala de Chile, el modelo proyecta "
        "menos precipitación anual en el centro-sur (del orden de −10 a −20 % en gran "
        "parte del dominio) y, a la vez, un aumento de la duración media de dry "
        "spells de varios días en el norte y el centro (Figura 8). Eso es compatible "
        "con un secado dominado por más días secos más que por una mera baja de "
        "intensidad (Polade et al., 2014; Wainwright et al., 2021). Las subsecciones "
        "siguientes descomponen ese cambio en probabilidades de excedencia (RR) y en "
        "la forma de las PDFs.",
    )

    add_figure(
        doc,
        FIG / "mapas_aladin_pr_anual_y_dryspells_hist_fut.png",
        "Figura 8. ALADIN CHP12, criterio i. Fila superior: precipitación anual media "
        "en 1980–2014, 2040–2074 (SSP5-8.5) y cambio relativo (%). Fila inferior: "
        "duración media de dry spells en ambos períodos y diferencia absoluta (días). "
        "El centro-sur pierde lluvia anual y alarga las rachas secas; no es un simple "
        "reescalado de la media de precipitación.",
        width=Inches(6.2),
    )

    add_h(doc, "4.1 Risk ratios: más dry spells largos", size=12)

    add_body(
        doc,
        "El risk ratio RR(D) = P_fut(t ≥ D) / P_hist(t ≥ D) se estima en pools de "
        "regiones administrativas, con spells de inicio en marzo–noviembre y bootstrap "
        "por años (Figura 9; Tabla 4). A D = 20 días, Coquimbo pasa de 15,4 % a 19,6 % "
        "(RR = 1,27; IC95 [1,09, 1,48]); O’Higgins, de 5,8 % a 7,5 % (RR = 1,30; "
        "[1,04, 1,63]); La Araucanía, de 0,74 % a 1,37 % (RR = 1,84; [1,16, 3,12]). "
        "En esas tres regiones el IC95 queda por completo sobre 1: el aumento es "
        "significativo. Los Lagos tiene RR = 2,24 (0,13 % → 0,29 %), pero el IC95 "
        "[0,91, 9,06] cruza 1: el cociente es inestable porque los spells ≥ 20 días "
        "son raros en el histórico (601 eventos en 471 361 spells).",
    )

    add_body(
        doc,
        "La señal no es un desplazamiento uniforme de toda la distribución. En "
        "Coquimbo, a D = 5 días el RR es 1,06 y el IC95 aún toca 1; el aumento se "
        "vuelve significativo desde D ≈ 10 días y se mantiene en la cola (p. ej. "
        "RR = 1,42 a D = 30 días, IC95 [1,18, 1,68]). En La Araucanía el RR crece con "
        "D y alcanza 3,71 a 30 días, con IC95 muy ancho ([1,37, 12,5]): hay más "
        "probabilidad relativa de spells largos, pero la rareza histórica infla el "
        "cociente. Eso es exactamente lo que un CDD anual no muestra: el cambio está "
        "en la frecuencia de excedencias a lo largo de la cola, no en un único máximo "
        "por año (Sillmann et al., 2013b; Wainwright et al., 2021).",
    )

    add_caption(
        doc,
        "Tabla 4. Risk ratio a D = 20 días. Pools regionales, ALADIN, R₀ = 5,285 mm, "
        "inicio marzo–noviembre. IC95 por bootstrap de años (1 000 réplicas). "
        "Aumento significativo si RR > 1 y el IC no incluye 1.",
    )
    add_table(
        doc,
        ["Región", "P_hist (t≥20)", "P_fut (t≥20)", "RR", "IC95", "¿Aumento sig.?"],
        [
            ["Coquimbo", "15,4 %", "19,6 %", "1,27", "[1,09, 1,48]", "Sí"],
            ["O’Higgins", "5,8 %", "7,5 %", "1,30", "[1,04, 1,63]", "Sí"],
            ["La Araucanía", "0,74 %", "1,37 %", "1,84", "[1,16, 3,12]", "Sí"],
            ["Los Lagos", "0,13 %", "0,29 %", "2,24", "[0,91, 9,06]", "No (IC cruza 1)"],
        ],
    )
    doc.add_paragraph()

    add_figure(
        doc,
        FIG / "fig08_rr_curves_aladin.png",
        "Figura 9. Risk ratio RR(D) de dry spells con duración al menos D días. "
        "ALADIN, SSP5-8.5 (2040–2074) respecto de 1980–2014, R₀ = 5,285 mm, inicio "
        "marzo–noviembre. La banda es el IC95 por bootstrap de años. La línea "
        "punteada marca RR = 1. Coquimbo, O’Higgins y La Araucanía superan 1 con "
        "IC que no cruza 1 a D = 20 días; Los Lagos no. En el sur la banda se abre "
        "cuando D crece y los eventos históricos escasean.",
        width=Inches(6.1),
    )

    add_h(doc, "4.2 PDFs por cuenca: extensión de la cola futura", size=12)

    add_body(
        doc,
        "Las mismas PDFs de la Figura 6, ahora leyendo la curva roja, muestran el "
        "cambio de forma y de n_spells. En las cuatro cuencas el futuro tiene menos "
        "spells que el histórico de ALADIN (p. ej. Maipo 181 049 → 159 887; Maule "
        "228 077 → 202 463): hay menos interrupciones, es decir, rachas más largas. "
        "La media sube en todas (Loa 54,6 → 56,6 días; Maipo 8,6 → 10,0; Maule 7,3 → "
        "8,6; Biobío 5,3 → 6,0). El p99 también (360 → 385; 87 → 112; 57 → 76; "
        "34 → 39 días). En escala log–log las curvas no son una traslación paralela: "
        "el exceso futuro se nota más a la derecha, en spells de decenas a cientos de "
        "días (Figura 6 y Figura 10).",
    )

    add_body(
        doc,
        "El máximo muestral es un estadístico ruidoso y no debe usarse solo. En Loa "
        "el máximo pasa de 1 156 a 1 853 días; en Maipo y Maule también aumenta "
        "(194 → 224 y 160 → 188). En Biobío el máximo del pool baja (153 → 120) "
        "aunque media, p90, p99 y P(t ≥ 20) suben. Eso ilustra por qué el objeto "
        "del trabajo es la distribución y las excedencias, no el récord de un píxel.",
    )

    add_figure(
        doc,
        FIG / "demostracion_alargamiento_futuro.png",
        "Figura 10. Diagnósticos de alargamiento futuro por cuenca (criterio i). "
        "(a, d) RR(D) del pool de cada ventana. (b) Cambio relativo de media, p90 y "
        "p99. (c) Probabilidad de spells ≥ 20 días y RR asociado. Loa cambia poco en "
        "términos relativos (ya es árido). Maipo, Maule y Biobío aumentan la cola: "
        "RR(D=20) ≈ 1,22–1,30 y el p99 sube ~15–30 %.",
        width=Inches(6.2),
    )

    add_h(doc, "4.3 Consistencia entre RR regionales y PDFs por cuenca", size=12)

    add_body(
        doc,
        "RR y PDF son diagnósticos complementarios, no duplicados. El RR resume un "
        "umbral de excedencia en un pool administrativo; la PDF muestra toda la "
        "duración y el recuento de eventos en una ventana de cuenca. Las regiones con "
        "RR significativo a 20 días —Coquimbo, O’Higgins, La Araucanía— coinciden con "
        "el tramo latitudinal de Maipo–Maule–Biobío, donde las PDFs futuras extienden "
        "la cola y baja n_spells. En Loa el RR de cuenca a 20 días es solo 1,05: el "
        "clima ya es de spells largos y el cambio relativo es pequeño, aunque media y "
        "p99 suben. En Los Lagos la no significancia del RR refleja el límite del "
        "bootstrap cuando P_hist(t ≥ 20) ≈ 0,13 %: hay pocos eventos para estimar la "
        "cola, no una evidencia de que el sur húmedo se libere del alargamiento.",
    )

    add_body(
        doc,
        "El resultado conjunto es, por tanto, doble. Históricamente, ALADIN calibrado "
        "reproduce la forma de las PDFs en el centro y el sur y subestima persistencia "
        "en el norte y en t99. Hacia mediados de siglo, el mismo modelo, bajo "
        "SSP5-8.5, proyecta mayor probabilidad de dry spells largos en varias regiones "
        "de Chile, con señal que se refuerza en la cola y que no se reduce a un descenso "
        "de la precipitación media. Ese es el objeto distributivo planteado en la "
        "Introducción. El RR(D) describe un cambio en la probabilidad de duraciones "
        "largas, no en la tasa de inicios: el clustering y la memoria de onsets son "
        "el objeto de Herrera-Marín (2026) y Herrera-Marín et al. (2026). Ambos "
        "diagnósticos son complementarios.",
    )

    add_h(doc, "Referencias citadas en Resultados")
    refs = [
        "Boisier, J. P., Rondanelli, R., Garreaud, R. D., y Muñoz, F. (2016). Anthropogenic and natural contributions to the Southeast Pacific precipitation decline and recent megadrought in central Chile. Geophysical Research Letters, 43(1), 413–421. https://doi.org/10.1002/2015GL067265",
        "Boisier, J. P., Alvarez-Garretón, C., Cordero, R. R., Damiani, A., Gallardo, L., Garreaud, R. D., Lambert, F., Ramallo, C., Rojas, M., y Rondanelli, R. (2018). Anthropogenic drying in central-southern Chile evidenced by long-term observations and climate model simulations. Elementa: Science of the Anthropocene, 6, 74. https://doi.org/10.1525/elementa.328",
        "Garreaud, R. D., Alvarez-Garreton, C., Barichivich, J., Boisier, J. P., Christie, D., Galleguillos, M., … Zambrano-Bigiarini, M. (2017). The 2010–2015 megadrought in central Chile: Impacts on regional hydroclimate and vegetation. Hydrology and Earth System Sciences, 21(12), 6307–6327. https://doi.org/10.5194/hess-21-6307-2017",
        "Garreaud, R. D., Boisier, J. P., Rondanelli, R., Damiani, A., Veloso-Aguila, D., y Muñoz, F. (2020). The Central Chile Mega Drought (2010–2018): A climate dynamics perspective. International Journal of Climatology, 40(1), 421–439. https://doi.org/10.1002/joc.6219",
        "Herrera-Marín, M. (2026). A stochastic hazard decomposition for dry-spell onset dynamics under non-stationary hydroclimatic forcing. Stochastic Environmental Research and Risk Assessment, 40, 172. https://doi.org/10.1007/s00477-026-03309-9",
        "Herrera-Marín, M., Rivera, D., y Godoy-Faúndez, A. (2026). Drought begets drought: self-exciting onsets, spell persistence, and network structure in hydroclimatic extremes. Environmental Research Communications, 8, 051001. https://doi.org/10.1088/2515-7620/ae639a",
        "Latoja, D., Rivera, D., Lillo-Saavedra, M., y Godoy-Faúndez, Á. (2025). Shifting dry spells in Central Chile: Implications for agricultural water planning. Agricultural & Environmental Letters, 10, e70037. https://doi.org/10.1002/ael2.70037",
        "Petrova, I. Y., Miralles, D. G., Brient, F., Donat, M. G., Min, S.-K., Kim, Y.-H., y Bador, M. (2024). Observation-constrained projections reveal longer-than-expected dry spells. Nature, 633, 594–600. https://doi.org/10.1038/s41586-024-07887-y",
        "Polade, S. D., Pierce, D. W., Cayan, D. R., Gershunov, A., y Dettinger, M. D. (2014). The key role of dry days in changing regional climate and precipitation regimes. Scientific Reports, 4, 4364. https://doi.org/10.1038/srep04364",
        "Sillmann, J., Kharin, V. V., Zhang, X., Zwiers, F. W., y Bronaugh, D. (2013a). Climate extremes indices in the CMIP5 multimodel ensemble: Part 1. Model evaluation in the present climate. Journal of Geophysical Research: Atmospheres, 118, 1716–1733. https://doi.org/10.1002/jgrd.50203",
        "Sillmann, J., Kharin, V. V., Zwiers, F. W., Zhang, X., y Bronaugh, D. (2013b). Climate extremes indices in the CMIP5 multimodel ensemble: Part 2. Future climate projections. Journal of Geophysical Research: Atmospheres, 118, 2473–2493. https://doi.org/10.1002/jgrd.50188",
        "Stephens, G. L., L’Ecuyer, T., Forbes, R., Gettelman, A., Golaz, J.-C., Bodas-Salcedo, A., Suzuki, K., Gabriel, P., y Haynes, J. (2010). Dreary state of precipitation in global models. Journal of Geophysical Research: Atmospheres, 115, D24211. https://doi.org/10.1029/2010JD014532",
        "Wainwright, C. M., Black, E., y Allan, R. P. (2021). Future changes in wet and dry season characteristics in CMIP5 and CMIP6 simulations. Journal of Hydrometeorology, 22, 2339–2357. https://doi.org/10.1175/JHM-D-21-0017.1",
        "Zhang, X., Alexander, L., Hegerl, G. C., Jones, P., Klein Tank, A., Peterson, T. C., Trewin, B., y Zwiers, F. W. (2011). Indices for monitoring changes in extremes based on daily temperature and precipitation data. Wiley Interdisciplinary Reviews: Climate Change, 2(6), 851–870. https://doi.org/10.1002/wcc.147",
    ]
    for r in refs:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_after = Pt(4)
        pf.line_spacing = 1.15
        pf.left_indent = Cm(1.25)
        pf.first_line_indent = Cm(-1.25)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run = p.add_run(r)
        set_run_font(run, size=11)

    doc.save(OUT_REPO)
    try:
        doc.save(OUT_DL)
    except OSError:
        pass
    return OUT_REPO


if __name__ == "__main__":
    print("Wrote", build())
