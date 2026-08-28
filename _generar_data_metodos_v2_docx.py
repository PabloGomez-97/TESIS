"""Genera Datos y métodos v2, y un documento combinado con la Introducción del usuario."""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "paper_figuras"
USER_DOC = Path("/Users/pablogomez/Downloads/Introducción v1 (1).docx")

OUT_METHODS = ROOT / "Datos_y_Metodos_v2.docx"
OUT_COMBINED = ROOT / "Introduccion_y_Metodos_v2.docx"
OUT_DL_METHODS = Path("/Users/pablogomez/Downloads/Datos_y_Metodos_v2.docx")
OUT_DL_COMBINED = Path("/Users/pablogomez/Downloads/Introduccion_y_Metodos_v2.docx")


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


def add_body(doc, text, indent=True):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(8)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.5
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        pf.first_line_indent = Inches(0.3)
    run = p.add_run(text)
    set_run_font(run, size=12)
    return p


def add_eq(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    set_run_font(run, size=12, italic=True)
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


def add_ref(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(4)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.15
    pf.left_indent = Cm(1.25)
    pf.first_line_indent = Cm(-1.25)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    set_run_font(run, size=11)
    return p


def shade_header(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = tcPr.makeelement(
        qn("w:shd"),
        {
            qn("w:val"): "clear",
            qn("w:color"): "auto",
            qn("w:fill"): "E8E8E8",
        },
    )
    tcPr.append(shd)


def set_cell_text(cell, text, *, bold=False, size=8):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[j], h, bold=True, size=8)
        shade_header(table.rows[0].cells[j])
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            set_cell_text(table.rows[i + 1].cells[j], val, size=8)
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Cm(w)
    return table


def copy_intro(src_doc, dst_doc):
    """Copia la Introducción del usuario hasta (sin incluir) Métodos."""
    for para in src_doc.paragraphs:
        t = para.text.strip()
        if t in {
            "Datos y métodos",
            "Datos y Métodos",
            "2. Datos y métodos",
            "2. Métodos",
            "Métodos",
            "Methods",
            "2. Methods",
        }:
            break
        if para.style and para.style.name.startswith("Heading") and (
            t.lower().startswith("datos") or t.lower().startswith("métodos") or t.lower() == "methods"
        ):
            break
        p = dst_doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_after = Pt(8)
        pf.line_spacing = 1.5
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if t == "Introducción":
            run = p.add_run(para.text)
            set_run_font(run, size=14, bold=True)
            p.paragraph_format.first_line_indent = None
        elif t:
            pf.first_line_indent = Inches(0.3)
            run = p.add_run(para.text)
            set_run_font(run, size=12)
    return dst_doc


REFS = [
    "Álvarez-Garretón, C., Mendoza, P. A., Boisier, J. P., Addor, N., Galleguillos, M., Zambrano-Bigiarini, M., Lara, A., Puelma, C., Cortes, G., Garreaud, R., McPhee, J., y Ayala, A. (2018). The CAMELS-CL dataset: catchment attributes and meteorology for large sample studies – Chile. Hydrology and Earth System Sciences, 22, 5817–5846. https://doi.org/10.5194/hess-22-5817-2018",
    "Araya-Osses, D., Casanueva, A., Román-Figueroa, C., Uribe, J. M., y Paneque, M. (2020). Climate change projections of temperature and precipitation in Chile based on statistical downscaling. Climate Dynamics, 54, 4309–4330. https://doi.org/10.1007/s00382-020-05231-4",
    "Báez-Villanueva, O. M., Zambrano-Bigiarini, M., Miralles, D. G., Beck, H. E., Siegmund, J. F., Alvarez-Garreton, C., Verbist, K., Garreaud, R., Boisier, J. P., y Galleguillos, M. (2024). On the timescale of drought indices for monitoring streamflow drought considering catchment hydrological regimes. Hydrology and Earth System Sciences, 28, 1415–1439. https://doi.org/10.5194/hess-28-1415-2024",
    "Boisier, J. P., Rondanelli, R., Garreaud, R. D., y Muñoz, F. (2016). Anthropogenic and natural contributions to the Southeast Pacific precipitation decline and recent megadrought in central Chile. Geophysical Research Letters, 43(1), 413–421. https://doi.org/10.1002/2015GL067265",
    "Boisier, J. P., Alvarez-Garretón, C., Cordero, R. R., Damiani, A., Gallardo, L., Garreaud, R. D., Lambert, F., Ramallo, C., Rojas, M., y Rondanelli, R. (2018). Anthropogenic drying in central-southern Chile evidenced by long-term observations and climate model simulations. Elementa: Science of the Anthropocene, 6, 74. https://doi.org/10.1525/elementa.328",
    "Bozkurt, D., Rojas, M., Boisier, J. P., y Valdivieso, J. (2018). Projected hydroclimate changes over Andean basins in central Chile from downscaled CMIP5 models under the low and high emission scenarios. Climatic Change, 150, 131–154. https://doi.org/10.1007/s10584-018-2246-7",
    "Garreaud, R. D., Alvarez-Garreton, C., Barichivich, J., Boisier, J. P., Christie, D., Galleguillos, M., … Zambrano-Bigiarini, M. (2017). The 2010–2015 megadrought in central Chile: Impacts on regional hydroclimate and vegetation. Hydrology and Earth System Sciences, 21(12), 6307–6327. https://doi.org/10.5194/hess-21-6307-2017",
    "Garreaud, R. D., Boisier, J. P., Rondanelli, R., Damiani, A., Veloso-Aguila, D., y Muñoz, F. (2020). The Central Chile Mega Drought (2010–2018): A climate dynamics perspective. International Journal of Climatology, 40(1), 421–439. https://doi.org/10.1002/joc.6219",
    "Giorgi, F., Im, E.-S., Coppola, E., Diffenbaugh, N. S., Gao, X. J., Mariotti, L., y Shi, Y. (2011). Higher hydroclimatic intensity with global warming. Journal of Climate, 24(20), 5309–5324. https://doi.org/10.1175/2011JCLI3979.1",
    "Giorgi, F., Raffaele, F., y Coppola, E. (2019). The response of precipitation characteristics to global warming from climate projections. Earth System Dynamics, 10, 73–89. https://doi.org/10.5194/esd-10-73-2019",
    "Herrera-Marín, M. (2026). A stochastic hazard decomposition for dry-spell onset dynamics under non-stationary hydroclimatic forcing. Stochastic Environmental Research and Risk Assessment, 40, 172. https://doi.org/10.1007/s00477-026-03309-9",
    "Herrera-Marín, M., Rivera, D., y Godoy-Faúndez, A. (2026). Drought begets drought: self-exciting onsets, spell persistence, and network structure in hydroclimatic extremes. Environmental Research Communications, 8, 051001. https://doi.org/10.1088/2515-7620/ae639a",
    "Lana, X., Martínez, M. D., Burgueño, A., Serra, C., Martín-Vide, J., y Gómez, L. (2006). Distributions of long dry spells in the Iberian Peninsula, years 1951–1990. International Journal of Climatology, 26(14), 1999–2021. https://doi.org/10.1002/joc.1354",
    "Latoja, D., Rivera, D., Lillo-Saavedra, M., y Godoy-Faúndez, Á. (2025). Shifting dry spells in Central Chile: Implications for agricultural water planning. Agricultural & Environmental Letters, 10, e70037. https://doi.org/10.1002/ael2.70037",
    "Martinez-Villalobos, C., y Neelin, J. D. (2018). Shifts in precipitation accumulation extremes during the warm season over the United States. Geophysical Research Letters, 45(16), 8586–8595. https://doi.org/10.1029/2018GL078465",
    "Martinez-Villalobos, C., y Neelin, J. D. (2019). Why do precipitation intensities tend to follow gamma distributions? Journal of the Atmospheric Sciences, 76(11), 3611–3631. https://doi.org/10.1175/JAS-D-18-0343.1",
    "Martinez-Villalobos, C., Neelin, J. D., y Pendergrass, A. G. (2022). Metrics for evaluating CMIP6 representation of daily precipitation probability distributions. Journal of Climate, 35(17), 5719–5743. https://doi.org/10.1175/JCLI-D-21-0617.1",
    "Pendergrass, A. G., y Knutti, R. (2018). The uneven nature of daily precipitation and its change. Geophysical Research Letters, 45(21), 11980–11988. https://doi.org/10.1029/2018GL080298",
    "Petrova, I. Y., Miralles, D. G., Brient, F., Donat, M. G., Min, S.-K., Kim, Y.-H., y Bador, M. (2024). Observation-constrained projections reveal longer-than-expected dry spells. Nature, 633, 594–600. https://doi.org/10.1038/s41586-024-07887-y",
    "Polade, S. D., Pierce, D. W., Cayan, D. R., Gershunov, A., y Dettinger, M. D. (2014). The key role of dry days in changing regional climate and precipitation regimes. Scientific Reports, 4, 4364. https://doi.org/10.1038/srep04364",
    "Polade, S. D., Gershunov, A., Cayan, D. R., Dettinger, M. D., y Pierce, D. W. (2017). Precipitation in a warming world: Assessing projected hydro-climate changes in California and other Mediterranean climate regions. Scientific Reports, 7, 10783. https://doi.org/10.1038/s41598-017-11285-y",
    "Reboita, M. S., da Rocha, R. P., de Souza, C. A., Baldoni, T. C., Silva, P. L. L. da S., y Ferreira, G. W. S. (2022). Future projections of extreme precipitation climate indices over South America based on CORDEX-CORE multimodel ensemble. Atmosphere, 13(9), 1463. https://doi.org/10.3390/atmos13091463",
    "Salazar, Á., Thatcher, M., Goubanova, K., Bernal, P., Gutiérrez, J., y Squeo, F. (2024). CMIP6 precipitation and temperature projections for Chile. Climate Dynamics. https://doi.org/10.1007/s00382-023-07034-9",
    "Sillmann, J., Kharin, V. V., Zhang, X., Zwiers, F. W., y Bronaugh, D. (2013a). Climate extremes indices in the CMIP5 multimodel ensemble: Part 1. Model evaluation in the present climate. Journal of Geophysical Research: Atmospheres, 118, 1716–1733. https://doi.org/10.1002/jgrd.50203",
    "Sillmann, J., Kharin, V. V., Zwiers, F. W., Zhang, X., y Bronaugh, D. (2013b). Climate extremes indices in the CMIP5 multimodel ensemble: Part 2. Future climate projections. Journal of Geophysical Research: Atmospheres, 118, 2473–2493. https://doi.org/10.1002/jgrd.50188",
    "Skansi, M. de los M., Brunet, M., Sigró, J., Aguilar, E., Groening, J. A. A., Bentancur, O. J., … Jones, P. D. (2013). Warming and wetting signals emerging from analysis of changes in climate extreme indices over South America. Global and Planetary Change, 100, 295–307. https://doi.org/10.1016/j.gloplacha.2012.11.004",
    "Stephens, G. L., L’Ecuyer, T., Forbes, R., Gettelman, A., Golaz, J.-C., Bodas-Salcedo, A., Suzuki, K., Gabriel, P., y Haynes, J. (2010). Dreary state of precipitation in global models. Journal of Geophysical Research: Atmospheres, 115, D24211. https://doi.org/10.1029/2010JD014532",
    "Trenberth, K. E., Dai, A., Rasmussen, R. M., y Parsons, D. B. (2003). The changing character of precipitation. Bulletin of the American Meteorological Society, 84(9), 1205–1218. https://doi.org/10.1175/BAMS-84-9-1205",
    "Wainwright, C. M., Black, E., y Allan, R. P. (2021). Future changes in wet and dry season characteristics in CMIP5 and CMIP6 simulations. Journal of Hydrometeorology, 22, 2339–2357. https://doi.org/10.1175/JHM-D-21-0017.1",
    "Zhang, X., Alexander, L., Hegerl, G. C., Jones, P., Klein Tank, A., Peterson, T. C., Trewin, B., y Zwiers, F. W. (2011). Indices for monitoring changes in extremes based on daily temperature and precipitation data. Wiley Interdisciplinary Reviews: Climate Change, 2(6), 851–870. https://doi.org/10.1002/wcc.147",
    "Zolina, O., Simmer, C., Belyaev, K., Gulev, S. K., y Koltermann, P. (2013). Changes in the duration of European wet and dry spells during the last 60 years. Journal of Climate, 26(6), 2022–2047. https://doi.org/10.1175/JCLI-D-11-00498.1",
]


def write_methods(doc):
    add_h(doc, "2. Métodos")

    add_h(doc, "2.1 Datos", size=12)

    add_body(
        doc,
        "El análisis utiliza dos fuentes de precipitación diaria. Como referencia "
        "observacional del período histórico se emplea CR2MET v2.5 (Centro de Ciencia "
        "del Clima y la Resiliencia, CR2), un producto gridded de precipitación diaria "
        "para Chile en una grilla regular de latitud–longitud (~0,05°; ~5 km), con "
        "unidades en mm día⁻¹. CR2MET combina observaciones de estación con información "
        "de reanálisis y se usa de manera rutinaria como referencia hidroclimática en "
        "el país (Álvarez-Garretón et al., 2018; Latoja et al., 2025; "
        "Herrera-Marín et al., 2026). Como modelo "
        "climático regional se utiliza ALADIN CHP12 (configuración CNRM-ALADIN64 v1, "
        "dominio chileno de ~12 km), forzado por el modelo global CNRM-ESM2-1 "
        "(miembro r1i1p1f2). La precipitación de ALADIN se entrega en kg m⁻² s⁻¹ y se "
        "convierte a mm día⁻¹ multiplicando por 86 400. Se emplea un único RCM y un "
        "único escenario de altas emisiones (SSP5-8.5) de manera deliberada: el objetivo "
        "no es un ensamble de incertidumbre, sino evaluar la persistencia de rachas "
        "secas dentro de un mismo modelo, en el histórico y en el futuro.",
    )

    add_body(
        doc,
        "El dominio de estudio es Chile continental. La máscara terrestre se construye "
        "a partir de la geometría de Chile en Natural Earth (10 m). Sobre la grilla "
        "nativa de ALADIN, el dominio continental comprende aproximadamente 3 587 "
        "celdas válidas. Para permitir comparaciones píxel a píxel, CR2MET se interpola "
        "linealmente a los centros de celda de esa grilla. Esta decisión fija ALADIN "
        "como marco espacial de referencia, porque es el producto que se evalúa en el "
        "histórico y el único disponible para el futuro. La interpolación lineal es "
        "reproducible; se reconoce, no obstante, que introduce un suavizado leve de "
        "CR2MET y que no conserva exactamente la precipitación integrada del dominio.",
    )

    add_body(
        doc,
        "Se analizan dos ventanas de 35 años de igual longitud: el histórico 1980–2014 "
        "(CR2MET y ALADIN) y el futuro 2040–2074 bajo SSP5-8.5 (solo ALADIN). El año "
        "2014 coincide con el término del experimento histórico de CMIP6, de modo que "
        "el histórico de ALADIN no mezcla forzante histórico y de escenario. El "
        "emparejamiento de longitudes evita que diferencias en el tamaño muestral "
        "condicionen la comparación histórico–futuro.",
    )

    add_body(
        doc,
        "Los análisis de distribuciones de duración se concentran en cuatro ventanas "
        "representativas del gradiente climático chileno: Loa (norte árido), "
        "Maipo y Maule (zona central mediterránea) y Biobío "
        "(sur más húmedo). No son polígonos oficiales de cuenca, sino cajas de "
        "longitud–latitud sobre la grilla ALADIN, intersectadas con la máscara "
        "continental; se eligieron como casos de estudio a lo largo del gradiente "
        "norte–sur, no como delineación hidrológica. Los risk ratios, en cambio, se "
        "calculan sobre pools definidos por regiones administrativas (Coquimbo, "
        "O’Higgins, La Araucanía y Los Lagos; geometrías Natural Earth admin_1). "
        "La distinción es deliberada: las cajas de cuenca sirven para resolver la "
        "forma de la PDF —en particular la cola— en climas contrastantes; las regiones "
        "administrativas aportan un tamaño muestral mayor para las probabilidades de "
        "excedencia y coinciden con unidades usadas en gestión hídrica. Las dos "
        "geografías no son intercambiables y no deben leerse como el mismo dominio "
        "(Figura 1; Tabla 1).",
    )

    add_figure(
        doc,
        FIG / "fig02_dominio_cuencas_regiones.png",
        "Figura 1. Dominio de estudio sobre Chile continental. Los rectángulos "
        "indican las ventanas usadas para las PDFs de duración de dry spells (Loa, "
        "Maipo, Maule y Biobío). Las áreas sombreadas son las regiones administrativas "
        "usadas para los risk ratios (Coquimbo, O’Higgins, La Araucanía y Los Lagos). "
        "Ambas geografías cubren el gradiente norte árido–sur húmedo, pero no coinciden "
        "como polígonos.",
        width=Inches(3.35),
    )

    add_caption(
        doc,
        "Tabla 1. Conjuntos de datos: producto, rol, grilla, período, variable y umbral de día húmedo.",
    )
    add_table(
        doc,
        ["Dataset", "Rol", "Grilla", "Período", "Umbral"],
        [
            [
                "CR2MET v2.5",
                "Referencia observacional gridded",
                "Lat/lon 0,05°; interpolada a ALADIN",
                "1980–2014",
                "R = 1 mm día⁻¹",
            ],
            [
                "ALADIN CHP12 histórico",
                "RCM CNRM-ALADIN64; forzado por CNRM-ESM2-1",
                "Nativa ~12 km; 3 587 celdas en Chile",
                "1980–2014 (35 años)",
                "R₀ = 5,285 mm día⁻¹ (criterio i)",
            ],
            [
                "ALADIN CHP12 SSP5-8.5",
                "Proyección del mismo RCM",
                "Idéntica al histórico",
                "2040–2074 (35 años)",
                "Mismo R₀; RR con inicio mar–nov",
            ],
        ],
        col_widths=[4.0, 4.2, 4.0, 3.2, 3.6],
    )
    doc.add_paragraph()

    add_h(doc, "2.2 Definición de días húmedos/secos y dry spells", size=12)

    add_body(
        doc,
        "Se define wet day (día húmedo) como un día con precipitación diaria pr ≥ R, "
        "donde R es un umbral en mm día⁻¹, y dry day (día seco) como un día con pr < R. "
        "Un dry spell es una racha consecutiva de dry days; su duración t es el número "
        "de días de la racha, con mínimo de un día (Figura 2). El umbral de 1 mm día⁻¹ "
        "en CR2MET sigue la convención ETCCDI/Climdex usada para CDD (Zhang et al., "
        "2011; Sillmann et al., 2013a) y la misma definición operativa de dry day que "
        "Latoja et al. (2025) y Herrera-Marín et al. (2026). Los días con valor faltante (NaN) no se tratan "
        "como secos. Sobre cada ventana temporal (1980–2014 o 2040–2074), primero se "
        "identifican todas las rachas en la serie diaria completa. Se excluyen las "
        "rachas censuradas: aquellas que comienzan el primer día de la ventana o "
        "terminan el último día, porque su duración verdadera es desconocida. Las "
        "rachas que ya estaban activas al inicio del período (p. ej., una sequía "
        "multianual en curso en enero de 1980) no se cuentan como un nuevo evento "
        "en 1980; solo entran spells cuyo inicio queda dentro de la ventana y cuya "
        "duración está completamente observada. En Loa, donde las rachas pueden "
        "superar varios años, esta regla evita subestimar o sobrestimar duraciones "
        "en los bordes del período.",
    )

    add_body(
        doc,
        "A diferencia del índice CDD —el máximo anual de días secos consecutivos, que "
        "en muchas implementaciones se interrumpe al cambio de año (Sillmann et al., "
        "2013a)—, aquí un dry spell puede cruzar el 1 de enero dentro de la ventana "
        "de análisis: la racha se conserva entera y se etiqueta con el año calendario "
        "de su inicio. En el norte árido eso es esencial, porque las rachas de "
        "cientos de días no caben en un año calendario. Tampoco se restringe el "
        "análisis al invierno austral, a diferencia de Herrera-Marín et al. (2026), "
        "quienes modelan inicios de racha en mayo–agosto. El objeto de estudio es "
        "la duración de cada racha a lo largo del año, no el máximo anual ni la tasa "
        "de onsets.",
    )

    add_figure(
        doc,
        FIG / "fig_esquema_dryspell.png",
        "Figura 2. Definición operativa. Un día es húmedo si pr ≥ R y seco si pr < R. "
        "Un dry spell es la racha consecutiva de días secos; su duración t es el número "
        "de días de la racha. En el ejemplo, R = 1 mm día⁻¹ produce un spell de t = 6 "
        "días y otro de t = 2 días, separados por un día húmedo.",
        width=Inches(6.1),
    )

    add_body(
        doc,
        "La comparación histórica entre CR2MET y ALADIN no puede hacerse con el mismo "
        "umbral fijo. Con R = 1 mm día⁻¹ en ambos productos, ALADIN presenta una "
        "fracción espacial media de días húmedos sustancialmente mayor que CR2MET "
        "(30,68 % frente a 20,94 %), coherente con el sesgo de llovizna de los modelos "
        "(Stephens et al., 2010; Petrova et al., 2024). Ese exceso de días húmedos "
        "fragmenta las rachas secas y genera spells artificialmente más cortos y "
        "numerosos. Por ello el análisis principal adopta el criterio i (umbral global "
        "calibrado): CR2MET con R = 1 mm día⁻¹ y ALADIN con R₀ = 5,285 mm día⁻¹, "
        "elegido para igualar la fracción espacial media de wet days de CR2MET a 1 mm "
        "(§2.3). La lógica de emparejar primero la frecuencia de días húmedos antes de "
        "comparar estadísticas de rachas sigue Martinez-Villalobos et al. (2022; "
        "JCLI-D-21-0590).",
    )

    add_body(
        doc,
        "Como sensibilidad se consideran el criterio ii (1 mm día⁻¹ en ambos) y el "
        "criterio iii (calibración local: un R₀(x,y) por píxel, de modo que en cada "
        "celda la fracción temporal de wet days de ALADIN iguale la de CR2MET a 1 mm). "
        "El criterio i es la configuración de los resultados principales; ii y iii "
        "permiten separar cuánto del sesgo histórico se debe a la definición de día "
        "húmedo y cuánto permanece como diferencia estructural del modelo (Tabla 2).",
    )

    add_caption(
        doc,
        "Tabla 2. Criterios de umbral para definir días húmedos. El análisis principal usa el criterio i.",
    )
    add_table(
        doc,
        ["Criterio", "CR2MET", "ALADIN", "Uso"],
        [
            ["i (global)", "1 mm", "R₀ = 5,285 mm (único, Chile)", "Resultados principales"],
            ["ii (mismo umbral)", "1 mm", "1 mm", "Sensibilidad: sesgo sin calibrar"],
            ["iii (local)", "1 mm", "R₀(x,y) por píxel", "Sensibilidad espacial"],
        ],
        col_widths=[3.5, 3.2, 5.5, 5.0],
    )
    doc.add_paragraph()

    add_h(doc, "2.3 Calibración del umbral R₀", size=12)

    add_body(
        doc,
        "La calibración de R₀ busca el umbral tal que el promedio espacial sobre Chile "
        "continental de la fracción temporal de días con pr ≥ R en ALADIN iguale la "
        "fracción correspondiente de CR2MET con R = 1 mm día⁻¹. Si F(R) denota esa "
        "fracción espacial media de wet days, se resuelve F_ALADIN(R₀) = F_CR2MET(1 mm). "
        "En 1980–2014, CR2MET a 1 mm produce F = 20,94 % y ALADIN a 1 mm produce "
        "F = 30,68 % (+9,7 puntos porcentuales). El umbral R₀ = 5,285 mm día⁻¹ "
        "reproduce F = 20,94 % en ALADIN (Figura 3).",
    )

    add_figure(
        doc,
        FIG / "fig03_calibracion_R0_ftau.png",
        "Figura 3. Calibración del umbral de día húmedo en ALADIN frente a CR2MET "
        "(1980–2014). La curva azul es F(R) de ALADIN. El punto rojo es CR2MET a 1 mm "
        "(F = 20,9 %). El cuadrado marca R₀ = 5,285 mm, umbral con el que ALADIN iguala "
        "esa fracción. La diferencia a 1 mm (+9,7 puntos porcentuales) es el exceso de "
        "días húmedos que, sin calibrar, acorta las rachas secas.",
        width=Inches(5.4),
    )

    add_body(
        doc,
        "Esta calibración sigue la lógica de emparejar la frecuencia de días húmedos "
        "antes de comparar estadísticas de rachas o intensidades (Martinez-Villalobos "
        "et al., 2022; JCLI-D-21-0590). R₀ es un paso metodológico previo, no la "
        "pregunta científica central; la evaluación histórica (§3) verifica si, tras "
        "igualar wet days, ALADIN reproduce la distribución de duraciones. La variante "
        "local R₀(x,y) reduce residuos espaciales del sesgo de wet days, pero introduce "
        "un campo de umbrales heterogéneo; por ello se reserva a sensibilidad (§2.7).",
    )

    add_h(doc, "2.4 Construcción de las PDFs", size=12)

    add_body(
        doc,
        "Para caracterizar la distribución de duraciones se construyen PDFs empíricas "
        "de la variable t (días). Se usan 25 bins con bordes aproximadamente uniformes "
        "en log₁₀(t): primero se fijan los extremos como el mínimo y máximo entero de "
        "las duraciones del panel, se generan 26 bordes con np.logspace entre esos "
        "extremos y se redondean a enteros (bordes comunes para todos los productos y "
        "períodos de una misma cuenca, de modo que las figuras sean comparables). En "
        "cada bin, f(t) es el cociente entre el número de spells en el intervalo y el "
        "ancho del bin; la normalización hace que el área bajo la curva sea igual al "
        "número total de spells del pool (no a 1), de modo que cambios en frecuencia "
        "y en forma sean visibles a la vez.",
    )

    add_h(doc, "2.5 Agrupación regional y estadísticas espaciales", size=12)

    add_body(
        doc,
        "Para las PDFs por cuenca se adopta un pool regional: todos los dry spells de "
        "todos los píxeles de la ventana se agrupan en un único conjunto de eventos "
        "(Martinez-Villalobos y Neelin, 2018). El pool aumenta el tamaño muestral y "
        "permite resolver la cola; a cambio, mezcla heterogeneidad espacial dentro de "
        "la caja. Los píxeles no son independientes, de modo que n_spells no debe "
        "leerse como tamaño muestral efectivo para inferencia espacial. El pool se usa "
        "para describir la forma de la distribución, no para inflar la significancia.",
    )

    add_body(
        doc,
        "Para los mapas se calculan, en cada píxel, la duración media y el percentil 99 "
        "de las duraciones (t99), y se reportan las diferencias Δ = ALADIN − CR2MET en "
        "1980–2014. La significancia se evalúa con bootstrap por años (1 000 réplicas, "
        "semilla fija). El procedimiento es el siguiente: (i) identificar spells "
        "completos en la serie diaria de cada píxel; (ii) asignar cada spell al año "
        "calendario de su inicio (los spells que cruzan el 1 de enero permanecen "
        "enteros y se agrupan por año de onset); (iii) formar un diccionario {año: "
        "duraciones} por píxel; (iv) en cada réplica, remuestrear años con reemplazo "
        "(no bloques diarios), concatenar las duraciones de los años sorteados y "
        "recalcular la métrica; (v) construir el IC95 como percentiles 2,5 y 97,5 de "
        "la distribución bootstrap de Δ (no BCa). Una diferencia se considera "
        "significativa al 95 % cuando el IC95 de Δ no incluye el cero.",
    )

    add_h(doc, "2.6 Risk ratios y cambios futuros", size=12)

    add_body(
        doc,
        "El cambio proyectado en la probabilidad de dry spells largos se cuantifica con "
        "el risk ratio",
    )
    add_eq(doc, "(2)    RR(D) = P_fut(t ≥ D) / P_hist(t ≥ D)")
    add_body(
        doc,
        "donde P(t ≥ D) es la fracción de dry spells cuya duración es al menos D días, "
        "evaluada en ALADIN futuro (2040–2074) y ALADIN histórico (1980–2014) con el "
        "mismo R₀ = 5,285 mm día⁻¹. Un RR(D) > 1 indica mayor probabilidad de spells "
        "largos en el futuro. Como el futuro no existe en CR2MET, la comparación se "
        "hace necesariamente dentro del mismo modelo. Las curvas RR(D) y su "
        "interpretación se presentan en los resultados (§4), no aquí: en Métodos solo "
        "se define el estimador.",
        indent=True,
    )

    add_body(
        doc,
        "Los risk ratios se estiman sobre pools por región administrativa, considerando "
        "spells cuyo día de inicio cae entre marzo y noviembre. En Chile central el "
        "verano (diciembre–febrero) es seco de manera climatológica, de modo que las "
        "rachas que empiezan en pleno verano reflejan en gran parte la estación seca "
        "esperada. Mar–nov enfoca las rachas que interrumpen la temporada de lluvias y "
        "la transición, más relevantes para recarga y agricultura. Se calculan curvas "
        "RR versus D e intervalos de confianza al 95 % mediante el mismo bootstrap por "
        "años de inicio descrito arriba: spells identificados en la serie completa, "
        "agrupados por año de onset, remuestreo de años con reemplazo e IC95 por "
        "percentiles 2,5 y 97,5 de la distribución bootstrap de RR (1 000 réplicas). "
        "Se interpreta aumento significativo cuando RR > 1 y el IC95 queda por completo "
        "queda por completo por encima de 1. Cuando la probabilidad histórica de "
        "spells ≥ D es muy baja —típico del sur húmedo para D grandes—, el cociente "
        "puede volverse grande e inestable; un IC95 amplio refleja entonces rareza de "
        "eventos, no necesariamente una señal climática robusta.",
    )

    add_h(doc, "2.7 Sensibilidad a la elección del umbral", size=12)

    add_body(
        doc,
        "Como experimento de sensibilidad, se comparan en Loa, Maule y Biobío cuatro "
        "configuraciones en 1980–2014: CR2MET a 1 mm, ALADIN a 1 mm, ALADIN a R₀ "
        "global y ALADIN a R₀ local. Para cada cuenca se construyen PDFs con el mismo "
        "esquema de binning log (§2.4). Los resultados de este experimento —incluida la "
        "Figura 4— se presentan en §3.2; aquí solo se define el diseño.",
    )

    add_h(doc, "2.8 Tendencias en el histórico de ALADIN", size=12)

    add_body(
        doc,
        "Para verificar si el propio modelo muestra deriva temporal en persistencia "
        "secas (Figura 7), se construye por píxel una serie anual de duración media y "
        "de t99. A diferencia del análisis principal (§2.2), aquí las rachas se "
        "calculan dentro de cada año calendario por separado, de modo que no cruzan el "
        "1 de enero. Solo se usan años con al menos un número mínimo de spells por "
        "píxel (≥ 30 años válidos en 1980–2014). Sobre cada serie anual se ajusta una "
        "regresión lineal ordinaria; la pendiente se reporta en días por década. La "
        "significancia se evalúa con el p-valor bilateral de la correlación de Pearson "
        "entre año y estadístico anual (prueba t con n − 2 grados de libertad). Un "
        "píxel se marca como significativo al 5 % si p < 0,05.",
    )

    add_body(
        doc,
        "En síntesis, la cadena metodológica es: (1) alinear espacialmente CR2MET y "
        "ALADIN; (2) definir wet/dry days y dry spells, excluyendo rachas censuradas "
        "en los bordes del período y permitiendo que las rachas crucen el año "
        "calendario dentro de la ventana; (3) calibrar R₀ para una comparación "
        "histórica equitativa de la frecuencia de días húmedos; (4) construir PDFs "
        "con bins log reproducibles y normalización a n_spells mediante pool regional "
        "por cuenca; (5) mapear media y t99 con significancia por bootstrap de años "
        "de inicio; (6) cuantificar el cambio futuro con RR(D) e IC95, dentro de "
        "ALADIN; y (7) evaluar tendencias intra-anuales en el histórico del modelo. "
        "Las secciones siguientes aplican este marco a la evaluación histórica (§3) y "
        "a las proyecciones SSP5-8.5 (§4).",
    )


def save_both(doc, path_repo, path_dl):
    doc.save(path_repo)
    try:
        doc.save(path_dl)
    except OSError:
        pass


def build_methods_only():
    doc = Document()
    section = doc.sections[0]
    for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, m, Cm(2.5))
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    write_methods(doc)
    add_h(doc, "Referencias citadas en Datos y métodos")
    # subset most relevant to methods plus those named
    for r in REFS:
        add_ref(doc, r)
    save_both(doc, OUT_METHODS, OUT_DL_METHODS)
    return OUT_METHODS


def build_combined():
    doc = Document()
    section = doc.sections[0]
    for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, m, Cm(2.5))
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    src = Document(str(USER_DOC))
    copy_intro(src, doc)
    write_methods(doc)
    add_h(doc, "Referencias")
    for r in REFS:
        add_ref(doc, r)
    save_both(doc, OUT_COMBINED, OUT_DL_COMBINED)
    return OUT_COMBINED


if __name__ == "__main__":
    m = build_methods_only()
    c = build_combined()
    print("Wrote", m)
    print("Wrote", c)
