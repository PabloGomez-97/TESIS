from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, PageBreak
from PIL import Image as PILImage

R = Path('/Users/pablogomez/Documents/GitHub/TESIS')
OUT = R / 'entregables' / 'Resumen_para_Cristian.pdf'
OUT.parent.mkdir(exist_ok=True)

AZUL = colors.HexColor('#1F4E79')
GRIS = colors.HexColor('#555555')
CAJA = colors.HexColor('#EAF1F8')
AMBAR = colors.HexColor('#FFF4DC')

base = ParagraphStyle('b', fontName='Helvetica', fontSize=10.3, leading=14.6, alignment=TA_LEFT, spaceAfter=6)
h1 = ParagraphStyle('h1', parent=base, fontName='Helvetica-Bold', fontSize=21, leading=25, textColor=AZUL, spaceAfter=4)
sub = ParagraphStyle('sub', parent=base, fontSize=9.5, textColor=GRIS, spaceAfter=12)
h2 = ParagraphStyle('h2', parent=base, fontName='Helvetica-Bold', fontSize=13, leading=17, textColor=AZUL, spaceBefore=10, spaceAfter=5)
cap = ParagraphStyle('cap', parent=base, fontName='Helvetica-Oblique', fontSize=8.6, leading=11.5, textColor=GRIS, spaceAfter=10)
bul = ParagraphStyle('bul', parent=base, leftIndent=14, bulletIndent=2, spaceAfter=4)
cel = ParagraphStyle('cel', parent=base, fontSize=9, leading=12, spaceAfter=0)
celb = ParagraphStyle('celb', parent=cel, fontName='Helvetica-Bold')


def img(path, ancho_cm):
    w, h = PILImage.open(path).size
    return Image(str(path), width=ancho_cm * cm, height=ancho_cm * cm * h / w)


def caja(textos, fondo=CAJA, borde=AZUL):
    t = Table([[[Paragraph(x, base) for x in textos]]], colWidths=[17 * cm])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), fondo), ('BOX', (0, 0), (-1, -1), 0.8, borde),
                           ('LEFTPADDING', (0, 0), (-1, -1), 10), ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                           ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    return t


def pie(c, d):
    c.saveState()
    c.setFont('Helvetica', 8)
    c.setFillColor(GRIS)
    c.drawString(2 * cm, 1.2 * cm, 'Pablo Gómez · Resumen para Cristian Martínez · 19 de octubre de 2026')
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f'{d.page}')
    c.restoreState()


s = []
s.append(Paragraph('Un hallazgo que puede mejorar el paper y la tesis', h1))
s.append(Paragraph('Para: Cristian Martínez-Villalobos &nbsp;·&nbsp; De: Pablo Gómez &nbsp;·&nbsp; 19 de octubre de 2026', sub))
s.append(Paragraph(
    'Profe: mientras aplicaba sus comentarios al paper, quise comprobar algo que me daba vueltas: si el aumento de rachas secas '
    'largas que proyecta ALADIN se debe a que los días secos se agrupan de otra manera, o simplemente a que hay menos días húmedos. '
    'Encontré algo que creo que nos ayuda a mejorar tanto el paper como la tesis, y se lo quiero contar de la forma más simple posible.', base))
s.append(caja(['<b>En una frase.</b> ALADIN proyecta menos días húmedos en el futuro, y esa caída explica casi por completo por qué '
               'aumentan las rachas secas largas. Con estos datos no hace falta suponer que las rachas se reorganizan en el tiempo.']))

s.append(Paragraph('1. Qué comprobé, paso a paso', h2))
s.append(Paragraph('<b>Primero,</b> miré cuántos días húmedos hay. Con el mismo umbral que usamos en el pasado (R<sub>0</sub> = 5,285 mm), '
                   'la fracción de días húmedos de ALADIN baja de 20,94 % a 19,61 % en 2040-2074 (SSP5-8.5). Por región baja entre 3 % y 17 % '
                   'desde Atacama hacia el sur; en el norte casi no cambia.', base))
s.append(Paragraph('<b>Segundo,</b> simulé el clima de cada celda con una regla muy simple (la probabilidad de que mañana sea seco depende de si hoy lo es), '
                   'cambiando solamente cuántos días secos hay y dejando igual cómo se agrupan. El aumento de rachas largas que predice coincide con el observado '
                   'en las 19 unidades que analicé (14 regiones, 4 macrozonas y el dominio completo), dentro del intervalo de confianza.', base))
s.append(Paragraph('<b>Tercero,</b> hice una prueba de control: repetí la comparación imponiendo que el futuro tenga los mismos días húmedos que el pasado, región por región. '
                   'El análisis principal no cambia: sigue usando el mismo umbral (R<sub>0</sub> = 5,285 mm) en ambos periodos. Con esta prueba de control, el aumento de rachas largas casi desaparece.', base))

s.append(KeepTogether([img(R / 'paper_figuras' / 'fig10_frecuencia_vs_agrupacion.png', 12.0),
    Paragraph('Figura 1. Arriba: cambio en la fracción de días húmedos por región (con el umbral fijo). Abajo: aumento de las rachas de 20 días o más '
              '(RR = rachas futuras / rachas pasadas; sobre 1 hay aumento). Círculos negros: lo observado. Cuadrados: lo esperado si solo cambia la frecuencia. '
              'Triángulos azules: prueba de control, con el futuro teniendo los mismos días húmedos que el pasado. Las líneas verticales son intervalos de confianza del 95 %.', cap)]))

tabla = [[Paragraph('Zona', celb), Paragraph('RR observado (umbral fijo)', celb), Paragraph('RR en la prueba de control (misma frecuencia)', celb)],
         ['Norte árido', '1,01 [0,91 - 1,12]', '1,01 [0,92 - 1,13]'],
         ['Norte semiárido', '1,08 [1,00 - 1,17]', '1,08 [1,00 - 1,17]'],
         ['Centro mediterráneo', '1,20 [1,00 - 1,43]', '1,02 [0,85 - 1,21]'],
         ['Sur templado', '1,73 [1,16 - 2,66]', '1,26 [0,83 - 2,00]'],
         ['Dominio (14 regiones)', '1,08 [1,00 - 1,17]', '1,05 [0,97 - 1,13]']]
t = Table(tabla, colWidths=[5.2 * cm, 5.9 * cm, 5.9 * cm])
t.setStyle(TableStyle([('FONT', (0, 1), (-1, -1), 'Helvetica', 9), ('BACKGROUND', (0, 0), (-1, 0), CAJA),
                       ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#AAAAAA')), ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                       ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
s.append(KeepTogether([t, Paragraph('Tabla 1. RR de rachas de 20 días o más (ALADIN SSP5-8.5 2040-2074 frente a 1980-2014), con intervalo de confianza del 95 % entre corchetes.', cap)]))

s.append(Paragraph('2. Qué significa para el paper y la tesis', h2))
for x in ['<b>Lo que se mantiene:</b> hay más rachas largas desde Atacama hacia el sur (aumento significativo en Coquimbo, Ñuble, Biobío y La Araucanía), '
          'y casi nada en el norte árido. Los números no cambian.',
          '<b>Lo que cambia:</b> la explicación. En vez de "reorganización temporal", el resultado es consistente con "menos días húmedos". '
          'Eso afecta el resumen, la discusión y las conclusiones.',
          '<b>Conecta con su comentario</b> sobre Latoja et al. y Petrova et al.: secarse no implica una misma reorganización de las rachas, '
          'y ahora tenemos evidencia propia al respecto. También va en la línea de Polade et al. (2014, 2017), que destacan el aumento de días secos como parte central del secado en climas subtropicales y mediterráneos.',
          '<b>Es una pregunta más completa:</b> el trabajo ya no dice solo cuánto aumentan las rachas largas, sino por qué.']:
    s.append(Paragraph(x, bul, bulletText='•'))
s.append(Spacer(1, 4))
s.append(caja(['<b>Lo que no puedo afirmar.</b> No puedo decir que no exista reorganización, solo que no hace falta para explicar lo observado. '
               'Es un solo modelo y un solo miembro; la simulación simple da estimaciones puntuales, sin incertidumbre de sus parámetros; '
               'y CR2MET es un producto estadístico de referencia, no una observación directa.'], fondo=AMBAR, borde=colors.HexColor('#C98500')))

s.append(Paragraph('3. Dos cambios de diseño que quiero consultarle', h2))
s.append(Paragraph('Revisando el trabajo noté que usaba dos geografías distintas: cuatro ventanas (Loa, Maipo, Maule y Biobío) para las curvas de duración y cuatro regiones '
                   'administrativas para los risk ratios. Eran lugares diferentes y no tenía una regla clara de selección; además, en los risk ratios quedaba fuera el norte. Propongo:', base))
for x in ['<b>Cuatro macrozonas</b> (Norte árido, Norte semiárido, Centro mediterráneo y Sur templado), formadas por regiones administrativas completas, en lugar de las cuatro ventanas. '
          'Cubren el 93 % de las celdas del análisis, frente al 24 % de las ventanas.',
          '<b>Risk ratios en las 14 regiones</b> del dominio. Las cuatro regiones que ya usaba dan exactamente los mismos valores (por ejemplo, Coquimbo 1,12 y La Araucanía 1,72).']:
    s.append(Paragraph(x, bul, bulletText='•'))
s.append(KeepTogether([img(R / 'paper_figuras' / 'fig_ventanas_vs_macrozonas.png', 5.4),
    Paragraph('Figura 2. Colores: las cuatro macrozonas propuestas. Cajas negras: las cuatro ventanas actuales. Cada punto es una celda de ALADIN (~12 km).', cap)]))
s.append(Paragraph('Revisé que el cambio sea coherente: con las dos geografías, ALADIN acorta las rachas en todas partes, el sesgo en la cola es mayor en el centro, '
                   'y el aumento futuro es pequeño en el norte árido y mayor en el centro y el sur. Por ejemplo, el sesgo de la duración media es de -22 % tanto en Maipo y Maule como en el Centro mediterráneo.', base))

s.append(Paragraph('4. Sus comentarios ya aplicados', h2))
for x in ['<b>Bootstrap independiente</b> entre CR2MET y ALADIN en el sesgo histórico. El resultado casi no cambia: el porcentaje de celdas donde ALADIN es significativamente más corto pasa de 65,3 % a 66,3 % en la media, y de 42,6 % a 43,1 % en el percentil 99.',
          '<b>Regla explícita para los casos con pocas rachas largas</b> en el histórico (N_hist = 0 o muy bajo): se informa la fracción de réplicas con cero, y el RR solo se declara estimable si hay al menos 30 rachas largas en el histórico y ninguna réplica cae en cero.']:
    s.append(Paragraph(x, bul, bulletText='•'))

s.append(Paragraph('5. Lo que le consulto', h2))
for x in ['¿Le parece bien reemplazar las ventanas por macrozonas, y mostrar los risk ratios en las 14 regiones? ¿Pondría Ñuble en el Centro o en el Sur?',
          '¿Le parece bien el nuevo mensaje (más rachas largas, consistentes con menos días húmedos) y reformular el Objetivo 3 como: '
          '"cuantificar el cambio proyectado en la frecuencia de rachas secas largas y determinar qué parte se explica por el cambio en la frecuencia de días húmedos"?',
          '¿Cree que conviene incorporar este resultado también al paper?']:
    s.append(Paragraph(x, bul, bulletText='•'))
s.append(Paragraph('Con su visto bueno, sigo con la reescritura de los capítulos de métodos y resultados, y preparo los cambios para el paper. '
                   'Quedo atento a sus comentarios. Muchas gracias, profe.', base))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title='Un hallazgo que puede mejorar el paper y la tesis', author='Pablo Gómez')
doc.build(s, onFirstPage=pie, onLaterPages=pie)
print('ok', OUT)
