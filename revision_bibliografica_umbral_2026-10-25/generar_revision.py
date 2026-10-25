"""Registra qué pasaje se leyó; no confunde DOI verificado con método verificado."""
from pathlib import Path
import csv,json,hashlib
R=Path(__file__).resolve().parent
meta=json.loads((R/'metadatos_crossref.json').read_text())
# Paráfrasis breves y localizadores; no se reproducen artículos ni largos extractos.
evidence={
'10.1002/wcc.147':dict(cita='Zhang et al. (2011)',umbral='1 mm',nivel='TEXTO: PDF del autor',localizador='Tabla 1, p. 5 del PDF; discusión de cantidades pequeñas, p. 9 del PDF',fuente='https://www.geos.ed.ac.uk/~ghegerl/assets/WIRES_index.pdf',hallazgo='Define días húmedos ≥1 mm y CDD con días <1 mm. Explica la convención por el subregistro de cantidades pequeñas.',limite='CDD es el máximo, no la distribución de todas las rachas. No demuestra un óptimo físico universal.'),
'10.1029/2005JD006290':dict(cita='Alexander et al. (2006)',umbral='NO ASIGNADO',nivel='METADATOS y texto general',localizador='No se recuperó la tabla de umbrales en esta revisión',fuente='https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2005JD006290',hallazgo='Referencia fundacional de índices globales; DOI y metadatos verificados.',limite='No se cuenta a favor de un umbral sin verificar su definición específica.'),
'10.1038/srep04364':dict(cita='Polade et al. (2014)',umbral='1 mm',nivel='TEXTO: Methods',localizador='Methods → Definition of dry day',fuente='https://pmc.ncbi.nlm.nih.gov/articles/PMC3952143/',hallazgo='Elige 1 mm tras considerar 0,1/0,5 mm y alternativas locales para comparar modelos y observaciones.',limite='Estudia frecuencia de días secos con GCM y grilla de 2°; no valida automáticamente una elección para ALADIN a 12 km.'),
'10.1038/s41586-024-07887-y':dict(cita='Petrova et al. (2024)',umbral='1 mm',nivel='TEXTO: pasaje metodológico de la editorial',localizador='Methods → Longest annual dry spell',fuente='https://www.nature.com/articles/s41586-024-07887-y',hallazgo='Define día seco <1 mm y lo presenta como límite aceptado en índices climáticos; analiza proyecciones de la racha anual más larga.',limite='Excluye casos históricos LAD>300 días; no trasladar su validación al norte hiperárido ni a todas las rachas.'),
'10.1002/ael2.70037':dict(cita='Latoja et al. (2025)',umbral='1 mm',nivel='TEXTO: Methods',localizador='Sección 2 Methods',fuente='https://acsess.onlinelibrary.wiley.com/doi/10.1002/ael2.70037',hallazgo='Con CR2MET en Chile central, describe 1 mm como umbral usual y remite al análisis previo de sensibilidad.',limite='32°–40°S; agregación estacional e hidrológica distinta de nuestras rachas completas.'),
'10.1007/s41748-024-00526-6':dict(cita='Latoja et al. (2024 online; volumen 2025)',umbral='0,1; 1; 2,5; 5; 10 mm',nivel='TEXTO: Methods y Results',localizador='Sección 2.3 y sección 3.1',fuente='https://link.springer.com/article/10.1007/s41748-024-00526-6',hallazgo='Compara cinco umbrales en CR2MET. Identifica 0,1 y 1 mm como muy utilizados y usa 1 mm en varias presentaciones.',limite='Cuenta días secos; semejanza de patrones no garantiza idénticas colas de duración ni RR.'),
'10.1088/1748-9326/ab7d05':dict(cita='Breinl et al. (2020)',umbral='0,1 mm',nivel='TEXTO: PDF en repositorio del autor',localizador='Sección 2.1 Study area and data, p. 2 del artículo',fuente='https://pub.epsilon.slu.se/17421/1/breinl_k_et_al_200819.pdf',hallazgo='Define días secos <0,1 mm y húmedos ≥0,1 mm; estudia extremos de duración y fecha con registros de estaciones.',limite='Contraejemplo claro a una exclusividad de 1 mm. Estaciones y máximos anuales no son nuestro producto grillado y pool de eventos.'),
'10.1016/j.jhydrol.2022.127645':dict(cita='Paton (2022)',umbral='0; 0,1; 1; 2; 5 mm y otras definiciones',nivel='TEXTO: fragmento de métodos indexado por la editorial',localizador='Fixed and variable threshold definitions for dry spells',fuente='https://www.sciencedirect.com/science/article/pii/S0022169422002207',hallazgo='Compara definiciones fijas y alternativas; documenta dependencia de las distribuciones respecto de la definición.',limite='No se accedió al artículo completo; no se extraen resultados numéricos ni se afirma que identifique un ganador universal.'),
'10.1175/JHM-D-15-0100.1':dict(cita='Mandapaka, Qin y Lo (2016)',umbral='0,1 y 1 mm',nivel='TEXTO: fragmento de métodos indexado por la editorial',localizador='Sección 3a Extracting wet and dry spells',fuente='https://journals.ametsoc.org/view/journals/hydr/17/5/jhm-d-15-0100_1.xml',hallazgo='Usa ambos; 0,1 mm es el mínimo de sus datos y añade 1 mm por incertidumbre al medir lluvia débil.',limite='No afirmar que el mínimo de ese conjunto sea la precisión universal de todo producto de precipitación.'),
'10.3354/cr031035':dict(cita='Bärring et al. (2006)',umbral='0,1 y 1 mm observacionales; ajuste del modelo',nivel='RESUMEN: repositorio de la institución de autores',localizador='Abstract',fuente='https://portal.research.lu.se/en/publications/defining-drywet-spells-for-point-observations-observed-area-avera',hallazgo='Compara observaciones puntuales, promedios de área y un RCM; examina umbrales observados y ajustados al modelo.',limite='Respalda distinguir referencia y ajuste del modelo; no prueba nuestro algoritmo de igualdad anual por celda.'),
'10.5194/hess-25-3493-2021':dict(cita='Tabari et al. (2021)',umbral='1 mm',nivel='TEXTO: artículo completo en editorial',localizador='Sección 2.4 Research indicators',fuente='https://hess.copernicus.org/articles/25/3493/2021/',hallazgo='Adopta 1 mm para proyecciones de rachas y explica la coexistencia de 0,1 mm observacional con 1 mm en estudios de modelos.',limite='Parte de la motivación es drizzle en GCM. En nuestro caso este sesgo también se trata mediante calibración local.'),
'10.1175/JCLI-D-11-00498.1':dict(cita='Zolina et al. (2013)',umbral='NO ASIGNADO',nivel='RESUMEN y metadatos',localizador='No se verificó el pasaje de umbral',fuente='https://journals.ametsoc.org/view/journals/clim/26/6/jcli-d-11-00498.1.xml',hallazgo='Es pertinente para distribuciones de rachas; no se recuperó aquí la definición numérica con suficiente claridad.',limite='No inferir el umbral desde una cita de otro artículo; excluido del recuento.'),
'10.1175/JCLI-D-21-0590.1':dict(cita='Leung et al. (2022)',umbral='NO ASIGNADO para la referencia de rachas',nivel='TEXTO: sección 3b; suplemento no recuperado',localizador='Sección 3b y Fig. 2; suplemento DOI 10.1175/JCLI-D-21-0590.s1',fuente='https://escholarship.org/content/qt0q62r1tc/qt0q62r1tc_noSplash_1e7d6166d9fc639da1260db1ec86ddee.pdf?t=rqxyay',hallazgo='El diagnóstico de rachas emplea umbral dependiente del modelo y remite el detalle al suplemento.',limite='El 1 mm de la sección de frentes corresponde a otro diagnóstico: no usarlo como prueba del umbral de rachas.'),
'10.5194/nhess-21-1051-2021':dict(cita='Fall et al. (2021)',umbral='1 mm',nivel='TEXTO: artículo completo en editorial',localizador='Secciones 2.3 y 2.3.1',fuente='https://nhess.copernicus.org/articles/21/1051/2021/',hallazgo='Usa 1 mm y discute también 0,1 mm al comparar rachas entre observaciones, satélites y reanálisis.',limite='No adoptar como ley universal la afirmación citada allí sobre evaporación de lluvias menores de 1 mm.')
}
rows=[]
for m in meta:
 d=m['doi_solicitado'];e=evidence[d]
 rows.append(dict(doi=d,cita=e['cita'],titulo=m.get('titulo',''),revista=m.get('revista',''),doi_verificado=m['estado']=='verificado',**{k:v for k,v in e.items() if k!='cita'}))
with (R/'matriz_evidencia.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(R/'matriz_evidencia.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
# Bibliografía verificada por DOI. Año de volumen del artículo Springer: 2025 (online en 2024).
bib=[]
for m in meta:
 if m['estado']!='verificado':continue
 doi=m['doi_solicitado'];year=2025 if doi=='10.1007/s41748-024-00526-6' else m['publicado'][0][0]
 key='ref_'+doi.replace('/','_').replace('.','_').replace('-','_')
 fields={'title':m['titulo'],'author':' and '.join(m['autores']),'journal':m['revista'],'year':str(year),'volume':m.get('volumen'),'number':m.get('numero'),'pages':m.get('paginas') or m.get('articulo'),'doi':doi,'url':'https://doi.org/'+doi}
 bib.append('@article{'+key+',\n'+',\n'.join('  '+k+' = {'+str(v)+'}' for k,v in fields.items() if v)+'\n}')
(R/'referencias_verificadas.bib').write_text('\n\n'.join(bib),encoding='utf-8')

def table(data):
 return '| Referencia y revista | Umbral y aporte | Lectura y localizador | DOI |\n| --- | --- | --- | --- |\n'+'\n'.join('| '+r['cita']+' — '+r['revista']+' | '+r['umbral']+'. '+r['hallazgo']+' | '+r['nivel']+'; '+r['localizador']+' | ['+r['doi']+'](https://doi.org/'+r['doi']+') |' for r in data)
classified=[r for r in rows if not r['umbral'].startswith('NO ASIGNADO')]
unassigned=[r for r in rows if r['umbral'].startswith('NO ASIGNADO')]
text='''# Elección de día húmedo: revisión dirigida de 0,1 frente a 1 mm/día

Fecha de consulta: 25 de octubre de 2026. Investigación separada de los análisis numéricos; no modifica resultados ni Docs del paper.

## Conclusión para la decisión

**Recomiendo CR2MET ≥1 mm/día como definición principal de día húmedo, con calibración histórica de ALADIN por celda y campo fijo para el futuro. Mantendría 0,1 mm como sensibilidad explícita.** Esta es una recomendación para la pregunta meteorológica y comparativa de este paper, no una ley física que declare incorrectas las demás definiciones.

1 mm cuenta con una convención internacional formal en ETCCDI; 0,1 mm también tiene antecedentes científicos sólidos, incluidos estudios de rachas extremas. La formulación defendible es «1 mm es la referencia más estandarizada para índices climáticos diarios», no «todos los papers usan 1 mm» ni «1 mm es universalmente más correcto».

La evidencia decisiva combina el estándar, un trabajo de proyecciones de rachas en Nature y antecedentes con CR2MET en Chile: [ETCCDI](https://etccdi.pacificclimate.org/list_27_indices.shtml), [Zhang et al.](https://doi.org/10.1002/wcc.147), [Petrova et al.](https://doi.org/10.1038/s41586-024-07887-y), [Latoja et al.](https://doi.org/10.1002/ael2.70037). El alcance y las diferencias entre sus métricas se registran en la matriz.

## Qué significa «más repetido» y qué se verificó

Esta es una revisión dirigida, no una revisión sistemática ni un censo bibliométrico. Se buscaron estándares, proyecciones, distribuciones de rachas, comparación entre productos y antecedentes chilenos, incluyendo contraejemplos de 0,1 mm. Se verificaron 14 DOI en Crossref y se separó esa verificación bibliográfica de la lectura del método.

En 11 referencias se identificó la definición en un pasaje o resumen pertinente: seis adoptan 1 mm, una adopta 0,1 mm y cuatro comparan varias referencias. Ocho se verificaron en texto metodológico, dos en fragmentos de métodos indexados por la editorial y una en el resumen de la institución de autores. Tres quedan sin asignación numérica para esta pregunta. **Estos recuentos describen la selección consultada, no estiman la proporción de toda la literatura.** Una selección dirigida no permite demostrar cuál gana un conteo global de publicaciones.

El respaldo para elegir 1 mm procede de la estandarización y la pertinencia de los antecedentes, no de votar artículos. El estudio chileno que compara cinco umbrales señala que 0,1 y 1 mm se utilizan mucho; no prueba una frecuencia mundial mayor de uno sobre otro ([Latoja et al., online 2024, volumen 2025](https://doi.org/10.1007/s41748-024-00526-6)).

## Estándar y significado operativo

ETCCDI define días húmedos con precipitación ≥1 mm y secos con precipitación <1 mm. Sus índices CWD y CDD usan ese límite. La revisión de Zhang explica la convención también por problemas de subregistro de cantidades pequeñas ([tabla oficial](https://etccdi.pacificclimate.org/list_27_indices.shtml), [revisión](https://doi.org/10.1002/wcc.147)).

Aplicar ese umbral a CR2MET no convierte automáticamente nuestro análisis en el índice CDD: CDD resume la racha máxima, mientras nuestro estudio analiza todas las rachas completas y su distribución. En el texto corresponde decir que se adopta **la convención de clasificación seco/húmedo**, manteniendo explícita la métrica propia.

Con 1 mm, un día con 0,3 mm sigue clasificado como seco; con 0,1 mm interrumpe la racha. «Seco» debe leerse como «bajo el umbral», no como ausencia absoluta de precipitación.

## Referencias con definición identificada

'''+table(classified)+'''

## Referencias que no se usaron para decidir el umbral numérico

'''+table(unassigned)+'''

No se atribuye a estas tres referencias un umbral solo porque su DOI exista o porque otro trabajo las cite. En particular, el diagnóstico de frentes y el de rachas de Leung deben distinguirse; el suplemento no recuperado limita la atribución numérica, no la lectura del diagnóstico general.

## Cómo trasladar la evidencia a este estudio

**Separar dos decisiones.** La primera es el umbral de CR2MET que define el fenómeno. La segunda es el ajuste del umbral de ALADIN para hacer equivalente la frecuencia húmeda histórica por celda. Adoptar 1 mm en CR2MET no implica imponer 1 mm en ALADIN. El antecedente de Bärring respalda considerar umbrales distintos para observación y modelo, pero no demuestra por sí solo que nuestro algoritmo particular sea óptimo ([DOI](https://doi.org/10.3354/cr031035)).

**Priorizar comparabilidad sin afirmar equivalencia física.** 1 mm facilita relacionar la definición con índices climáticos y antecedentes de CR2MET. La explicación de drizzle en GCM es pertinente, pero no debe transferirse sin matices a un RCM de 12 km ni presentarse como la única razón de nuestra referencia: la calibración local ya trata la diferencia de frecuencia ([Polade et al.](https://doi.org/10.1038/srep04364), [Tabari et al.](https://doi.org/10.5194/hess-25-3493-2021)).

**Conservar 0,1 mm como una alternativa real.** Su aplicación a rachas extremas está verificada en Breinl, no es un umbral inválido. Sería defendible como principal si la pregunta priorizara la interrupción por cantidades muy pequeñas y hubiera respaldo para su representación en el producto de referencia ([DOI](https://doi.org/10.1088/1748-9326/ab7d05)). Que un instrumento o un registro tenga resolución de 0,1 mm no prueba esa exactitud en CR2MET después de interpolar. Este último punto es una inferencia metodológica para nuestros datos, no una conclusión medida por los artículos revisados.

**Reconocer el gradiente y el alcance.** Los antecedentes chilenos citados cubren Chile central, no todo nuestro dominio ni el norte hiperárido. La elección de 1 mm no debe presentarse como físicamente validada en cada clima del gradiente. La sensibilidad ya calculada a 0,1 mm permite mostrar dónde las conclusiones dependen de la definición. La exclusión de zonas hiperáridas en Petrova limita su extrapolación a ese extremo del dominio.

**No elegir mirando significancia o error.** La aparición de Coquimbo como significativo con 0,1 mm no justifica cambiar la definición; su ausencia con 1 mm tampoco la invalida. Al modificar la referencia también cambian las rachas observadas. La pequeña corrección de precisión computacional de los diagnósticos auxiliares no demuestra que 0,1 mm sea científicamente incorrecto: se corrigió y es una cuestión distinta.

## Justificación propuesta en español

> Se adoptó 1 mm/día como umbral de día húmedo en CR2MET, en concordancia con la convención de clasificación utilizada por ETCCDI y con antecedentes de análisis de rachas secas en Chile central. Se evaluó adicionalmente 0,1 mm/día para cuantificar la sensibilidad de los resultados a la precipitación débil. Para cada referencia, el umbral histórico de ALADIN se calibró por celda para igualar la frecuencia anual de días húmedos y se mantuvo fijo en las proyecciones futuras.

Las dos primeras afirmaciones se sustentan en [Zhang et al. (2011)](https://doi.org/10.1002/wcc.147) y [Latoja et al. (2025)](https://doi.org/10.1002/ael2.70037); el resto describe nuestros experimentos. No se afirma que todos los autores adopten 1 mm ni que se hayan validado impactos agrícolas.

## Wording para el paper

> Wet days in CR2MET were defined as days with precipitation of at least 1 mm, following the ETCCDI wet-day convention and previous dry-spell analyses in central Chile. A 0.1 mm reference was also evaluated to assess sensitivity to light precipitation. For each reference threshold, historical ALADIN thresholds were calibrated separately at each grid cell to match the annual wet-day frequency and were held fixed in the future simulations.

Citas para la primera oración: Zhang et al. (2011) y Latoja et al. (2025). El párrafo no pretende que el análisis de todas las rachas sea idéntico al índice anual CDD.

## Argumentos que conviene evitar

- «Toda lluvia menor de 1 mm se evapora»: no es una ley válida para todos los climas, superficies y estaciones.
- «1 mm es el mínimo que puede medir un pluviómetro»: no es una afirmación universal correcta.
- «0,1 mm es la precisión de CR2MET porque es la precisión de algunos instrumentos»: no está demostrado por estas fuentes.
- «1 mm es obligatorio por la OMM para cualquier aplicación»: el respaldo es una convención de índices concretos.
- «Más regiones significativas demuestran que el umbral es mejor»: confunde selección metodológica e inferencia.
- «Nuestra muestra demuestra qué porcentaje de todos los papers usa cada umbral»: no se hizo un muestreo bibliométrico representativo.

## Archivos de evidencia

- `matriz_evidencia.csv` y `.json`: definición, nivel de lectura, pasaje, enlace y límites de cada referencia.
- `metadatos_crossref.json`: títulos, autores, revista, año, volumen, DOI y enlaces verificados.
- `referencias_verificadas.bib`: bibliografía de los 14 DOI; las tres referencias sin asignación no respaldan la elección numérica.
- `verificar_doi.py`: consulta reproducible de metadatos; no verifica automáticamente el contenido metodológico.
- `consultas_y_alcance.md`: búsquedas y criterio de revisión.

El artículo de Latoja en Earth Systems and Environment fue publicado online el 21 de noviembre de 2024 y aparece en el volumen 9 de 2025. Se registran ambos datos para evitar confundirlo con el artículo de Agricultural & Environmental Letters de 2025.
'''
(R/'REVISION_UMBRAL_DIA_HUMEDO_0p1_VS_1mm.md').write_text(text,encoding='utf-8')
assert len(rows)==14 and len(classified)==11 and len(unassigned)==3
assert all(r['doi_verificado'] for r in rows)
print(dict(DOI_verificados=len(rows),definiciones_identificadas=len(classified),sin_asignar=len(unassigned),informe=str(R/'REVISION_UMBRAL_DIA_HUMEDO_0p1_VS_1mm.md')))
