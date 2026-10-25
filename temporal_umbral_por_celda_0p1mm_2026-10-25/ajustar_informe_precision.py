from pathlib import Path
p=Path(__file__).resolve().parent/'generar_comparativa.py';s=p.read_text()
a="ck('Caché de precipitación regrillada sin cambios',compare_file(A/'_test_regrillado/cr2met_lineal_vs_area.npz',B/'_test_regrillado/cr2met_lineal_vs_area.npz')['equal'])"
b="""oldcache=np.load(A/'_test_regrillado/cr2met_lineal_vs_area.npz');newcache=np.load(B/'_test_regrillado/cr2met_lineal_vs_area.npz')
ck('Caché float64 conserva los valores a la precisión float32 previa',all(np.array_equal(oldcache[k],newcache[k].astype(oldcache[k].dtype),equal_nan=True) for k in oldcache.files))
precision={}
for version in ('lin','area'):
    oldvalues=oldcache[version];newvalues=newcache[version]
    changed_days=np.zeros(oldvalues.shape[1],dtype=int)
    for start in range(0,oldvalues.shape[0],365):
        changed_days+=((oldvalues[start:start+365]>=0.1)!=(newvalues[start:start+365]>=0.1)).sum(0)
    precision[version]=dict(dias_celda_reclasificados=int(changed_days.sum()),celdas_afectadas=int((changed_days>0).sum()))
    del oldvalues,newvalues
precision['nota']='Clasificación float64 frente a caché float32 previa a 0,1 mm; se repitieron regrillado, censura y T01.'
(R/'precision_auxiliar.json').write_text(json.dumps(precision,indent=2,ensure_ascii=False))"""
assert a in s;s=s.replace(a,b)
a="`15_regrillado.csv`.')"
b="""`15_regrillado.csv`.\\n\\n'+
f"**Corrección de precisión auxiliar:** el primer control detectó una racha de CR2MET adicional en Biobío en el diagnóstico de censura (357 068 frente a 357 067), debido al redondeo de la caché interpolada a float32 cerca de 0,1 mm. Se conservó esa primera salida en `revision_precision_float32/` y se regeneró la caché en float64, manteniendo pesos, fechas y máscara. En la interpolación lineal, ese redondeo cambia la clasificación de {precision['lin']['dias_celda_reclasificados']} días-celda en {precision['lin']['celdas_afectadas']} celdas. En el promedio alternativo por centros son {precision['area']['dias_celda_reclasificados']} días-celda en {precision['area']['celdas_afectadas']} celdas. Se repitieron regrillado, censura y T01, y se verificaron el campo local y las métricas lineales frente al principal. La calibración, bootstrap, PDFs, Tabla 3, RR y persistencia principales ya usaban la precisión canónica y no se modificaron por esta corrección. Véanse `NOTA_PRECISION.md` y `precision_auxiliar.json`.")"""
assert a in s;s=s.replace(a,b);p.write_text(s)
p=p.parent/'README.md';s=p.read_text();s+='\nSe corrigió una diferencia de precisión de la caché auxiliar cerca de 0,1 mm; la primera ejecución se conserva en `revision_precision_float32/` y la explicación en [NOTA_PRECISION.md](NOTA_PRECISION.md). Los resultados principales no cambiaron por esa corrección.\n';p.write_text(s)
