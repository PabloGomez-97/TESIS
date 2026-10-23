"""Bloques del Material Suplementario v2 (propuesta «Chile central»): mismo esquema que v1, con cifras tomadas de las salidas.

Base: v1 con las sugerencias aceptadas (base/supl_v1_acc.json). Tipos de bloque como en _v2_paper.py.
"""
import json
from pathlib import Path

from _v2_common import *

V1 = json.load(open(Path(__file__).parent / 'base' / 'supl_v1_acc.json'))
K = KEYS


def base(i):
    return V1[i][2].rstrip('\n')


def sub(text, old, new, n=1):
    assert text.count(old) == n, (old[:70], text.count(old))
    return text.replace(old, new)


# ------------------------------------------------------------------ tablas
def tabla_S1():
    rows = [['Criterio', 'Métrica', 'CR2MET (días)', 'ALADIN (días)', 'Δ medio (días)', 'Δ mediano (días)',
             '% celdas ALADIN < CR2MET', '% celdas ALADIN > CR2MET', 'Celdas']]
    lab = {'i': 'i: R₀ global (5,285 mm)', 'ii': 'ii: 1 mm en ambos', 'iii': 'iii: R₀ local'}
    for r in tS1.itertuples(index=False):
        crit = r[0].split(' ')[0]
        rows.append([lab[crit], r[1], fnum(r[2]), fnum(r[3]), fnum(r[4]), fnum(r[5]), fnum(r[6]), fnum(r[7]), fint(r[8])])
    return rows


def tabla_S1b():
    rows = [['Mínimo de rachas', 'Celdas', 'Δ media (días)', 'Δ t99 (días)', 'Δ mediano t99 (días)',
             '% celdas ALADIN < CR2MET, media', '% celdas ALADIN < CR2MET, t99']]
    for r in minr.itertuples():
        rows.append([('30 (análisis principal)' if r.minimo_rachas == 30 else str(r.minimo_rachas)), fint(r.celdas),
                     fnum(r.delta_media_promedio), fnum(r.delta_t99_promedio), fnum(r.delta_t99_mediana),
                     fnum(r.pct_media_ALADIN_menor_sig), fnum(r.pct_t99_ALADIN_menor_sig)])
    return rows


def tabla_S2():
    rows = [['Región', 'Serie', 'n', 'Media (días)', 'Mediana (días)', 'p90 (días)', 'p99 (días)', 'Máximo (días)']]
    ser = {'CR2MET 1 mm': 'CR2MET 1 mm', 'ALADIN 1 mm': 'ALADIN 1 mm', 'ALADIN R0 global': 'ALADIN R₀ global',
           'ALADIN R0 local': 'ALADIN R₀ local'}
    for k in K:
        for s_, lab in ser.items():
            r = res_s[(res_s.cuenca == k) & (res_s.serie == s_)].iloc[0]
            rows.append([LAB[k], lab, fint(r.n_spells), fnum(r.media_dias), fint(r.mediana_dias), fint(r.p90_dias),
                         fint(r.p99_dias), fint(r.max_dias)])
    return rows


def tabla_S3():
    rows = [['Región', 'R₀ global: N_hist → N_fut', 'R₀ global: RR(20) [IC95]', '1 mm: N_hist → N_fut', '1 mm: RR(20) [IC95]']]
    for k in K:
        a, b = rr_row('main', k), rr_row('mm1', k)
        rows.append([LAB[k], f"{fint(a['N_hist'])} → {fint(a['N_fut'])}", f"{fnum(a['RR observado'], 2)} {ci(a['IC95 inferior'], a['IC95 superior'])}",
                     f"{fint(b['N_hist'])} → {fint(b['N_fut'])}", f"{fnum(b['RR observado'], 2)} {ci(b['IC95 inferior'], b['IC95 superior'])}"])
    return rows


def tabla_S4():
    rows = [['', 'Interpolación lineal', 'Promedio por área']]
    rows.append(['Fracción de días húmedos CR2MET (≥ 1 mm)', fnum(r0json['lineal']['fwet_cr2_1mm_pct'], 3) + ' %', fnum(r0json['area']['fwet_cr2_1mm_pct'], 3) + ' %'])
    rows.append(['R₀ calibrado (mm día⁻¹)', fnum(cal['R0_usado_mm'], 3), fnum(r0json['area']['R0'], 3)])
    for crit, nombre in (('i', 'i'), ('ii', 'ii'), ('iii', 'iii')):
        for met, mn in (('media', 'duración media'), ('t99', 't99')):
            a = mapas[(mapas.cr2met == 'lineal') & (mapas.criterio == crit) & (mapas.metrica == met)].Delta.iloc[0]
            b = mapas[(mapas.cr2met == 'area') & (mapas.criterio == crit) & (mapas.metrica == met)].Delta.iloc[0]
            rows.append([f'Δ {mn}, criterio {nombre} (días)', fnum(a), fnum(b)])
    return rows


def tabla_S5():
    rows = [['Región', 'N_hist', 'SSP5-8.5: N_fut', 'SSP5-8.5: RR(20)', 'SSP3-7.0: N_fut', 'SSP3-7.0: RR(20)']]
    for k in K:
        a, b = rr_row('main', k), rr_row('s370', k)
        rows.append([LAB[k], fint(a['N_hist']), fint(a['N_fut']), f"{fnum(a['RR observado'], 2)} {ci(a['IC95 inferior'], a['IC95 superior'])}",
                     fint(b['N_fut']), f"{fnum(b['RR observado'], 2)} {ci(b['IC95 inferior'], b['IC95 superior'])}"])
        assert a['N_hist'] == b['N_hist']
    return rows


def tabla_S6():
    rows = [['Región', 'Frecuencia seca, hist → fut (Δ [IC95], pp)', 'ρ hist → fut', 'Δρ [IC95]', 'IC95 de Δρ incluye 0']]
    for k in K:
        c = ctr.loc[LAB[k]]
        rows.append([LAB[k],
                     f"{fnum(100 * c.frec_seca_hist)} → {fnum(100 * c.frec_seca_fut)} %; {fnum(100 * c.frec_seca_delta)} [{fnum(100 * c.frec_seca_delta_IC95_inf)}; {fnum(100 * c.frec_seca_delta_IC95_sup)}]",
                     f"{fnum(c.rho_hist, 3)} → {fnum(c.rho_fut, 3)}",
                     f"{fnum(c.rho_delta, 3)} [{fnum(c.rho_delta_IC95_inf, 3)}; {fnum(c.rho_delta_IC95_sup, 3)}]",
                     'Sí' if c.rho_IC_incluye_0 else 'No'])
    return rows


# ------------------------------------------------------------------ textos modificados
intro = sub(base(1), 'las PDFs históricas por cuenca con cuatro definiciones de umbral', 'las PDFs históricas por región con cuatro definiciones de umbral')
cap_S2 = 'Tabla S2. Estadísticos de duración de las rachas secas por región (1980–2014) para CR2MET con 1 mm y ALADIN con 1 mm (criterio ii), R₀ global (criterio i) y R₀ local (criterio iii). n = número de rachas.'
cap_FS1 = ('Figura S1. Sensibilidad de las PDFs históricas (1980–2014) de duración de rachas secas a la elección del umbral de día húmedo, en las regiones (a) Valparaíso, (b) Metropolitana, (c) O\'Higgins y (d) Maule. '
           'Se comparan CR2MET con 1 mm (negro) y ALADIN con 1 mm (criterio ii, rojo), con R₀ global = 5,285 mm (criterio i, azul) y con R₀ local por píxel (criterio iii, verde). '
           'Ejes logarítmicos; los bins son comunes a las cuatro series de cada región (§2.4) y la integral de cada curva es igual a su número de rachas (n).')
for k in K:  # comprobaciones de las afirmaciones cualitativas
    a1, cr = res_s[(res_s.cuenca == k) & (res_s.serie == 'ALADIN 1 mm')].iloc[0], res_s[(res_s.cuenca == k) & (res_s.serie == 'CR2MET 1 mm')].iloc[0]
    assert a1.n_spells > cr.n_spells and a1.p99_dias < cr.p99_dias and a1.media_dias < cr.media_dias
loc = {k: relrow(k, 'iii (R0 local)') for k in K}
assert loc['Valparaiso'].p99_ALADIN > loc['Valparaiso'].p99_CR2MET and loc['Metropolitana'].p99_ALADIN > loc['Metropolitana'].p99_CR2MET
assert loc['OHiggins'].p99_ALADIN < loc['OHiggins'].p99_CR2MET and loc['Maule'].p99_ALADIN < loc['Maule'].p99_CR2MET
txt_S2 = ('Con 1 mm, ALADIN produce más rachas y más cortas que CR2MET en las cuatro regiones (mayor n y menor p99). La calibración del umbral acerca las PDFs de ALADIN a la de CR2MET; '
          f'con R₀ local el p99 de ALADIN supera al de CR2MET en Valparaíso ({fint(loc["Valparaiso"].p99_ALADIN)} frente a {fint(loc["Valparaiso"].p99_CR2MET)} días) y Metropolitana ({fint(loc["Metropolitana"].p99_ALADIN)} frente a {fint(loc["Metropolitana"].p99_CR2MET)}), '
          f'mientras que en O\'Higgins ({fint(loc["OHiggins"].p99_ALADIN)} frente a {fint(loc["OHiggins"].p99_CR2MET)}) y Maule ({fint(loc["Maule"].p99_ALADIN)} frente a {fint(loc["Maule"].p99_CR2MET)}) la cola de ALADIN sigue siendo más corta.')
r0l = DER['r0_local']
cap_FS2 = sub(base(29), '(82 de 3587)', f'({r0l["sin_solucion"]} de {fint(r0l["celdas"])})')
assert r0l['sin_solucion'] == 82 and r0l['celdas'] == 3587
txt_S3 = (f'El R₀ local tiene una mediana de {fnum(r0l["mediana"])} mm (p10 = {fnum(r0l["p10"])} mm; p90 = {fnum(r0l["p90"])} mm) y es inferior al R₀ global en el {fnum(r0l["pct_menor_global_de_validas"], 0)} % de los píxeles. '
          f'Los valores más altos se concentran en la zona centro-norte (aproximadamente {fnum(abs(r0l["lat_cuartil_superior_p10_p90"][1]), 0)}–{fnum(abs(r0l["lat_cuartil_superior_p10_p90"][0]), 1)} °S) y los más bajos en el extremo norte y en el sur. '
          'Esta heterogeneidad es la razón por la que el análisis principal usa un único umbral global (§2.3): corrige la frecuencia de días húmedos sin introducir un campo de umbrales que varía de un píxel a otro.')
mm = {k: rr_row('mm1', k) for k in K}
mn_, mx_ = min(mm[k]['RR observado'] for k in K), max(mm[k]['RR observado'] for k in K)
Dl = lambda which, k: lista([str(d) for d in sig_D(which, k)])
assert sig_D('mm1', 'OHiggins') == [] and mm['Metropolitana']['IC95 inferior'] <= 1
txt_S4 = (f'Con 1 mm, el RR(20) es mayor que 1 en las cuatro regiones (entre {fnum(mn_, 2)} y {fnum(mx_, 2)}). El IC95 queda por encima de 1 en Valparaíso a D = {Dl("mm1", "Valparaiso")} días, en Metropolitana a D = {Dl("mm1", "Metropolitana")}, '
          f'en Maule a D = {Dl("mm1", "Maule")} y en ningún valor de D en O\'Higgins. En Metropolitana y O\'Higgins el IC95 de RR(20) incluye 1 ({ci(mm["Metropolitana"]["IC95 inferior"], mm["Metropolitana"]["IC95 superior"])} y {ci(mm["OHiggins"]["IC95 inferior"], mm["OHiggins"]["IC95 superior"])}), igual que en el análisis principal. '
          'La señal futura no depende, por lo tanto, de la calibración del umbral.')
cap_S3 = 'Tabla S3. Risk ratio RR(20) = N_fut(t ≥ 20 d) / N_hist(t ≥ 20 d) por región con R₀ global (análisis principal, Tabla 4) y con 1 mm en ALADIN. Entre corchetes, el IC95 bootstrap. Son significativos los aumentos con IC95 completamente sobre 1 (Valparaíso y Maule con 1 mm; Valparaíso con R₀ global).'
cap_FS3 = ('Figura S3. Risk ratio RR(D) = N_fut(t ≥ D) / N_hist(t ≥ D) en función del umbral de duración D para las cuatro regiones, calculado con 1 mm en ALADIN (SSP5-8.5, 2040–2074 frente a 1980–2014). '
           'La banda sombreada es el IC95 bootstrap por años; la línea punteada marca RR = 1.')
dz = lambda: (abs(DER['regrillado']['dif_fwet_pp']), DER['regrillado']['dif_max_delta'], DER['regrillado']['dif_max_regiones'], DER['regrillado']['mediana_centros'])
d_fw, d_dl, d_rg, med_c = dz()
assert d_dl['media'] < 0.4 and d_dl['t99'] < 3
txt_S5 = ('En el análisis principal, CR2MET se interpola linealmente a los centros de celda de ALADIN. Como alternativa, se calculó para cada celda de ALADIN el promedio de las celdas de CR2MET cuyo centro cae dentro de ella '
          f'(mediana de {fnum(med_c, 0)} celdas), y se repitieron la calibración del umbral y los mapas de duración. El promedio por área aumenta la fracción de días húmedos de CR2MET en solo {fnum(d_fw, 3)} puntos porcentuales, por lo que el umbral calibrado prácticamente no cambia. '
          'Las diferencias de duración entre ALADIN y CR2MET varían en menos de 0,4 días en la media y menos de 3 días en el percentil 99, en los tres criterios de umbral. '
          f'Las estadísticas de CR2MET en las cuatro regiones cambian como máximo {fnum(d_rg["media"], 2)} días en la media, {fnum(d_rg["p90"], 0)} días en el p90 y {fnum(d_rg["p99"], 0)} días en el p99. Las conclusiones no dependen del método de regrillado.')
cap_S4 = base(48)
s5, s3m = {k: rr_row('s370', k) for k in K}, {k: rr_row('main', k) for k in K}
assert s5['OHiggins']['RR observado'].__round__(2) == s3m['OHiggins']['RR observado'].__round__(2)
assert all(s5[k]['RR observado'] < s3m[k]['RR observado'] for k in ('Valparaiso', 'Metropolitana', 'Maule'))
assert all(s5[k]['RR observado'] > 1 and s5[k]['IC95 inferior'] <= 1 for k in K)
txt_S6 = ('Los risk ratios se repitieron con el escenario SSP3-7.0 (2040–2074) de la misma simulación ALADIN, manteniendo el período histórico, el umbral R₀ = 5,285 mm día⁻¹ y la restricción a rachas iniciadas entre marzo y noviembre (Tabla S5; Figura S4). '
          'En las cuatro regiones, RR(20) es mayor que 1 en ambos escenarios. '
          f'Con SSP3-7.0 los cambios son algo menores en Valparaíso, Metropolitana y Maule e iguales en O\'Higgins ({fnum(s5["OHiggins"]["RR observado"], 2)} en ambos), y ninguno de los cuatro IC95 de RR(20) queda por encima de 1 (en el análisis principal, Valparaíso sí). '
          'La significancia de valores individuales cambia en los casos cercanos al límite, pero la dirección de la señal es la misma. '
          'Como ambos escenarios aún tienen forzamientos similares a mediados de siglo y se dispone de un solo miembro por escenario, las diferencias entre ellos no deben atribuirse al forzamiento.')
cap_FS4 = ('Figura S4. Risk ratio RR(D) = N_fut(t ≥ D) / N_hist(t ≥ D) en función del umbral de duración D para las cuatro regiones, con ALADIN bajo SSP3-7.0 (2040–2074) frente al histórico (1980–2014), '
           'R₀ = 5,285 mm día⁻¹ y rachas iniciadas entre marzo y noviembre. La banda sombreada es el intervalo de confianza del 95 % por bootstrap por años.')
txt_S7 = sub(base(61), 'en cada región de los risk ratios', 'en cada región')
cap_S6 = base(62)

B = []
add = B.append
add(('h1', 'Material suplementario'))
add(('p', intro))
add(('h2', 'S1. Sensibilidad del sesgo histórico a la elección del umbral'))
add(('p', base(3)))
add(('p', base(5)))
add(('table', 'TS1', tabla_S1(), None, None, 8))
add(('p', base(9)))
add(('p', base(10)))
add(('p', 'Tabla S1b. Sensibilidad del sesgo histórico (criterio i) al mínimo de rachas por celda. Δ = ALADIN − CR2MET, promediado sobre las celdas con al menos ese número de rachas en ambos productos; '
          'los porcentajes corresponden a las celdas donde ALADIN es significativamente más corto (bootstrap por años). El análisis principal usa 30 rachas; los otros mínimos se evaluaron después de conocer ese resultado.'))
add(('table', 'TS1b', tabla_S1b(), None, None, 8))
add(('h2', 'S2. PDFs históricas por región con distintos umbrales'))
add(('p', cap_S2))
add(('table', 'TS2', tabla_S2(), None, None, 8))
add(('fig', 'S1', 'figS1_sensibilidad_umbral_regiones.png', cap_FS1))
add(('p', txt_S2))
add(('h2', 'S3. Mapa de R₀ local'))
add(('fig', 'S2', 'figS2_mapa_R0_local.png', cap_FS2))
add(('p', txt_S3))
add(('h2', 'S4. Robustez de la señal futura al umbral'))
add(('p', base(33)))
add(('p', cap_S3))
add(('table', 'TS3', tabla_S3(), None, None, 9))
add(('fig', 'S3', 'figS_rr_regiones_1mm.png', cap_FS3))
add(('p', txt_S4))
add(('h2', 'S5. Sensibilidad al método de regrillado de CR2MET'))
add(('p', txt_S5))
add(('p', cap_S4))
add(('table', 'TS4', tabla_S4(), None, None, 9))
add(('h2', 'S6. Sensibilidad de los risk ratios al escenario de emisiones'))
add(('p', txt_S6))
add(('p', base(54)))
add(('table', 'TS5', tabla_S5(), None, None, 9))
add(('fig', 'S4', 'figS_rr_regiones_ssp370.png', cap_FS4))
add(('h2', 'S7. Frecuencia seca y persistencia de primer orden'))
add(('p', txt_S7))
add(('p', cap_S6))
add(('table', 'TS6', tabla_S6(), None, None, 9))

if __name__ == '__main__':
    out = []
    for b in B:
        if b[0] == 'table':
            out.append(f'[TABLA {b[1]}]'); out.extend('   ' + ' | '.join(r) for r in b[2])
        elif b[0] == 'fig':
            out.append(f'[FIGURA {b[1]}: {b[2]}]'); out.append(b[3])
        else:
            out.append(f'{b[0].upper() if b[0] != "p" else ""} {b[1]}'.strip())
    Path(__file__).parent.joinpath('v2_supl_preview.txt').write_text('\n\n'.join(out), encoding='utf-8')
    print(len(B), 'bloques')
