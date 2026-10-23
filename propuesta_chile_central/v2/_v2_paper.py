"""Bloques de la Introduccion v2 (propuesta «Chile central»): mismo esquema que v1, con cifras tomadas de las salidas.

Base: v1 con las sugerencias aceptadas (base/paper_v1_acc.json). Los parrafos que no dependen de las unidades se copian tal cual;
los demas se modifican con sustituciones verificadas (sub) o se reescriben con f-strings que leen las salidas.
Tipos de bloque: ('title'|'h1'|'h2'|'h3'|'p', texto), ('fig', n, archivo, pie), ('table', id, filas, pie_arriba, pie, fuente_pt).
"""
import json
from pathlib import Path

from _v2_common import *

V1 = json.load(open(Path(__file__).parent / 'base' / 'paper_v1_acc.json'))
SUPL_LINK = 'https://docs.google.com/document/d/1DHlIpq80mnN5artK3y37Ru1RAmmN8QGhJE6yJBNmnwU/edit?usp=sharing'


def base(i):
    return V1[i][2].rstrip('\n')


def sub(text, old, new, n=1):
    assert text.count(old) == n, (old[:70], text.count(old))
    return text.replace(old, new)


def pct_int(x):
    return f'{round(x):d}'


# ------------------------------------------------------------------ cifras que se usan en varios parrafos
K = KEYS
cr_frac = [est.loc[LAB[k], 'CR2MET_fraccion_marnov_pct'] for k in K]
al_frac = [est.loc[LAB[k], 'ALADIN_fraccion_marnov_pct'] for k in K]
exc_R0 = [frec.loc[k, 'exceso_R0_pp'] for k in K]
sesg_i_media = {k: relrow(k, 'i (R0 global)').media_sesgo_rel_pct for k in K}
sesg_i_p99 = {k: relrow(k, 'i (R0 global)').p99_sesgo_rel_pct for k in K}
sesg_iii_media = {k: relrow(k, 'iii (R0 local)').media_sesgo_rel_pct for k in K}
sesg_iii_p99 = {k: relrow(k, 'iii (R0 local)').p99_sesgo_rel_pct for k in K}
cmax = lambda d: max(d, key=lambda k: d[k])
cmin = lambda d: min(d, key=lambda k: d[k])
lat_min, lat_max = abs(DER['unidades_lat_total']['max']), abs(DER['unidades_lat_total']['min'])
cells = {k: DER['unidades'][k]['celdas'] for k in K}
cens_pct = cens.pct_censuradas
bins = list(DER['bins_efectivos'].values())
rr_main = {k: rr_row('main', k) for k in K}
rr_mm1 = {k: rr_row('mm1', k) for k in K}
rr_s370 = {k: rr_row('s370', k) for k in K}
fut_n = {k: DER['tabla3_derivadas'][k] for k in K}
d_n = [fut_n[k]['n_spells_cambio_pct'] for k in K]
d_media = [fut_n[k]['media_cambio_pct'] for k in K]
d_p99 = [fut_n[k]['p99_cambio_pct'] for k in K]
perc = DER['percentil_D20']
perc_vals = [perc[LAB[k]]['percentil'] for k in K]
cens_fut = {k: int(cens_row('ALADIN 2040-2074 (R0)', k).censuradas_sobre_p99) for k in K}
cens_hist_max = max(int(cens_row(s, k).censuradas_sobre_p99) for k in K for s in ('CR2MET 1980-2014 (1 mm)', 'ALADIN 1980-2014 (R0)'))
assert cens_hist_max == 0
ex = DER['celdas_excluidas']
fall = [frec.loc[k, 'cambio_rel_ssp585_pct'] for k in K]
fall_dom = frec.loc['Dominio', 'cambio_rel_ssp585_pct']
mk_ok = mk[mk.N_hist_obs >= 100]
n_ok, n_in = len(mk_ok), int(mk_ok.solo_frec_dentro_IC95_obs.sum())
mk_out = mk_ok[~mk_ok.solo_frec_dentro_IC95_obs]
mk20 = mk[mk.D == 20].set_index('region')
rho_hw = [(ctr.loc[LAB[k], 'rho_delta_IC95_sup'] - ctr.loc[LAB[k], 'rho_delta_IC95_inf']) / 2 for k in K]
fs_delta = [ctr.loc[LAB[k], 'frec_seca_delta'] * 100 for k in K]
assert all(ctr.loc[LAB[k], 'rho_IC_incluye_0'] for k in K) and all(not ctr.loc[LAB[k], 'frec_seca_IC_incluye_0'] for k in K)


def tabla3_rows():
    rows = [['Región', 'Serie', 'n_spells', 'Media', 'Mediana', 'p90', 'p99', 'Máx.', 'Rachas ≥ 20 d (%)']]
    names = {'CR2MET hist': 'CR2MET histórico', 'ALADIN hist': 'ALADIN histórico', 'ALADIN fut': 'ALADIN futuro'}
    for k in K:
        for prod in ('CR2MET hist', 'ALADIN hist', 'ALADIN fut'):
            r = t3row(k, prod)
            rows.append([LAB[k], names[prod], fint(r.n_spells), fnum(r.media), fint(r.mediana), fint(r.p90), fint(r.p99),
                         fint(r['max']), fnum(r.frac_ge20_pct)])
    return rows


def tabla4_rows():
    rows = [['Región', 'N_hist (t≥20)', 'N_fut (t≥20)', 'RR', 'IC95', '¿Aumento sig.?']]
    for k in K:
        r = rr_main[k]
        rows.append([LAB[k], fint(r['N_hist']), fint(r['N_fut']), fnum(r['RR observado'], 2),
                     ci(r['IC95 inferior'], r['IC95 superior']), 'Sí' if r['IC95 inferior'] > 1 else 'No (IC cruza 1)'])
    return rows


def rrtxt(r, nd=2):
    return f"{fnum(r['RR observado'], nd)} {ci(r['IC95 inferior'], r['IC95 superior'])}"


# ------------------------------------------------------------------ parrafos modificados
# --- 1. Introduccion
p1 = sub(base(10),
         'La pregunta no es exclusiva de Chile central: el país abarca desde el desierto de Atacama, con rachas secas de meses o años, hasta el sur húmedo, donde las rachas son mucho más cortas, lo que permite estudiar la persistencia seca a lo largo de un gradiente climático extremo.',
         f'En Chile central la precipitación se concentra en la estación fría: en las cuatro regiones estudiadas, entre el {pct_int(min(cr_frac))} y el {pct_int(max(cr_frac))} % del total anual de CR2MET cae entre marzo y noviembre, de modo que una racha seca prolongada que comienza en esos meses interrumpe la estación en la que se concentra el aporte de agua.')
p_final = (f'En este estudio, centrado en Chile central (regiones de Valparaíso, Metropolitana, O\'Higgins y Maule; {fnum(lat_min)}°–{fnum(lat_max)}°S) '
           'y con la comparación histórica de mapas sobre todo Chile (17,7°S–45,9°S), examinamos cómo cambia la distribución de duraciones de dry spells hacia mediados de siglo bajo SSP5-8.5 y, en particular, la frecuencia de rachas prolongadas. '
           + base(16).split('particular, la frecuencia de rachas prolongadas. ', 1)[1])
# --- 2.1
p21_2 = base(22) + ' La calibración del umbral (§2.3) y los mapas de la comparación histórica (§2.5) usan este dominio completo; las PDFs y los risk ratios, las cuatro regiones de Chile central descritas a continuación.'
p21_4 = (f'Las distribuciones de duración y los risk ratios se calculan en cuatro regiones administrativas contiguas de Chile central, de norte a sur: Valparaíso, Metropolitana de Santiago, O\'Higgins y Maule '
         f'(geometrías de Natural Earth 1:10 m; celdas de ALADIN cuyo centro cae dentro de cada región: {cells["Valparaiso"]}, {cells["Metropolitana"]}, {cells["OHiggins"]} y {cells["Maule"]}, entre {fnum(lat_min)}°S y {fnum(lat_max)}°S), '
         'unidades usadas en la gestión hídrica y agrícola. Las mismas regiones se emplean para describir la forma de las PDFs, en particular su cola, y para cuantificar el cambio en el número de rachas largas (Figura 1; Tabla 1). '
         'Son cuatro casos del centro del país y no una muestra representativa de todo Chile.')
cap1 = (f'Figura 1. Dominio de estudio: Chile entre 17,7°S y 45,9°S. Las áreas sombreadas son las cuatro regiones administrativas de Chile central usadas para las PDFs de duración de dry spells y para los risk ratios: '
        f'Valparaíso ({cells["Valparaiso"]} celdas), Metropolitana ({cells["Metropolitana"]}), O\'Higgins ({cells["OHiggins"]}) y Maule ({cells["Maule"]}).')
# --- 2.2
p22_1 = sub(base(32),
            'En Loa, donde las rachas pueden superar varios años, esta regla evita incluir duraciones truncadas, que subestimarían la cola, a costa de descartar algunas de las rachas más largas: las rachas censuradas representan menos del 1,1 % del total en Loa y menos del 0,2 % en las demás ventanas.',
            f'Esta regla evita incluir duraciones truncadas, que subestimarían la cola, a costa de descartar algunas de las rachas más largas: en las cuatro regiones, las rachas censuradas representan entre {fnum(cens_pct.min(), 2)} y {fnum(cens_pct.max(), 2)} % del total de cada serie.')
p22_2 = sub(base(33), 'En el norte árido esto es esencial, porque las rachas de cientos de días no caben en un año calendario.',
            f'En Chile central también hay rachas de cientos de días (el máximo en CR2MET es de {fint(max(DER["max_cr2met"].values()))} días en Valparaíso; Tabla 3), que no caben en un año calendario.')
assert max(DER['max_cr2met'], key=DER['max_cr2met'].get) == 'Valparaiso'
# --- 2.3
p23_last = sub(base(43), base(43)[base(43).index('con R₀ global la frecuencia queda'):],
               f'con R₀ global la frecuencia de días húmedos queda entre {fnum(min(exc_R0))} y {fnum(max(exc_R0))} puntos porcentuales sobre la de CR2MET en las cuatro regiones, y con R₀ local (criterio iii) el sesgo relativo de la media pasa de entre {fnum(max(sesg_i_media.values()), 0)} y {fnum(min(sesg_i_media.values()), 0)} % con R₀ global a entre {fnum(max(sesg_iii_media.values()), 0)} y {fnum(min(sesg_iii_media.values()), 0)} % (Tabla S2).')
# --- 2.4
p24 = sub(sub(sub(base(45), 'En cada cuenca se usan', 'En cada región se usan'), 'es de 21 a 23 según la cuenca',
              f'es de {min(bins)} a {max(bins)} según la región'), 'x', 'x') if False else None
p24 = base(45)
p24 = sub(p24, 'En cada cuenca se usan', 'En cada región se usan')
p24 = sub(p24, 'es de 21 a 23 según la cuenca', f'es de {min(bins)} a {max(bins)} según la región')
# --- 2.5
p25 = base(48)
p25 = sub(p25, 'Para las PDFs por cuenca se adopta un pool regional: todos los dry spells de todos los píxeles de la ventana',
          'Para las PDFs por región se adopta un pool regional: todos los dry spells de todos los píxeles de la región')
p25 = sub(p25, 'dentro de la caja', 'dentro de la región')
# --- 2.6
p26 = base(55)
p26 = sub(p26, 'meses que concentran entre el 85 % y el 97 % de la precipitación anual en esas regiones según CR2MET y entre el 81 % y el 96 % según ALADIN',
          f'meses que concentran entre el {pct_int(min(cr_frac))} % y el {pct_int(max(cr_frac))} % de la precipitación anual en esas regiones según CR2MET y entre el {pct_int(min(al_frac))} % y el {pct_int(max(al_frac))} % según ALADIN')
p26 = sub(p26, 'porque corresponde aproximadamente al percentil 90 de las duraciones en Chile central (Tabla 3; el p90 de la tabla es de todo el año y de las ventanas, mientras que los risk ratios usan inicio entre marzo y noviembre y regiones administrativas)',
          f'porque, en ALADIN histórico, corresponde a los percentiles {pct_int(min(perc_vals))} a {pct_int(max(perc_vals))} de las duraciones de las rachas con inicio entre marzo y noviembre en las cuatro regiones')
p26 = sub(p26, 'a D = 20 no se descarta ninguna.', 'no se descarta ninguna réplica en ninguna región ni para ningún valor de D.')
assert all(c['Replicas descartadas'] == 0 for k in K for c in curve('main', k))
# --- 2.7
p27 = sub(base(57), 'A escala de cuenca, se comparan las PDFs de Loa, Maipo, Maule y Biobío en 1980–2014',
          'A escala de región, se comparan las PDFs de Valparaíso, Metropolitana, O\'Higgins y Maule en 1980–2014')
# --- 3.1
p31 = sub(base(66), '¿reproduce ALADIN la distribución histórica de duraciones de dry spells?',
          '¿en qué medida reproduce ALADIN la distribución histórica de duraciones de dry spells y su cola en Chile central?')
m = lambda k, prod, c: t3row(k, prod)[c]
p311_1 = (f'CR2MET muestra un contraste norte–sur en la persistencia de rachas secas incluso dentro de Chile central, coherente con diagnósticos recientes de dry spells en la zona central (Latoja et al., 2025; Herrera-Marín et al., 2026). '
          f'En Valparaíso, la región más seca, el pool regional tiene media de {fnum(m("Valparaiso", "CR2MET hist", "media"))} días y p99 de {fint(m("Valparaiso", "CR2MET hist", "p99"))} días, con un máximo de {fint(m("Valparaiso", "CR2MET hist", "max"))} días. '
          f'En el Maule, la más húmeda, las rachas son más cortas y numerosas: la media es {fnum(m("Maule", "CR2MET hist", "media"))} días y el p99, {fint(m("Maule", "CR2MET hist", "p99"))} días. '
          f'Metropolitana y O\'Higgins ocupan el tramo intermedio (medias de {fnum(m("Metropolitana", "CR2MET hist", "media"))} y {fnum(m("OHiggins", "CR2MET hist", "media"))} días; p99 de {fint(m("Metropolitana", "CR2MET hist", "p99"))} y {fint(m("OHiggins", "CR2MET hist", "p99"))} días). '
          'Ese gradiente no es un artefacto de un índice anual: aparece tanto en la media como en la cola (t99) a escala de píxel (Figura 4, columnas CR2MET), donde se ve además el contraste con el norte árido, fuera de las cuatro regiones.')
_pr = {k: est.loc[LAB[k], 'CR2MET_total_mm_anio'] for k in K}
assert _pr['Valparaiso'] == min(_pr.values()) and _pr['Maule'] == max(_pr.values())
assert m('Maule', 'CR2MET hist', 'n_spells') == max(t3row(k, 'CR2MET hist').n_spells for k in K)
p311_2 = (f'Dos implicaciones siguen de inmediato. Primera: un mismo estadístico agregado —por ejemplo, el CDD— comprime climas distintos en un solo máximo por año (Zhang et al., 2011; Sillmann et al., 2013a), '
          f'mientras que aquí el p99 de Valparaíso y el del Maule difieren en un factor cercano a dos ({fint(m("Valparaiso", "CR2MET hist", "p99"))} frente a {fint(m("Maule", "CR2MET hist", "p99"))} días). '
          'Segunda: cualquier evaluación de ALADIN tiene que respetar ese contraste latitudinal, porque un mismo sesgo relativo corresponde a diferencias en días muy distintas en la región seca y en la húmeda.')
assert 1.9 < DER['razon_p99_cr2met_mayor_menor'] < 2.1
# --- 3.1.3
rl = lambda d, f: (cmax(d) if f == 'max' else cmin(d))
p313_1 = (f'Las distribuciones regionales muestran que ALADIN subestima las duraciones en las cuatro regiones (Figura 5; Tabla 3). '
          f'En días, el mayor sesgo está en Valparaíso, donde las rachas son más largas (media {fnum(m("Valparaiso", "ALADIN hist", "media"))} frente a {fnum(m("Valparaiso", "CR2MET hist", "media"))} días; p99 {fint(m("Valparaiso", "ALADIN hist", "p99"))} frente a {fint(m("Valparaiso", "CR2MET hist", "p99"))} días). '
          f'En porcentaje, el déficit de la media va de {fnum(max(sesg_i_media.values()), 0)} % ({LAB[cmax(sesg_i_media)]}) a {fnum(min(sesg_i_media.values()), 0)} % ({LAB[cmin(sesg_i_media)]}) y el del p99 de {fnum(max(sesg_i_p99.values()), 0)} % ({LAB[cmax(sesg_i_p99)]}) a {fnum(min(sesg_i_p99.values()), 0)} % ({LAB[cmin(sesg_i_p99)]}). '
          f'Las medias son {fnum(m("Metropolitana", "ALADIN hist", "media"))} frente a {fnum(m("Metropolitana", "CR2MET hist", "media"))} días en Metropolitana, {fnum(m("OHiggins", "ALADIN hist", "media"))} frente a {fnum(m("OHiggins", "CR2MET hist", "media"))} en O\'Higgins y {fnum(m("Maule", "ALADIN hist", "media"))} frente a {fnum(m("Maule", "CR2MET hist", "media"))} en Maule. '
          f'ALADIN produce, en todas las regiones, más spells que CR2MET (p. ej. {fint(m("Valparaiso", "ALADIN hist", "n_spells"))} frente a {fint(m("Valparaiso", "CR2MET hist", "n_spells"))} en Valparaíso), coherente con rachas algo más cortas y, con R₀ global, '
          f'la frecuencia de días húmedos queda entre {fnum(min(exc_R0))} y {fnum(max(exc_R0))} puntos porcentuales sobre la de CR2MET (la igualdad vale solo para el promedio del dominio).')
assert all(t3row(k, 'ALADIN hist').n_spells > t3row(k, 'CR2MET hist').n_spells for k in K)
assert cmin(sesg_i_media) == 'Valparaiso' and cmax(sesg_i_media) == 'Maule'
assert cmax(sesg_i_p99) == 'Valparaiso' and cmin(sesg_i_p99) == 'OHiggins'
p313_2 = (f'En conjunto, la evaluación histórica muestra que ALADIN subestima las duraciones y la cola en las cuatro regiones. '
          f'Con R₀ local (criterio iii) el sesgo relativo de la media baja a entre {fnum(max(sesg_iii_media.values()), 0)} % y {fnum(min(sesg_iii_media.values()), 0)} %, '
          f'y el del p99 pasa a +{fnum(sesg_iii_p99["Valparaiso"], 0)} % en Valparaíso y +{fnum(sesg_iii_p99["Metropolitana"], 0)} % en Metropolitana (la cola de ALADIN supera entonces a la de CR2MET), '
          f'pero persiste en O\'Higgins ({fnum(sesg_iii_p99["OHiggins"], 0)} %) y Maule ({fnum(sesg_iii_p99["Maule"], 0)} %): en las dos regiones del norte, parte del déficit de cola con el criterio i corresponde a frecuencia residual de días húmedos.')
assert sesg_iii_p99['Valparaiso'] > 0 and sesg_iii_p99['Metropolitana'] > 0 and sesg_iii_p99['OHiggins'] < 0 and sesg_iii_p99['Maule'] < 0
cap3 = ('Tabla 3. Estadísticos de la distribución de duraciones de dry spells por región (criterio i; 1980–2014 y 2040–2074 bajo SSP5-8.5). '
        'n_spells es el número de rachas del pool de la región; media, mediana, p90, p99 y máximo están en días. Los estadísticos se calculan sobre el conjunto de todas las rachas de todas las celdas de la región.')
cap5 = ('Figura 5. PDFs empíricas de duración de dry spells (criterio i, escala log–log; ∫ f dt = n_spells). Negro: CR2MET 1980–2014. Azul: ALADIN histórico. Rojo discontinuo: ALADIN futuro (SSP5-8.5, 2040–2074). '
        'ALADIN histórico subestima el percentil 99 en las cuatro regiones (Tabla 3). Paneles, de izquierda a derecha y de arriba abajo: Valparaíso, Metropolitana, O\'Higgins y Maule. '
        'El futuro se discute en §3.2.2: n_spells disminuye y la cola derecha se extiende en las cuatro regiones.')
# --- 3.2
p32 = 'La segunda pregunta es si, bajo SSP5-8.5, cambia la frecuencia (el número) de rachas prolongadas en Chile central. La comparación es ALADIN histórico versus ALADIN futuro, con el mismo R₀. '
lows = [rr_main[k]['IC95 inferior'] for k in K if rr_main[k]['IC95 inferior'] <= 1]
p321_1 = ('El risk ratio RR(D) = N_fut(t ≥ D) / N_hist(t ≥ D) se estima en pools de las cuatro regiones, con spells de inicio en marzo–noviembre y bootstrap por años (Figura 6; Tabla 4). '
          'A D = 20 días, el número de spells largos pasa de '
          + f'{fint(rr_main["Valparaiso"]["N_hist"])} a {fint(rr_main["Valparaiso"]["N_fut"])} en Valparaíso (RR = {rrtxt(rr_main["Valparaiso"])}), '
          + f'de {fint(rr_main["Metropolitana"]["N_hist"])} a {fint(rr_main["Metropolitana"]["N_fut"])} en Metropolitana (RR = {rrtxt(rr_main["Metropolitana"])}), '
          + f'de {fint(rr_main["OHiggins"]["N_hist"])} a {fint(rr_main["OHiggins"]["N_fut"])} en O\'Higgins (RR = {rrtxt(rr_main["OHiggins"])}) '
          + f'y de {fint(rr_main["Maule"]["N_hist"])} a {fint(rr_main["Maule"]["N_fut"])} en Maule (RR = {rrtxt(rr_main["Maule"])}). '
          + 'Las cuatro estimaciones puntuales son similares y mayores que 1, pero solo en Valparaíso el IC95 queda por completo sobre 1: '
          + f'en Metropolitana, O\'Higgins y Maule el límite inferior queda entre {fnum(min(lows), 2)} y {fnum(max(lows), 2)}, de modo que a D = 20 días no se interpretan como un aumento robusto.')
assert [k for k in K if rr_main[k]['IC95 inferior'] > 1] == ['Valparaiso'] and len(lows) == 3
rr5 = {k: cpt('main', k, 5) for k in K}
assert all(rr5[k]['IC95 superior'] < 1 for k in ('Valparaiso', 'Metropolitana', 'OHiggins')) and rr5['Maule']['IC95 superior'] > 1
assert len({fnum(rr5[k]['RR'], 2) for k in ('Valparaiso', 'Metropolitana', 'OHiggins')}) == 1


def Dlist(which, k):
    return lista([str(d) for d in sig_D(which, k)])


c_v30, c_m30, c_o60, c_ma40 = cpt('main', 'Valparaiso', 30), cpt('main', 'Metropolitana', 30), cpt('main', 'OHiggins', 60), cpt('main', 'Maule', 40)
f2 = lambda c: f"RR = {fnum(c['RR'], 2)}, IC95 {ci(c['IC95 inferior'], c['IC95 superior'])}"
p321_2 = ('La señal no es un desplazamiento uniforme de toda la distribución. '
          + f'A D = 5 días el RR de conteos es {fnum(rr5["Valparaiso"]["RR"], 2)} en Valparaíso, Metropolitana y O\'Higgins (IC95 por debajo de 1 en las tres) y {fnum(rr5["Maule"]["RR"], 2)} en Maule ({ci(rr5["Maule"]["IC95 inferior"], rr5["Maule"]["IC95 superior"])}): '
          + 'hay menos spells cortos, coherente con la caída de n_spells. '
          + f'El aumento crece con D: el IC95 queda por completo sobre 1 en Valparaíso a D = {Dlist("main", "Valparaiso")} días (p. ej. {f2(c_v30)} a D = 30), en Metropolitana a D = {Dlist("main", "Metropolitana")} ({f2(c_m30)} a D = 30), '
          + f'en O\'Higgins solo a D = {Dlist("main", "OHiggins")} ({f2(c_o60)}) y en Maule a D = {Dlist("main", "Maule")} ({f2(c_ma40)} a D = 40). '
          + 'Como se evalúan 40 pares región × D, no se interpreta cada valor aislado sino el comportamiento de la curva: en las cuatro regiones el RR puntual es menor que 1 para duraciones cortas y crece con D (Figura 6). '
          + 'Eso es lo que un CDD anual no muestra: el cambio está en la frecuencia de excedencias a lo largo de la cola, no en un único máximo por año (Sillmann et al., 2013b; Wainwright et al., 2021). '
          + f'Las sensibilidades matizan esta señal: con SSP3-7.0 los cuatro IC95 de RR(20) incluyen 1 (Valparaíso {ci(rr_s370["Valparaiso"]["IC95 inferior"], rr_s370["Valparaiso"]["IC95 superior"])}, Metropolitana {ci(rr_s370["Metropolitana"]["IC95 inferior"], rr_s370["Metropolitana"]["IC95 superior"])}, O\'Higgins {ci(rr_s370["OHiggins"]["IC95 inferior"], rr_s370["OHiggins"]["IC95 superior"])} y Maule {ci(rr_s370["Maule"]["IC95 inferior"], rr_s370["Maule"]["IC95 superior"])}); '
          + f'con 1 mm, Valparaíso (RR = {rrtxt(rr_mm1["Valparaiso"])}) y Maule (RR = {rrtxt(rr_mm1["Maule"])}) son significativos, y Metropolitana y O\'Higgins no (Tablas S3 y S5).')
assert all(rr_s370[k]['IC95 inferior'] <= 1 for k in K)
assert [k for k in K if rr_mm1[k]['IC95 inferior'] > 1] == ['Valparaiso', 'Maule']
rr_lo = min(rr_main[k]['RR observado'] for k in K)
rr_hi = max(rr_main[k]['RR observado'] for k in K)
low_lo, low_hi = min(lows), max(lows)
cap4t = 'Tabla 4. Risk ratio a D = 20 días. RR = N_fut(t ≥ 20) / N_hist(t ≥ 20). ALADIN, R₀ = 5,285 mm, inicio marzo–noviembre. IC95 por bootstrap de años (1 000 réplicas).'
cap6 = ('Figura 6. Risk ratio RR(D) = N_fut(t ≥ D) / N_hist(t ≥ D) de dry spells con duración al menos D días. ALADIN, SSP5-8.5 (2040–2074) respecto de 1980–2014, R₀ = 5,285 mm, inicio marzo–noviembre. '
        'La banda es el IC95 por bootstrap de años. La línea punteada marca RR = 1. '
        f'Solo Valparaíso supera 1 con IC que no cruza 1 a D = 20 días; en Metropolitana, O\'Higgins y Maule el límite inferior queda entre {fnum(low_lo, 2)} y {fnum(low_hi, 2)}. '
        'Los paneles son, de izquierda a derecha y de arriba abajo: Valparaíso, Metropolitana, O\'Higgins y Maule.')
# --- 3.2.2
p322_1 = (f'Las PDFs de la Figura 5, leídas ahora en la curva futura, muestran el cambio de forma y de n_spells. En las cuatro regiones el futuro de ALADIN tiene menos spells que el histórico (n_spells disminuye entre {pct_int(abs(max(d_n)))} y {pct_int(abs(min(d_n)))} %) '
          f'y suben la media (entre {pct_int(min(d_media))} y {pct_int(max(d_media))} %) y el p99 (entre {pct_int(min(d_p99))} y {pct_int(max(d_p99))} %): la cola se extiende hacia duraciones de decenas a cientos de días (Tabla 3; Figura 5). '
          'En escala log–log las curvas no son una traslación paralela: el exceso futuro se concentra a la derecha. '
          f'La exclusión de rachas censuradas afecta más a la cola futura que a la histórica, sobre todo en el Maule ({cens_fut["Maule"]} rachas censuradas superan el p99 de las completas en el futuro, frente a ninguna en el histórico; Metropolitana {cens_fut["Metropolitana"]}, O\'Higgins {cens_fut["OHiggins"]} y Valparaíso {cens_fut["Valparaiso"]}), '
          'por lo que el alargamiento proyectado en el Maule tiende a estar subestimado.')
fr = lambda k: f'de {fnum(t3row(k, "ALADIN hist").frac_ge20_pct)} a {fnum(t3row(k, "ALADIN fut").frac_ge20_pct)} %'
assert all(t3row(k, 'ALADIN fut').p90 > t3row(k, 'ALADIN hist').p90 and t3row(k, 'ALADIN fut').frac_ge20_pct > t3row(k, 'ALADIN hist').frac_ge20_pct for k in K)
p322_2 = ('El máximo muestral es un estadístico ruidoso y no debe usarse solo: por eso la distribución se describe además con la media, el p90, el p99 y la fracción de spells de al menos 20 días, que aumentan en las cuatro regiones '
          f'({fr("Valparaiso")} en Valparaíso, {fr("Metropolitana")} en Metropolitana, {fr("OHiggins")} en O\'Higgins y {fr("Maule")} en Maule; Tabla 3). '
          'Eso ilustra por qué el objeto del trabajo es la distribución y las excedencias, no el récord de un píxel.')
p322_3 = (f'En conjunto, los risk ratios y las distribuciones de duración muestran un alargamiento de los dry spells hacia mediados de siglo en las cuatro regiones: menos spells en total, más spells largos (RR(20) entre {fnum(rr_lo, 2)} y {fnum(rr_hi, 2)}) y una cola futura más extensa. '
          'La evidencia estadística a D = 20 días es más clara en Valparaíso, el único IC95 sobre 1 en el análisis principal, y no resiste el cambio de escenario a SSP3-7.0.')
# --- 4.1
fmin, fmax = fnum(min(exc_R0)), fnum(max(exc_R0))
p41_1 = base(101)
p41_1 = sub(p41_1, 'dentro de las ventanas la frecuencia queda desigual', f'dentro de las regiones la frecuencia queda entre {fmin} y {fmax} puntos porcentuales por encima de la de CR2MET')
p41_1 = p41_1 + ' [VERIFICAR: solo se leyó el resumen de Petrova et al. (2024); la relación entre sesgo histórico y proyección no está confirmada en el cuerpo del trabajo].'
p41_1 = sub(p41_1, 'proyectar aumentos futuros menores. [VERIFICAR', 'proyectar aumentos futuros menores [VERIFICAR')
p41_1 = p41_1.replace('trabajo]..', 'trabajo].')
e20 = lambda k: mk20.loc[LAB[k]]
p41_2 = ('Hacia 2040–2074 bajo SSP5-8.5, ALADIN proyecta menos dry spells, pero más largos: aumenta el número de rachas de 20 días o más '
         f'(RR entre {fnum(rr_lo, 2)} y {fnum(rr_hi, 2)} en las cuatro regiones), con señal significativa solo en Valparaíso en el análisis principal (no con SSP3-7.0; Tabla S5), y las PDFs de las cuatro regiones extienden la cola. '
         'El signo de este cambio es opuesto al que Latoja et al. (2025) documentan con CR2MET en Chile central (32°–40°S) entre 1960 y 2021: más dry spells, pero más cortos, junto con menos precipitación por racha húmeda; las cuatro regiones de este estudio caen dentro de ese dominio. '
         f'Con R₀ fijo, la frecuencia de días húmedos de ALADIN disminuye entre {pct_int(abs(max(fall)))} y {pct_int(abs(min(fall)))} % en las cuatro regiones ({pct_int(abs(fall_dom))} % en el dominio completo), y el aumento de rachas largas es de magnitud compatible con esa caída: '
         f'el RR que predice un modelo de Markov de primer orden que solo cambia la frecuencia cae dentro del IC95 del RR observado en {n_in} de los {n_ok} casos con N_hist ≥ 100 '
         f'(los dos restantes, {mk_out.iloc[0].region} a D = {int(mk_out.iloc[0].D)} y {mk_out.iloc[1].region} a D = {int(mk_out.iloc[1].D)}, quedan por encima del observado; '
         f'a D = 20 el RR esperado solo por frecuencia es {fnum(e20("Valparaiso").RR_solo_frec, 2)}, {fnum(e20("Metropolitana").RR_solo_frec, 2)}, {fnum(e20("OHiggins").RR_solo_frec, 2)} y {fnum(e20("Maule").RR_solo_frec, 2)} frente a {fnum(e20("Valparaiso").RR_obs, 2)}, {fnum(e20("Metropolitana").RR_obs, 2)}, {fnum(e20("OHiggins").RR_obs, 2)} y {fnum(e20("Maule").RR_obs, 2)} observados en Valparaíso, Metropolitana, O\'Higgins y Maule, '
         'de modo que en las tres regiones del sur la frecuencia sola predice un aumento algo mayor que el observado), '
         'y la persistencia de primer orden no cambia de forma detectable (el IC95 de Δρ incluye 0 en las cuatro regiones; Tabla S6). '
         f'La frecuencia de días secos de marzo a noviembre aumenta entre {fnum(min(fs_delta))} y {fnum(max(fs_delta))} pp (Tabla S6). '
         f'No encontramos evidencia de un cambio en la persistencia día a día de primer orden, aunque la potencia es limitada (el IC95 de Δρ mide aproximadamente ±{fnum(max(rho_hw), 2)}). '
         'Los mecanismos no se diagnostican aquí. Parte del contraste puede deberse al forzante: la megasequía combina variabilidad interna con una contribución antrópica (Boisier et al., 2016; Garreaud et al., 2020). '
         'Con un solo miembro, aquí no se separa la señal forzada de la variabilidad interna. También difieren la métrica y la ventana: Latoja et al. analizan la tendencia de la duración media en seis décadas, y aquí se compara la cola entre dos períodos de 35 años. '
         'El cambio en la duración de las rachas no puede inferirse solo del signo del cambio en la precipitación total.')
assert n_ok - n_in == 2 and (mk_out.solo_frec_dentro_IC95_obs == False).all()
assert all(e20(k).RR_solo_frec > e20(k).RR_obs for k in ('Metropolitana', 'OHiggins', 'Maule'))
# --- 4.2
p42_1 = sub(base(104), 'en climas mediterráneos y áridos', 'en climas mediterráneos')
# --- 4.3
l43 = base(107)
l43_head = l43[:l43.index('Cuarta,')]
pre = est.loc
prc = lambda k: f'{fint(est.loc[LAB[k], "ALADIN_total_mm_anio"])} frente a {fint(est.loc[LAB[k], "CR2MET_total_mm_anio"])} mm' if k == 'Valparaiso' else f'{fint(est.loc[LAB[k], "ALADIN_total_mm_anio"])} / {fint(est.loc[LAB[k], "CR2MET_total_mm_anio"])}'
rmin, rmax = DER['precipitacion_razon_rango']['min'], DER['precipitacion_razon_rango']['max']
p43_1 = (l43_head
         + 'Cuarta, las cuatro regiones son contiguas y comparten el forzante de gran escala, de modo que sus resultados no son cuatro estimaciones independientes; además, las unidades no cubren el norte árido ni el sur húmedo, donde el comportamiento puede diferir, aunque los mapas de la comparación histórica incluyen todo el dominio. '
         f'Quinta, con R₀ global la frecuencia de días húmedos de ALADIN queda entre {fmin} y {fmax} puntos porcentuales sobre la de CR2MET en estas regiones, de modo que parte del sesgo de persistencia con el criterio i es ese exceso residual. '
         f'Además, {ex["n"]} de {fint(ex["chile"])} celdas (entre {fnum(abs(ex["lat_max"]))}°S y {fnum(abs(ex["lat_min"]))}°S) no entran en los mapas por tener menos de 30 rachas, y la exclusión de las rachas censuradas elimina las más largas '
         f'(en el futuro de ALADIN, {cens_fut["Maule"]} rachas censuradas del Maule superan el p99 de las completas, frente a ninguna en el histórico; Metropolitana {cens_fut["Metropolitana"]} y O\'Higgins {cens_fut["OHiggins"]}). '
         f'Sexta, R₀ = 5,285 mm iguala frecuencias, no define un día seco físico: ALADIN produce entre {fnum(rmin)} y {fnum(rmax)} veces la precipitación anual de CR2MET en las cuatro regiones '
         f'(Valparaíso {prc("Valparaiso")}; Metropolitana {prc("Metropolitana")}; O\'Higgins {prc("OHiggins")}; Maule {prc("Maule")}), '
         'y Leung et al. (2022) encuentra rachas más largas en los modelos que en las observaciones, el signo opuesto al de este trabajo [VERIFICAR: sin leer el suplemento de Leung et al.]. '
         'Finalmente, el trabajo caracteriza el cambio en la distribución de duraciones de las rachas secas, pero no diagnostica los mecanismos de circulación que la producen.')
assert 'Quinta, en regiones muy húmedas' in l43 and l43.index('Cuarta,') < l43.index('Quinta,')
p43_3 = (f'Los resultados muestran que, hacia 2040–2074 y bajo SSP5-8.5, el modelo proyecta en Chile central rachas secas más prolongadas, de magnitud compatible con la menor frecuencia de días húmedos, '
         'con cambios visibles en la cola de la distribución de las cuatro regiones (significativos a D = 20 días solo en Valparaíso). '
         'Caracterizar la distribución de duraciones aporta una dimensión que un índice basado en un único extremo anual no recoge.')

# ------------------------------------------------------------------ armado
B = []
add = B.append
add(('p', f'Enlace a Material Suplementario v2: {SUPL_LINK}'))
add(('p', ''))
add(('title', 'Duración de las rachas secas en Chile central: evaluación de un modelo regional y cambio proyectado a mediados de siglo bajo SSP5-8.5'))
add(('p', ''))
add(('h1', '1. Introducción'))
add(('p', p1)); add(('p', base(11))); add(('p', base(12))); add(('p', base(13))); add(('p', base(14))); add(('p', base(15))); add(('p', p_final))
add(('p', ''))
add(('h1', '2. Métodos'))
add(('h2', '2.1 Datos'))
add(('p', base(21))); add(('p', p21_2)); add(('p', base(23))); add(('p', p21_4))
add(('fig', 1, 'fig01_dominio_regiones_chile_central.png', cap1))
add(('p', 'Tabla 1. Conjuntos de datos: producto, rol, grilla, período y umbral de día húmedo.'))
add(('table', 'T1', V1[29][2], None, None, 9))
add(('h2', '2.2 Definición de días húmedos/secos y dry spells'))
add(('p', p22_1)); add(('p', p22_2))
add(('fig', 2, 'fig_esquema_dryspell.png', base(35)))
add(('p', base(36)))
add(('table', 'T2', V1[37][2], None, base(38), 9))
add(('h2', '2.3 Calibración del umbral R₀'))
add(('p', base(40)))
add(('fig', 3, 'fig03_calibracion_R0_ftau.png', base(42)))
add(('p', sub(base(43), base(43)[base(43).index('con R₀ global la frecuencia queda'):], p23_last[p23_last.index('con R₀ global la frecuencia de días'):])))
add(('h2', '2.4 Construcción y normalización de las PDFs'))
add(('p', p24)); add(('p', base(46)))
add(('h2', '2.5 Agrupación regional y estadísticas espaciales'))
add(('p', p25)); add(('p', base(49))); add(('p', base(50)))
add(('h2', '2.6 Risk ratios y cambios futuros'))
add(('p', base(52))); add(('p', base(53))); add(('p', base(54))); add(('p', p26))
add(('h2', '2.7 Sensibilidad a la elección del umbral'))
add(('p', p27))
add(('p', ''))
add(('h1', '3. Resultados'))
add(('h2', '3.1 Evaluación histórica (1980–2014)'))
add(('p', p31))
add(('h3', '3.1.1 Climatología de persistencia en CR2MET'))
add(('p', p311_1)); add(('p', p311_2))
add(('h3', '3.1.2 ALADIN reproduce el gradiente espacial, pero subestima la persistencia y la cola'))
add(('p', base(71))); add(('p', base(72)))
add(('fig', 4, 'fig05_mapas_persistencia_criterio_i.png', base(74)))
add(('h3', '3.1.3 PDFs regionales: sesgo de ALADIN por región'))
add(('p', p313_1)); add(('p', p313_2))
add(('table', 'T3', tabla3_rows(), None, cap3, 8))
add(('fig', 5, 'fig_pdf_regiones_loglog_panel.png', cap5))
add(('h2', '3.2 Proyecciones futuras (SSP5-8.5, 2040–2074)'))
add(('p', p32))
add(('h3', '3.2.1 Risk ratios: más dry spells largos'))
add(('p', p321_1)); add(('p', p321_2))
add(('table', 'T4', tabla4_rows(), None, cap4t, 9))
add(('fig', 6, 'fig_rr_regiones.png', cap6))
add(('h3', '3.2.2 PDFs por región: extensión de la cola futura'))
add(('p', p322_1)); add(('p', p322_2)); add(('p', p322_3))
add(('h1', '4. Discusión'))
add(('h2', '4.1 Síntesis e interpretación'))
add(('p', p41_1)); add(('p', p41_2))
add(('h2', '4.2 Implicaciones para evaluación de modelos y riesgo de sequía'))
add(('p', p42_1)); add(('p', base(105)))
add(('h2', '4.3 Limitaciones y perspectivas'))
add(('p', p43_1)); add(('p', base(108))); add(('p', p43_3))
add(('p', ''))
add(('h1', '5. Referencias'))
for i in range(113, 145):
    add(('p', base(i)))

# Leung ocupa tres parrafos en v1 (saltos de linea dentro de la referencia): se unen en uno
refs_start = [i for i, b in enumerate(B) if b[0] == 'h1' and b[1].startswith('5.')][0] + 1
j = [i for i in range(refs_start, len(B)) if B[i][1].startswith('Leung, L. R.')][0]
B[j] = ('p', ' '.join(x.strip() for x in (B[j][1], B[j + 1][1], B[j + 2][1])))
del B[j + 1:j + 3]

if __name__ == '__main__':
    out = []
    for b in B:
        if b[0] == 'table':
            out.append(f'[TABLA {b[1]}] ' + ' | '.join(b[2][0]) + f'  ({len(b[2]) - 1} filas)')
            if b[4]: out.append(b[4])
        elif b[0] == 'fig':
            out.append(f'[FIGURA {b[1]}: {b[2]}]'); out.append(b[3])
        else:
            out.append(f'{b[0].upper() if b[0] != "p" else ""} {b[1]}'.strip())
    Path(__file__).parent.joinpath('v2_paper_preview.txt').write_text('\n\n'.join(out), encoding='utf-8')
    print(len(B), 'bloques; caracteres:', sum(len(x) for x in out))
