# Mismo análisis del paper con umbral CR2MET = 0,1 mm (R₀ = 3,668 mm) vs. 1 mm (R₀ = 5,285 mm)

Carpeta: `umbral_0p1mm_nacional/`. Se copiaron los 20 scripts de la raíz y solo se cambió el umbral de CR2MET (1,0 → 0,1 mm) y R₀ (5,285 → 3,668 mm, recalibrado con el mismo script). Dominio nacional (3 587 celdas), 4 ventanas, 4 regiones, períodos, semillas, bootstrap y reglas idénticas. Se ejecutó desde datos crudos (`correr_todo.sh`; logs en `logs/`). Mapeo de las sensibilidades: criterio ii = «0,1 mm en ambos»; S4 = «ALADIN a 0,1 mm»; S6 = SSP3-7.0 con R₀ = 3,668. Fig. 2 (esquema) no se regeneró. «Viejo» = salidas de la raíz (= Doc).
Límite: no se replicó con código independiente (solo ejecución desde datos crudos).

## Calibración (Fig. 3)
| | Viejo (1 mm) | Nuevo (0,1 mm) |
|---|---|---|
| F CR2MET | 20,94 % | 23,47 % |
| F ALADIN al mismo umbral | 30,68 % | 40,66 % |
| Exceso de ALADIN | +9,7 pp | +17,2 pp |
| R₀ | 5,285 mm | **3,668 mm** (−1,617 mm; −31 %) |

## Evaluación histórica (§3.1)
| Afirmación del Doc | Con 0,1 mm |
|---|---|
| ALADIN subestima media y cola en las 4 ventanas | **Ya no.** Loa: media −0,3 %, p99 +1,9 % (antes −16,7 % / −21,1 %). Maipo −17,0 / −9,9 %; Maule −21,0 / −16,3 %; Biobío −15,5 / −17,1 % (siguen negativos, p99 menos) |
| Δ medio del dominio −8,2 d; Δ t99 −46,3 d | **Δ medio −0,1 d; Δ t99 +2,1 d** (3 575 celdas, antes 3 526). 61,0 % de celdas con ALADIN menor (sig.) y 12,4 % mayor (antes 66,3 / 7,8 %) |
| Gradiente norte–sur de CR2MET | Se mantiene, pero el contraste es menor: Loa media 43,8 d, p99 303 d (antes 65,6 / 456); Maipo 9,5 / 81; Maule 8,7 / 62; Biobío 5,9 / 35 |
| Sesgo robusto al mínimo de rachas | **Ya no es robusto en el dominio:** Δ media por mínimo 30/50/100/200 = −0,1 / −1,0 / −3,1 / −2,5 d (antes −8,2 / −7,9 / −6,8 / −3,9); Δ t99 = +2,1 / −0,3 / −6,9 / −6,7 d. El signo del promedio nacional depende del mínimo elegido |
| R₀ global no iguala la frecuencia en ventanas | Se mantiene: exceso Maipo +4,1 pp, Maule +4,0 pp, Biobío +2,2 pp, Loa −1,0 pp (antes +4,2, +3,3, +0,9, −0,2) |
| Criterio ii (mismo umbral) da sesgo enorme | Se mantiene: Δ media −21,4 d, Δ t99 −108 d (antes −27,2 / −135) |
| Criterio iii (R₀ local): sesgo casi desaparece en Maipo, persiste en Loa, Maule, Biobío | Cambia: Maipo −2,4 %, Maule −11,0 %, Biobío −11,0 %, Loa −19,3 % en media; **p99 +15,1 % en Maipo y +1,4 % en Maule** (sobreestima) |
| n_spells ALADIN > CR2MET en todas las ventanas | Se mantiene (Loa 82 229 vs 81 422, casi igual) |

## Futuro (§3.2 y risk ratios, SSP5-8.5, D = 20 d)
| Región | Viejo RR [IC95] | Nuevo RR [IC95] | N_hist → N_fut nuevo |
|---|---|---|---|
| Coquimbo | 1,12 [1,01; 1,25] sig. | 1,14 [1,02; 1,28] sig. | 23 911 → 27 308 |
| O'Higgins | 1,15 [0,95; 1,40] no | 1,13 [0,93; 1,39] no | 6 286 → 7 106 |
| La Araucanía | 1,72 [1,10; 2,88] sig. | 1,90 [1,12; 3,69] sig. | 1 542 → 2 933 |
| Los Lagos | 2,15 [0,90; 8,50] no | 2,07 [0,67; 13,94] no | 337 → 697 |

- La conclusión principal se mantiene (Coquimbo y La Araucanía significativas a D = 20; O'Higgins y Los Lagos no). Coquimbo ahora es significativa también a D = 25 (antes no). Los Lagos: IC más ancho porque N_hist baja de 601 a 337.
- Conclusión de las PDFs: n_spells baja y la media sube en las 4 ventanas (se mantiene). Loa: media +5,8 %, p99 +0,7 %; máximo 1 150 → 1 774 d. Biobío: el máximo baja (125 → 120) aunque media y fracción ≥ 20 d suben (3,30 → 4,42 %) (se mantiene).

## Sensibilidades de los RR
| Sensibilidad | Viejo | Nuevo |
|---|---|---|
| **SSP3-7.0** | Coquimbo [0,98; 1,24] y La Araucanía [0,99; 2,47] cruzan 1; Los Lagos RR 2,73 sig. | Coquimbo 1,11 [0,98; 1,26] cruza 1; **La Araucanía 1,66 [1,007; 2,95] sig. (límite inferior 1,007)**; Los Lagos 2,64 [0,88; 19,8] cruza 1. Cambia: la señal de La Araucanía sobrevive al escenario, la de Los Lagos no |
| **S4 (ALADIN a umbral bajo)** | Coquimbo y La Araucanía sig.; Los Lagos RR 0,25 (107 → 27) inestimable | A 0,1 mm ALADIN casi no produce rachas largas: N_hist(≥ 20 d) = 10 670 / 1 648 / **4** / **0**. RR Coquimbo 1,36 sig., O'Higgins 1,64 sig.; La Araucanía RR 36,8 con 4 eventos históricos (no interpretable; 52 réplicas descartadas); Los Lagos no definido (0 eventos). Esta sensibilidad no es informativa en el sur |

## Frecuencia vs agrupamiento (§4)
| Afirmación | Con 0,1 mm |
|---|---|
| Frecuencia húmeda con R₀ fijo cae: −6 % dominio, −10 a −13 % Maipo/Maule/Biobío, **Loa +1,4 %** | Dominio −6,5 %; Maipo −12,8 %; Maule −12,5 %; Biobío −9,9 %; **Loa −1,6 % (cambia de signo)** |
| RR solo por frecuencia cae dentro del IC95 en 31 de 31 casos (N_hist ≥ 100) | **28 de 29.** La excepción: O'Higgins D = 25 (RR solo-frecuencia 1,397 vs IC [0,866; 1,391], por 0,006). Con N_hist ≥ 100 quedan 29 casos (antes 31) |
| Cobertura de Markov a D = 20: 70 % Los Lagos, 89 % La Araucanía | 65 % y 90 % |
| Frecuencia seca mar–nov sube +2,1 a +3,9 pp, IC sin 0 | +2,3 / +3,6 / +4,0 / +2,6 pp (Coquimbo, O'Higgins, La Araucanía, Los Lagos); IC sin 0 en las 4 |
| ρ no cambia (IC95 incluye 0 en las 4; Los Lagos por 0,002) | IC95 incluye 0 en las 4; **Los Lagos [−0,001; 0,046] (por 0,001)** |

## Rachas censuradas
CR2MET Loa 0,73 % (antes 1,07 %); resto ≤ 0,15 %; ALADIN Loa 0,63–0,74 %. Censuradas por encima del p99 de las completas: Loa CR2MET 7 (antes 27), Loa ALADIN 21/17 (antes 40/22). **Nuevo:** CR2MET Biobío 66 (antes 0), ALADIN futuro Maule 27 (antes 39), Maipo 4 (antes 12), Biobío 3 (antes 21).

## Lectura (para discutir con el profesor, no verificada por pares)
1. Se mantienen: la señal de RR en Coquimbo y La Araucanía, la caída de n_spells con alargamiento, la compatibilidad con la menor frecuencia de días húmedos (28/29) y la ausencia de cambio detectable de ρ.
2. Cambian: el sesgo histórico. Con 0,1 mm el R₀ más bajo (3,668 mm) casi elimina el sesgo en Loa y en el promedio nacional, que antes era el resultado central; «ALADIN subestima la cola en las cuatro ventanas» ya no se sostiene, y el promedio nacional depende del mínimo de rachas.
3. Ahora se sostiene algo que antes no: La Araucanía sigue significativa con SSP3-7.0 (límite inferior 1,007, marginal). Pero ahora no se sostiene lo que antes sí con SSP3-7.0 (Los Lagos).
4. S4 (ALADIN a 0,1 mm) queda degenerada en el sur; no sirve como sensibilidad comparable.
5. Reserva: 0,1 mm está cerca de la resolución de un producto interpolado de estaciones; la fracción de días húmedos de CR2MET puede depender de la interpolación (el test de regrillado da F = 23,47 % con interpolación lineal y 24,05 % con área, R₀ 3,668 vs 3,354 mm: R₀ es sensible al regrillado, 8 % de diferencia).
