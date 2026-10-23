"""Compara un Doc v2 (leido con read_doc) con lo generado: parrafos, tablas (celda a celda), estilos de titulo y de marcadores de figura.
Uso: python _v2_check_full.py <json_doc> <paper|supl>"""
import json, re, sys

doc = json.load(open(sys.argv[1]))['content']
which = sys.argv[2]
meta = json.load(open('v2_meta.json'))[which]
body = doc['tabs'][0]['documentTab']['body']['content']


def ptext(el):
    return ''.join(e.get('textRun', {}).get('content', '') for e in el['paragraph']['elements']).rstrip('\n')


got, got_tables = [], []
for el in body:
    if 'paragraph' in el:
        got.append((ptext(el), el['paragraph'].get('paragraphStyle', {}).get('namedStyleType', ''), el))
    elif 'table' in el:
        rows = []
        for r in el['table']['tableRows']:
            row = []
            for c in r['tableCells']:
                row.append(''.join(ptext(p) for p in c['content'] if 'paragraph' in p))
            rows.append(row)
        got_tables.append((len(got), rows))
if got and got[-1][0] == '':
    got = got[:-1]

exp = meta['paras']  # [(tipo, texto)]
ph = re.compile(r'^\[\[TABLA (\w+)\]\]$')
exp_tables_order = [ph.match(t).group(1) for _, t in exp if ph.match(t)]

# parrafos: se ignoran los parrafos vacios que la API deja antes/despues de cada tabla y los marcadores de tabla ya reemplazados
exp_p = [t for _, t in exp if t != '' and not ph.match(t)]
n_ph_left = sum(1 for _, t in exp if ph.match(t) and any(g[0] == t for g in got))
got_p = [g[0] for g in got if g[0] != '' or False]
got_p = [g for g in got_p if not ph.match(g)]
mism = 0
for i in range(max(len(exp_p), len(got_p))):
    t = exp_p[i] if i < len(exp_p) else None
    g = got_p[i] if i < len(got_p) else None
    if t != g:
        mism += 1
        a, b = t or '', g or ''
        j = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        print(f'#{i} difiere en car. {j}: esperado …{a[max(0,j-30):j+40]!r} | leido …{b[max(0,j-30):j+40]!r}')
print(f'parrafos con texto: esperados {len(exp_p)}, leidos {len(got_p)}, diferencias {mism}, marcadores de tabla sin reemplazar {n_ph_left}')

# tablas
print('tablas esperadas (orden):', exp_tables_order, '| tablas en el Doc:', len(got_tables))
tbad = 0
done = [t for t in exp_tables_order]
k = 0
for tid in exp_tables_order:
    want = meta['tables'][tid]['rows']
    if tid_left := (k >= len(got_tables)):
        print(tid, 'sin tabla en el Doc'); continue
    pos, rows = got_tables[k]
    # verificar que la tabla correspondiente es la que sigue (misma dimension)
    if len(rows) != len(want) or any(len(a) != len(b) for a, b in zip(rows, want)):
        print(tid, 'dimension distinta: doc', len(rows), 'x', len(rows[0]), 'esperada', len(want), 'x', len(want[0]))
        tbad += 1
        continue
    k += 1
    d = 0
    for r, (a, b) in enumerate(zip(rows, want)):
        for c, (x, y) in enumerate(zip(a, b)):
            if x != y:
                d += 1
                print(f'  {tid} [{r},{c}] doc={x!r} esperado={y!r}')
    print(tid, f'{len(want)}x{len(want[0])} celdas distintas: {d}')
    tbad += d
print('diferencias en tablas:', tbad)

# estilos: titulos
heads = [(t, s) for t, s, _ in got if s.startswith('HEADING')]
print('titulos (estilo):')
for t, s in heads:
    print('  ', s, t[:90])

# marcadores de figura: fondo amarillo y negrita
nfig = 0
for t, s, el in got:
    if t.startswith('[FIGURA') or t.startswith('[Figura'):
        runs = el['paragraph']['elements']
        ok = all(r.get('textRun', {}).get('textStyle', {}).get('bold') and 'backgroundColor' in r['textRun']['textStyle'] for r in runs if r.get('textRun', {}).get('content', '').strip())
        nfig += 1
        print('  figura', 'OK' if ok else 'SIN ESTILO', t[:80])
print('marcadores de figura:', nfig)
