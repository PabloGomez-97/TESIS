"""Etapa B de los Docs v2: estilos de parrafo y tablas, a partir de la lectura del Doc (read_doc).

Uso: python _v2_stageB.py <json_doc> <paper|supl> [ID=inicio_tabla ...]
  - ID=inicio: tabla ya insertada (vacia) cuyo startIndex es `inicio`: solo se rellena y formatea.
Escribe stageB_<doc>_styles.json y stageB_<doc>_tables.json (listas de requests para update_doc).
Calibracion (probada en el Doc): al borrar el texto del marcador y hacer insertTable en p, la tabla empieza en p+1 y el
parrafo de la celda (r,c) empieza en s + 3 + r*(1+2C) + 2c, con C columnas.
"""
import json, sys
meta = json.load(open('v2_meta.json'))
doc = json.load(open(sys.argv[1]))['content']
which = sys.argv[2]
already = {a.split('=')[0]: int(a.split('=')[1]) for a in sys.argv[3:]}
M = meta[which]
body = doc['tabs'][0]['documentTab']['body']['content']
kinds = {}
for k, t in M['paras']:
    if k in ('title', 'h1', 'h2', 'h3', 'figph') and t:
        kinds[t] = k
STY = {'h1': 'HEADING_1', 'h2': 'HEADING_2', 'h3': 'HEADING_3'}
styles, ph = [], []
for el in body:
    if 'paragraph' not in el:
        continue
    txt = ''.join(e.get('textRun', {}).get('content', '') for e in el['paragraph']['elements'])
    t = txt.rstrip('\n')
    s, e = el['startIndex'], el['endIndex']
    if t.startswith('[[TABLA '):
        ph.append((t[len('[[TABLA '):-2], s, e))
    k = kinds.get(t)
    if k in STY:
        styles.append({'updateParagraphStyle': {'range': {'startIndex': s, 'endIndex': e, 'tabId': 't.0'},
                       'paragraphStyle': {'namedStyleType': STY[k]}, 'fields': 'namedStyleType'}})
    elif k == 'title':
        styles.append({'updateTextStyle': {'range': {'startIndex': s, 'endIndex': e - 1, 'tabId': 't.0'},
                       'textStyle': {'bold': True, 'fontSize': {'magnitude': 16, 'unit': 'PT'}}, 'fields': 'bold,fontSize'}})
    elif k == 'figph':
        styles.append({'updateTextStyle': {'range': {'startIndex': s, 'endIndex': e - 1, 'tabId': 't.0'},
                       'textStyle': {'bold': True, 'backgroundColor': {'color': {'rgbColor': {'red': 1, 'green': 0.95, 'blue': 0.5}}}},
                       'fields': 'bold,backgroundColor'}})
json.dump(styles, open(f'stageB_{which}_styles.json', 'w'), ensure_ascii=False)
print('estilos:', len(styles), '| marcadores de tabla:', [p[0] for p in ph], '| ya insertadas:', list(already))


def table_requests(tid, s, rows, fs):
    """s = startIndex de la tabla (ya insertada). Devuelve (rellenar en orden inverso, formato)."""
    R, C = len(rows), len(rows[0])
    stride = 1 + 2 * C
    pos = lambda r, c: s + 3 + r * stride + 2 * c
    fill = [{'insertText': {'text': rows[r][c], 'location': {'index': pos(r, c), 'tabId': 't.0'}}}
            for r in range(R) for c in range(C) if rows[r][c] != '']
    fill.reverse()
    fmt, shift = [], 0
    for r in range(R):
        for c in range(C):
            start = pos(r, c) + shift
            n = len(rows[r][c])
            if r == 0 and n:
                fmt.append({'updateTextStyle': {'range': {'startIndex': start, 'endIndex': start + n, 'tabId': 't.0'},
                            'textStyle': {'bold': True}, 'fields': 'bold'}})
            shift += n
    total = shift
    end = s + 1 + R * stride + total  # fin aproximado de la tabla (inicio de fila + celdas)
    fmt.insert(0, {'updateTextStyle': {'range': {'startIndex': s + 1, 'endIndex': end, 'tabId': 't.0'},
                   'textStyle': {'fontSize': {'magnitude': fs, 'unit': 'PT'}}, 'fields': 'fontSize'}})
    return fill, fmt


reqs = []
per = []
items = [(tid, s, e, None) for tid, s, e in ph] + [(tid, None, None, st) for tid, st in already.items()]
# orden inverso por posicion en el documento
def keypos(x):
    return x[1] if x[1] is not None else x[3]
for tid, s, e, st in sorted(items, key=keypos, reverse=True):
    rows, fs = M['tables'][tid]['rows'], M['tables'][tid]['fs']
    if st is None:
        reqs.append({'deleteContentRange': {'range': {'startIndex': s, 'endIndex': e - 1, 'tabId': 't.0'}}})
        reqs.append({'insertTable': {'rows': len(rows), 'columns': len(rows[0]), 'location': {'index': s, 'tabId': 't.0'}}})
        tstart = s + 1
    else:
        tstart = st
    fill, fmt = table_requests(tid, tstart, rows, fs)
    pre = reqs[-2:] if st is None else []
    reqs += fill + fmt
    per.append((tid, (pre if st is None else []) + fill + fmt))
json.dump(reqs, open(f'stageB_{which}_tables.json', 'w'), ensure_ascii=False)
for tid, rq in per:
    json.dump(rq, open(f'stageB_{which}_tbl_{tid}.json', 'w'), ensure_ascii=False)
print([(t, len(r)) for t, r in per])
print('requests de tablas:', len(reqs), '| caracteres:', len(json.dumps(reqs, ensure_ascii=False)))
