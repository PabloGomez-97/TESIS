"""Compara el texto de un Doc v2 (leido con read_doc) con los parrafos generados. Uso: python _v2_check.py <json_doc> <paper|supl>"""
import json, sys
doc = json.load(open(sys.argv[1]))['content']
which = sys.argv[2]
meta = json.load(open('v2_meta.json'))[which]
exp = [t for _, t in meta['paras']]
body = doc['tabs'][0]['documentTab']['body']['content']
got = []
for el in body:
    if 'paragraph' in el:
        got.append(''.join(e.get('textRun', {}).get('content', '') for e in el['paragraph']['elements']).rstrip('\n'))
    # las tablas ya insertadas se omiten
# alinear: quitar parrafos de tabla (marcadores ya reemplazados) y el parrafo final vacio
exp_cmp = [t for k, t in meta['paras']]
if len(got) and got[-1] == '': got = got[:-1]
print('esperados', len(exp_cmp), 'leidos', len(got))
bad = 0
n = min(len(exp_cmp), len(got))
for i in range(n):
    if exp_cmp[i] != got[i]:
        bad += 1
        a, b = exp_cmp[i], got[i]
        j = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        print(f'#{i} difiere en car. {j}: esperado …{a[max(0,j-30):j+40]!r} | leido …{b[max(0,j-30):j+40]!r}')
print('diferencias:', bad)
