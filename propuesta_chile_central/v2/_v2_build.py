"""Construye los bloques de insercion (etapa A) de los Docs v2 a partir de _v2_paper.py y _v2_supl.py.

Cada documento se inserta como texto plano por trozos (parrafos separados por salto de linea); las tablas quedan como parrafos
marcador «[[TABLA id]]» y las figuras como «[FIGURA n — pegar aqui: archivo]». La etapa B (estilos y tablas) se genera
leyendo el Doc ya escrito (_v2_stageB.py).
"""
import json
from pathlib import Path

import _v2_paper as P
import _v2_supl as S

HERE = Path(__file__).parent


def paragraphs(B):
    out = []
    for b in B:
        k = b[0]
        if k in ('title', 'h1', 'h2', 'h3', 'p'):
            out.append((k, b[1]))
        elif k == 'fig':
            out.append(('figph', f'[FIGURA {b[1]} — pegar aquí: {b[2]}]'))
            out.append(('p', b[3]))
        elif k == 'table':
            out.append(('tabph', f'[[TABLA {b[1]}]]'))
            if b[4]:
                out.append(('p', b[4]))
    return out


def chunks(paras, limite=11000):
    trozos, actual = [], []
    n = 0
    for k, t in paras:
        actual.append(t)
        n += len(t) + 1
        if n >= limite and k in ('p', 'figph', 'tabph') and t != '':
            trozos.append(actual); actual, n = [], 0
    if actual:
        trozos.append(actual)
    return trozos


meta = {}
for nombre, B in (('paper', P.B), ('supl', S.B)):
    paras = paragraphs(B)
    cs = chunks(paras)
    for i, c in enumerate(cs, 1):
        texto = '\n'.join(c) + '\n'
        req = [{'insertText': {'text': texto, 'endOfSegmentLocation': {'tabId': 't.0'}}}]
        (HERE / f'{nombre}_chunk_{i}.json').write_text(json.dumps(req, ensure_ascii=False), encoding='utf-8')
        print(nombre, i, len(texto))
    meta[nombre] = {'paras': paras, 'tables': {b[1]: {'rows': b[2], 'fs': b[5]} for b in B if b[0] == 'table'},
                    'chunks': len(cs), 'total_chars': sum(len(t) + 1 for _, t in paras)}
(HERE / 'v2_meta.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
print({k: (v['chunks'], v['total_chars']) for k, v in meta.items()})
