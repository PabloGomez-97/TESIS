from pathlib import Path
import json,hashlib,time
R=Path(__file__).resolve().parent;root=Path(json.loads((R/'configuracion.json').read_text())['repo_origen'])
expected={}
for line in (R/'datos_sha256_registrados.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'):continue
    digest,name=line.split(maxsplit=1);name=Path(name).name
    expected[('pr/' if name.startswith('CR2MET') else 'pr1/')+name]=digest
actual={};t=time.monotonic()
for rel in expected:
    h=hashlib.sha256()
    with (root/rel).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    actual[rel]=h.hexdigest()
result={'archivos':len(actual),'passed':actual==expected,'diferencias':[r for r in expected if actual[r]!=expected[r]],'seconds':time.monotonic()-t}
(R/'integridad_datos.json').write_text(json.dumps(result,indent=2));print(result)
raise SystemExit(0 if result['passed'] else 1)
