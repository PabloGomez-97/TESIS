"""Repite este experimento en otra carpeta nueva, sin reutilizar cachés meteorológicas."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destino',type=Path,required=True);a=p.parse_args()
s=Path(__file__).resolve().parent;d=a.destino.resolve()
d.mkdir(parents=True,exist_ok=False);(d/'ejecucion').mkdir();(d/'logs').mkdir()
for name in ('referencia_global','scripts_originales'):shutil.copytree(s/name,d/name)
for pattern in ('*.py','*.json','*.txt','*.md'):
    for f in s.glob(pattern):
        if f.name in ('estado.json','controles_consistencia.json','controles_comparativa.json','INFORME_COMPARATIVO.md','originales_sha256_despues.json','integridad_datos.json','scripts_locales_sha256.json'):continue
        shutil.copy2(f,d/f.name)
for f in (s/'ejecucion').glob('*.py'):shutil.copy2(f,d/'ejecucion'/f.name)
repo=Path(json.loads((d/'configuracion.json').read_text())['repo_origen'])
for name in ('pr','pr1'):(d/'ejecucion'/name).symlink_to(repo/name,target_is_directory=True)
subprocess.run([sys.executable,str(d/'verificar_datos.py')],check=True)
subprocess.run([sys.executable,str(d/'ejecutar_local.py')],check=True)
