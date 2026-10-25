"""Repite el experimento local 0,1 mm en un directorio nuevo sin cachés de análisis."""
from pathlib import Path
import argparse,shutil,json,sys,subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destino',type=Path,required=True);a=p.parse_args()
s=Path(__file__).resolve().parent;d=a.destino.resolve();d.mkdir(parents=True,exist_ok=False)
for name in ('ejecucion','logs','comparativa'):(d/name).mkdir()
for name in ('referencia_local_1mm','referencia_global_0p1mm','referencia_global_1mm','scripts_base_local_1mm','revision_precision_float32'):shutil.copytree(s/name,d/name)
for f in s.glob('*.py'):shutil.copy2(f,d/f.name)
for name in ('configuracion.json','datos_sha256_registrados.txt','versiones_anteriores_sha256_antes.json','README.md','NOTA_PRECISION.md'):shutil.copy2(s/name,d/name)
for f in (s/'ejecucion').glob('*.py'):shutil.copy2(f,d/'ejecucion'/f.name)
repo=Path(json.loads((d/'configuracion.json').read_text())['repo_origen'])
for name in ('pr','pr1'):(d/'ejecucion'/name).symlink_to(repo/name,target_is_directory=True)
steps=[[arg.replace(str(s/'ejecucion'),str(d/'ejecucion')).replace(str(s/'controles_consistencia.json'),str(d/'controles_consistencia.json')) for arg in c] for c in json.loads((s/'pasos.json').read_text())]
(d/'pasos.json').write_text(json.dumps(steps,indent=2))
subprocess.run([sys.executable,str(d/'ejecutar.py')],check=True)
