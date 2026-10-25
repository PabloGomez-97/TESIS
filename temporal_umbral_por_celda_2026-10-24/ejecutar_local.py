"""Ejecuta la cadena aislada; reanuda únicamente pasos terminados satisfactoriamente."""
from pathlib import Path
import subprocess, sys, json, time, os, hashlib
from datetime import datetime, timezone
R=Path(__file__).resolve().parent; W=R/'ejecucion'
steps=[['calibrar_local.py']]+json.loads((R/'pasos.json').read_text())
# Guardar diagnóstico de bisección global para comparación; es una referencia, no el umbral principal local.
steps += [['diagnostico_regional.py'], ['figura_calibracion_local.py'], ['verificar_consistencia_local.py','--ejecucion',str(W),'--salida',str(R/'controles_consistencia.json')], ['../verificar_datos.py'], ['../generar_comparativa.py']]
p=R/'estado.json'
state=json.loads(p.read_text()) if p.exists() else {'status':'ejecutando','started_utc':datetime.now(timezone.utc).isoformat(),'steps':[]}
def save():p.write_text(json.dumps(state,indent=2,ensure_ascii=False))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');env.pop('PYTHONPATH',None)
for k,command in enumerate(steps):
    if len(state['steps'])>k and state['steps'][k]['status']=='completo' and state['steps'][k]['command']==command: continue
    log=R/'logs'/f'{k:02d}_{Path(command[0]).stem}.log'
    item=dict(index=k,command=command,status='ejecutando',log=str(log))
    if len(state['steps'])>k:state['steps'][k]=item
    else:state['steps'].append(item)
    state['status']='ejecutando';save();print(f'[{k+1}/{len(steps)}] {command}',flush=True)
    t=time.monotonic()
    with log.open('w') as f: ret=subprocess.run([sys.executable,*command],cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    item.update(status='completo' if ret==0 else 'fallido',seconds=round(time.monotonic()-t,3),returncode=ret)
    save();print(item,flush=True)
    if ret:state['status']='fallido';save();raise SystemExit(ret)
state.update(status='calculos_completos',finished_utc=datetime.now(timezone.utc).isoformat());save()
