"""Cadena completa: NetCDF -> calibración local 0,1 mm -> resultados -> informe."""
from pathlib import Path
from datetime import datetime,timezone
import sys,os,subprocess,json,time
R=Path(__file__).resolve().parent;W=R/'ejecucion';p=R/'estado.json'
state=json.loads(p.read_text()) if p.exists() else dict(status='preparando',started_utc=datetime.now(timezone.utc).isoformat(),steps=[])
def save():p.write_text(json.dumps(state,indent=2,ensure_ascii=False))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');env.pop('PYTHONPATH',None)
steps=json.loads((R/'pasos.json').read_text())
for i,c in enumerate(steps):
    if len(state['steps'])>i and state['steps'][i]['status']=='completo' and state['steps'][i]['command']==c:continue
    log=R/'logs'/f'{i:02d}_{Path(c[0]).stem}.log'
    item=dict(index=i,command=c,status='ejecutando',log=str(log),started_utc=datetime.now(timezone.utc).isoformat())
    if len(state['steps'])>i:state['steps'][i]=item
    else:state['steps'].append(item)
    state['status']='ejecutando';save();print(f'[{i+1}/{len(steps)}] {c}',flush=True)
    t=time.monotonic()
    with log.open('w') as f:code=subprocess.run([sys.executable,*c],cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    item.update(status='completo' if code==0 else 'fallido',returncode=code,seconds=round(time.monotonic()-t,3));save()
    print(f"  {item['status']}: {item['seconds']} s",flush=True)
    if code:state['status']='fallido';save();raise SystemExit(code)
state.update(status='completo',finished_utc=datetime.now(timezone.utc).isoformat());save()
