"""Comprueba conservación de análisis previos y SHA-256 de los 680 NetCDF."""
from pathlib import Path
import sys,json,hashlib,importlib.metadata,platform
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent;ROOT=Path(json.loads((R/'configuracion.json').read_text())['repo_origen'])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
before=json.loads((R/'versiones_anteriores_sha256_antes.json').read_text())
after={k:{'sha256':sha(ROOT/k),'bytes':(ROOT/k).stat().st_size} for k in before}
(R/'versiones_anteriores_sha256_despues.json').write_text(json.dumps(after,indent=2))
changed=[k for k in before if before[k]!=after[k]]
expected={}
for line in (R/'datos_sha256_registrados.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'):continue
    h,n=line.split(maxsplit=1);name=Path(n).name;expected[('pr/' if name.startswith('CR2MET') else 'pr1/')+name]=h
raw={k:sha(ROOT/k) for k in expected}
result=dict(passed=not changed and raw==expected,archivos_previos_comprobados=len(before),versiones_previas_modificadas=changed,
    NetCDF_comprobados=len(raw),NetCDF_distintos=[k for k in expected if raw[k]!=expected[k]])
(R/'integridad.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
packages={}
for n in ('numpy','pandas','xarray','netCDF4','cftime','h5netcdf','dask','scipy','matplotlib','cartopy','shapely','Pillow'):
    try:packages[n]=importlib.metadata.version(n)
    except importlib.metadata.PackageNotFoundError:packages[n]=None
(R/'entorno.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=packages),indent=2))
print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
