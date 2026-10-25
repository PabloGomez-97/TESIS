"""Campo histórico fijo. No se calibra con datos futuros ni se redondea a milímetros."""
from pathlib import Path
import numpy as np
import xarray as xr
ROOT=Path(__file__).resolve().parent

def campo_umbral():
    with np.load(ROOT/'_pregunta7_outputs/R0_local_map.npz') as z:
        return xr.DataArray(z['r0'].copy(), dims=('y','x'), coords={
            'y':z['y'].copy(),'x':z['x'].copy(),
            'lat':(('y','x'),z['lat'].copy()), 'lon':(('y','x'),z['lon'].copy())})
