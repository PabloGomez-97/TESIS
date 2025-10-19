# %%
import xarray as xr, numpy as np, pandas as pd, matplotlib.pyplot as plt
from glob import glob

files = sorted(glob("pr/CR2MET_pr_v2.5_day_????_??_005deg.nc"))
len(files), files[:24]
# Abrir y concatenar en el eje 'time'
ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 365})
pr = ds["pr"]  # mm/día
print(pr)

# %%

punto = pr.sel(lat=-33.45, lon=-70.66, method="nearest")
serie = punto.to_series().sort_index()   # pandas Series con índice time
print(serie.index.min(), serie.index.max())


mensual = serie.resample("MS").sum()
anual   = serie.resample("YS").sum()

fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=False)
mensual.plot(ax=ax[0]); ax[0].set_title("Precipitación mensual (CR2MET) - Santiago")
ax[0].set_ylabel("promedio de precipitación mensual")

anual.plot(ax=ax[1]);   ax[1].set_title("Precipitación anual (promedio diario)")
ax[1].set_ylabel("promedio de precipitación anual")
plt.tight_layout(); plt.show()

# %%