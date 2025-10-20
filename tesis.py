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

# El modelo de Stechmann & Neelin, q(t) representa el contenido de vapor de agua en la columna (CWV). 
# CR2MET no tiene CWV vamos a construir un proxy que refleje el estado seco/húmedo de la atmosfera, por ejemplo un acumulado suavizado de la precipitación.

punto = pr.sel(lat=-33.45, lon=-70.66, method="nearest") # Agarramos latitud y longitud de Santiago central
serie = punto.to_series().sort_index()   # pandas Series con índice time, desde el 1960 al 2021
print(serie.index.min(), serie.index.max())


mensual = serie.resample("MS").sum() # suma del mes
anual   = serie.resample("YS").sum() # suma del año





# %%

# "q" proxy: acumulado suavizado (por ejemplo, media móvil de 15 días de precipitación)
# Si llovió harto en los últimos días, q es alto (atmósfera húmeda)
# Si no llovió, q es bajo (atmósfera seca)

q = serie.rolling(window=15, center=True, min_periods=5).mean()




# %%

# Acotamos el tiempo desde 1979 a 2021.

serie_1979 = serie["1979":"2021"]
serie_1979.plot(figsize=(12,4), alpha=0.6) # Gráfico






# %%

# Definir umbral de precipitación (p. ej., 1 mm/día)
umbral = 3.0
prec = serie_1979.fillna(0)

# Convertir a días secos = 1 si < umbral - dry es una serie binaria: 1 = seco, 0 = lluvioso
dry = (prec < umbral).astype(int)

# Identificar transiciones seco ↔ lluvioso
# Contar duración de secuencias consecutivas de días secos
dry_spells = []
count = 0
for val in dry:
    if val == 1:
        count += 1 # seguimos en seco
    else:
        if count > 0: # Terminó un período seco
            dry_spells.append(count)
            count = 0
if count > 0:
    dry_spells.append(count)

dry_spells = np.array(dry_spells)
print("Número de dry spells:", len(dry_spells))
print("Promedio de duración:", dry_spells.mean(), "días")










# %%

plt.figure(figsize=(6,4))
plt.hist(dry_spells, bins=np.arange(1,60,1), density=True, alpha=0.6)
plt.yscale("log"); plt.xscale("log")
plt.title("Distribución de duración de dry spells (1979–2021)")
plt.xlabel("Duración (días)")
plt.ylabel("Probabilidad")
plt.show()





# %% --------------------------------------------
# Dry spells por año (1979–2021)
# usa: serie_1979 (precip diaria) y umbral (mm/día)

prec = serie_1979.fillna(0.0)
dry  = (prec < umbral).astype("int8")   # 1 = seco, 0 = lluvioso

def run_lengths_of_ones(arr: np.ndarray) -> np.ndarray:
    """
    Longitudes de rachas consecutivas de 1s en un vector 1D (0/1).
    No cruza años: se usará dentro de cada grupo anual.
    """
    if arr.size == 0:
        return np.array([], dtype=int)
    # diff con ceros en los extremos para encontrar inicios y finales
    d = np.diff(np.r_[0, arr, 0])
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    return (ends - starts).astype(int)

# diccionario: {año: np.array(duraciones)}
dry_spells_year = {}
for year, g in dry.groupby(dry.index.year):
    dry_spells_year[year] = run_lengths_of_ones(g.values)

# resumen anual en un DataFrame
rows = []
for year in range(1979, 2022):
    arr = dry_spells_year.get(year, np.array([], dtype=int))
    if arr.size:
        rows.append({
            "year": year,
            "n_spells": int(arr.size),
            "mean_days": float(arr.mean()),
            "median_days": float(np.median(arr)),
            "max_days": int(arr.max()),
            # estimador pedido: tL ≈ var/mean
            "tL_days": float(arr.var(ddof=0) / arr.mean())
        })
    else:
        rows.append({
            "year": year,
            "n_spells": 0,
            "mean_days": np.nan,
            "median_days": np.nan,
            "max_days": 0,
            "tL_days": np.nan
        })

dry_year_summary = pd.DataFrame(rows).set_index("year").sort_index()

print(dry_year_summary.head())
print("\nPromedio 1979–2021:")
print(dry_year_summary[["n_spells","mean_days","max_days","tL_days"]].mean(numeric_only=True))

# gráficos rápidos
ax = dry_year_summary[["mean_days","max_days"]].plot(figsize=(10,4), marker="o")
ax.set_title("Dry spells en Santiago: media y máximo por año")
ax.set_ylabel("días"); ax.grid(True, alpha=.3)
plt.show()

ax = dry_year_summary["tL_days"].plot(figsize=(10,3), marker="s")
ax.set_title("Escala característica tL ≈ var/mean (por año)")
ax.set_ylabel("días"); ax.grid(True, alpha=.3)
plt.show()







# %%
# El mean_days es la duración promedio de los períodos secos en ese año
# El max_days es la duración del dry spell más largo registrado ese año
# La línea azul de mean_days se mantiene bastante estable entre 10 a 20 días -> lo que indica que el tamaño típico de los períodos secos no ha cambiado mucho.
# La línea naranja es mucho más variable, con picos grandes -> representa eventos de sequías prolongadas

# %%

# Duraciones por estación
months = prec.index.month
dry_months = []
count = 0
for i, val in enumerate(dry):
    if val == 1:
        count += 1
    else:
        if count > 0:
            month = months[i-count//2]  # mes "medio" del dry spell
            dry_months.append((month, count))
            count = 0
if count > 0:
    month = months[-count//2]
    dry_months.append((month, count))

df_dry = pd.DataFrame(dry_months, columns=["mes","duracion"])
plt.figure(figsize=(7,4))
df_dry.groupby("mes")["duracion"].mean().plot(kind="bar", color="skyblue")
plt.ylabel("Duración promedio (días)")
plt.title("Duración promedio de dry spells por mes")
plt.grid(alpha=0.3)
plt.show()

# %%


import xarray as xr

file = "pr_CHP12_CNRM-ESM2-1_historical_r1i1p1f2_CNRM-ALADIN64_v1_day_19500101-19781231_chile.nc"

# Opción A
ds = xr.open_dataset(file, engine="netcdf4")
print(ds)

# Si falla, Opción B
ds = xr.open_dataset(file, engine="h5netcdf")
print(ds)

# %%

from netCDF4 import Dataset
f = Dataset(file)      # si esto falla con “Not a netCDF file”, el archivo está corrupto o no es NetCDF
print(f.file_format)
f.close()

# %%
