"""Registra metadatos y continuidad de tiempo de cada NetCDF, sin cargar pr."""
import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for folder in ("pr", "pr1"):
        for p in sorted((args.repo / folder).glob("*.nc")):
            with netCDF4.Dataset(p) as ds:
                t = ds.variables["time"]
                values = t[:]
                calendar = getattr(t, "calendar", "standard")
                dates = netCDF4.num2date(
                    [values[0], values[-1]], t.units, calendar=calendar)
                # Convertir incremento de una unidad numérica a días.
                unit = t.units.split(" since ")[0]
                scale = {"days": 1, "day": 1, "hours": 1 / 24,
                         "seconds": 1 / 86400}.get(unit)
                daily = bool(np.allclose(np.diff(values) * scale, 1, rtol=0, atol=1e-8)) if scale else None
                attrs = {key: str(getattr(ds, key)) for key in
                         ("model_id", "driving_model_id", "driving_experiment_name",
                          "driving_model_ensemble_member", "arpege_minor_version",
                          "comment", "authors", "obs_file", "creation_date")
                         if key in ds.ncattrs()}
                rows.append({
                    "file": p.relative_to(args.repo).as_posix(),
                    "time_start": str(dates[0]), "time_end": str(dates[-1]),
                    "days": int(len(values)), "calendar": calendar,
                    "time_units": t.units, "continuous_daily": daily,
                    "pr_units": getattr(ds.variables["pr"], "units", None),
                    "pr_dimensions": list(ds.variables["pr"].dimensions),
                    "dimensions": {key: len(dim) for key, dim in ds.dimensions.items()},
                    "attributes": attrs,
                })
    cr = [r for r in rows if r["file"].startswith("pr/")]
    ala = [r for r in rows if r["file"].startswith("pr1/")]
    gaps = []
    for group in (cr, *[[r for r in ala if f"_{scenario}_" in r["file"]]
                       for scenario in ("historical", "ssp585", "ssp370")]):
        group.sort(key=lambda r: r["time_start"])
        import pandas as pd
        for previous, following in zip(group, group[1:]):
            delta = pd.Timestamp(following["time_start"]) - pd.Timestamp(previous["time_end"])
            if delta != pd.Timedelta(days=1):
                gaps.append([previous["file"], following["file"], str(delta)])
    result = {
        "files": len(rows), "cr2met": len(cr), "aladin": len(ala),
        "all_continuous_daily": all(r["continuous_daily"] is True for r in rows),
        "gaps_or_overlaps_between_files": gaps, "metadata": rows,
        "note": "La continuidad no resuelve inconsistencias de escenario en atributos.",
    }
    args.salida.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "metadata"}, ensure_ascii=False))
    return 0 if result["all_continuous_daily"] and not gaps else 1


if __name__ == "__main__":
    raise SystemExit(main())
