"""Reproduce los scripts del paper en un directorio nuevo y compara todas sus salidas.

No cambia scripts ni salidas de la raíz. Solo enlaza pr/ y pr1/; ninguna caché de
análisis se copia al directorio de ejecución. Los resultados y logs quedan fuera
del repositorio por defecto. Requiere el entorno científico existente.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

OUTPUT_DIRS = (
    "resultados_tesis", "paper_figuras", "_pregunta7_outputs",
    "_pregunta9_outputs", "_pregunta9_calibracion_outputs",
    "outputs_pregunta5_aladin", "outputs_pregunta5_aladin_1mm",
    "outputs_pregunta5_aladin_ssp370", "outputs_pregunta7_bootstrap",
    "_test_regrillado",
)
OUTPUT_TYPES = {".csv", ".json", ".npz", ".pkl", ".png"}
STEPS = [
    ["_generar_fig03_calibracion_R0.py"],
    ["_run_pregunta7_bootstrap.py"],
    ["_calcular_sesgo_ventanas.py"],
    ["_generar_tablaS1_sensibilidad_chile.py"],
    ["_run_pregunta5_aladin.py"],
    ["_run_pregunta5_aladin.py", "--umbral", "1"],
    ["_run_pregunta5_aladin.py", "--escenario", "ssp370"],
    ["_referencia_markov_rr_regiones.py"],
    ["_contraste_persistencia_regiones.py"],
    ["_generar_pregunta9_pdf_loglog_cuencas.py"],
    ["_generar_figS1_sensibilidad_umbral.py"],
    ["_calcular_tabla3_ventanas.py"],
    ["_calcular_frecuencia_humeda_ventanas.py"],
    ["_calcular_estacionalidad_marnov_regiones.py"],
    ["_generar_figS2_mapa_R0_local.py"],
    ["_test_regrillado_conservativo.py"],
    ["_cuantificar_censuradas.py"],
    ["_generar_figT01_referencia_sin_memoria.py"],
    ["_generar_fig_rr_regiones.py"],
    ["_generar_fig05_mapas_persistencia.py"],
    ["_generar_fig_dominio_ventanas_regiones.py"],
    ["_generar_fig02_esquema_dryspell.py"],
    ["_sensibilidad_minimo_rachas.py"],
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def outputs(root):
    return {
        p.relative_to(root).as_posix(): p
        for name in OUTPUT_DIRS if (root / name).exists()
        for p in (root / name).rglob("*")
        if p.is_file() and p.suffix in OUTPUT_TYPES
    }


def manifest(paths):
    return {str(name): {"sha256": sha(p), "bytes": p.stat().st_size}
            for name, p in sorted(paths.items())}


def equal_content(a, b):
    """Igualdad exacta, permitiendo NaN en las mismas posiciones."""
    import numpy as np
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and (
            np.array_equal(a, b, equal_nan=True)
            if a.dtype.kind in "fc" else np.array_equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal_content(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(
            equal_content(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def compare_file(a, b):
    import numpy as np
    from PIL import Image
    if sha(a) == sha(b):
        return {"equal": True, "method": "sha256"}
    suffix = a.suffix
    if suffix == ".npz":
        with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
            if set(x.files) != set(y.files):
                return {"equal": False, "method": "npz", "detail": "claves distintas"}
            changes = []
            for k in x.files:
                if not equal_content(x[k], y[k]):
                    item = {"key": k, "baseline_shape": list(x[k].shape),
                            "new_shape": list(y[k].shape)}
                    if x[k].shape == y[k].shape and x[k].dtype.kind in "fiu" and y[k].dtype.kind in "fiu":
                        finite = np.isfinite(x[k]) & np.isfinite(y[k])
                        item["max_abs"] = float(np.max(np.abs(
                            x[k][finite].astype(float) - y[k][finite].astype(float)))) if finite.any() else None
                        item["finite_masks_equal"] = bool(np.array_equal(np.isfinite(x[k]), np.isfinite(y[k])))
                    changes.append(item)
            return {"equal": not changes, "method": "npz arrays exactos", "changes": changes}
    if suffix == ".pkl":
        with a.open("rb") as f:
            x = pickle.load(f)
        with b.open("rb") as f:
            y = pickle.load(f)
        return {"equal": bool(equal_content(x, y)), "method": "pickle contenido exacto"}
    if suffix == ".json":
        return {"equal": bool(equal_content(json.loads(a.read_text()), json.loads(b.read_text()))),
                "method": "json contenido exacto"}
    if suffix == ".png":
        with Image.open(a) as im:
            x = np.asarray(im.convert("RGBA"))
        with Image.open(b) as im:
            y = np.asarray(im.convert("RGBA"))
        return {"equal": bool(np.array_equal(x, y)), "method": "pixeles RGBA exactos",
                "baseline_shape": list(x.shape), "new_shape": list(y.shape)}
    if suffix == ".csv":
        import pandas as pd
        x, y = pd.read_csv(a), pd.read_csv(b)
        try:
            pd.testing.assert_frame_equal(x, y, check_exact=True)
            return {"equal": True, "method": "CSV valores exactos"}
        except AssertionError as e:
            return {"equal": False, "method": "CSV valores exactos", "detail": str(e)[:1600]}
    return {"equal": False, "method": "sha256"}


def raw_files(root):
    return {p.relative_to(root).as_posix(): p
            for folder in ("pr", "pr1") for p in (root / folder).glob("*.nc")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destino", type=Path, help="Directorio nuevo; debe no existir.")
    args = parser.parse_args()
    repo = args.repo.resolve()
    stamp = datetime.now(ZoneInfo("America/Santiago")).strftime("%Y-%m-%d_%H%M%S")
    run = args.destino or Path.home() / "Documents" / "TESIS_reproducciones" / stamp
    run = run.resolve()
    if run == repo or repo in run.parents:
        raise ValueError("El destino debe estar fuera del repositorio.")
    run.mkdir(parents=True, exist_ok=False)
    work, baseline, logs = (run / n for n in ("ejecucion", "referencia", "logs"))
    for p in (work, baseline, logs):
        p.mkdir()
    state = {"status": "preparando", "repo": str(repo), "run": str(run),
             "started_utc": datetime.now(timezone.utc).isoformat(), "steps": []}
    write_json(run / "estado.json", state)
    print(f"RUN_DIR={run}", flush=True)
    try:
        scripts = sorted(repo.glob("_*.py"))
        expected_scripts = {step[0] for step in STEPS}
        if {p.name for p in scripts} != expected_scripts:
            raise ValueError("El inventario de scripts cambió; actualizar STEPS antes de ejecutar.")
        source_manifest = manifest({p.name: p for p in scripts})
        for p in scripts:
            shutil.copy2(p, work / p.name)
        write_json(run / "scripts_sha256.json", source_manifest)
        shutil.copy2(Path(__file__), run / "ejecutar.py")
        for name in ("REGISTRO_RESULTADOS.md", "CLAUDE.md", "requirements.txt",
                     "PROCEDENCIA_DATOS.md"):
            if (repo / name).exists():
                shutil.copy2(repo / name, run / name)
        for name in ("pr", "pr1"):
            (work / name).symlink_to(repo / name, target_is_directory=True)
        existing = outputs(repo)
        for rel, p in existing.items():
            target = baseline / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        before_outputs = manifest(existing)
        write_json(run / "salidas_referencia_sha256.json", before_outputs)
        assert not outputs(work), "El directorio nuevo contiene salidas previas."
        print(f"Preparado: {len(scripts)} scripts; {len(existing)} salidas de referencia; ninguna salida en ejecucion.", flush=True)
        print("Verificando hashes de datos crudos...", flush=True)
        inputs_before = manifest(raw_files(repo))
        write_json(run / "datos_sha256_antes.json", inputs_before)
        expected = {}
        for line in (repo / "resultados_tesis/sha256_datos_crudos.txt").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            digest, filename = line.split(maxsplit=1)
            filename = Path(filename).name
            folder = "pr" if filename.startswith("CR2MET") else "pr1"
            expected[f"{folder}/{filename}"] = digest
        raw_check = {"listed": len(expected), "found": len(inputs_before),
                     "mismatches": [p for p, h in expected.items()
                                    if inputs_before.get(p, {}).get("sha256") != h],
                     "unlisted": sorted(set(inputs_before) - set(expected))}
        write_json(run / "verificacion_datos_registrados.json", raw_check)
        if raw_check["mismatches"] or raw_check["unlisted"]:
            raise ValueError("Datos crudos distintos del registro; consultar verificacion_datos_registrados.json")
        import cartopy
        geometry_root = Path(cartopy.config["data_dir"])
        geo_paths = {str(p.relative_to(geometry_root)): p
                     for p in geometry_root.rglob("*") if p.is_file()}
        geo_before = manifest(geo_paths)
        write_json(run / "geometrias_sha256_antes.json", geo_before)
        packages = {}
        for name in ("numpy", "pandas", "xarray", "netCDF4", "cftime", "h5netcdf",
                     "dask", "scipy", "matplotlib", "cartopy", "shapely", "Pillow"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        write_json(run / "entorno.json", {
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages,
            "cartopy_data_dir": str(geometry_root),
            "thread_env": {k: os.environ.get(k) for k in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        })
        with (run / "pip_freeze.txt").open("w") as f:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, check=True)
        state["status"] = "ejecutando"
        state["raw_files_verified"] = len(expected)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        env.pop("PYTHONPATH", None)
        for i, command in enumerate(STEPS, 1):
            log_path = logs / f"{i:02d}_{Path(command[0]).stem}.log"
            item = {"index": i, "command": command, "status": "ejecutando",
                    "log": str(log_path), "started_utc": datetime.now(timezone.utc).isoformat()}
            state["steps"].append(item)
            write_json(run / "estado.json", state)
            print(f"[{i}/{len(STEPS)}] {' '.join(command)}", flush=True)
            t0 = time.monotonic()
            with log_path.open("w") as f:
                result = subprocess.run([sys.executable, *command], cwd=work,
                                        env=env, stdout=f, stderr=subprocess.STDOUT)
            item.update(status="completo" if result.returncode == 0 else "fallido",
                        returncode=result.returncode, seconds=round(time.monotonic() - t0, 3))
            write_json(run / "estado.json", state)
            print(f"  salida={result.returncode}; {item['seconds']:.1f} s", flush=True)
            if result.returncode:
                raise RuntimeError(f"Falló {' '.join(command)}; consultar {log_path}")
        state["status"] = "comparando"
        write_json(run / "estado.json", state)
        fresh = outputs(work)
        comparisons = []
        for rel in sorted(set(existing) | set(fresh)):
            if rel not in fresh:
                result = {"equal": False, "method": "archivo faltante"}
            elif rel not in existing:
                result = {"equal": False, "method": "archivo adicional"}
            else:
                result = compare_file(baseline / rel, fresh[rel])
            comparisons.append({"file": rel, **result})
        write_json(run / "comparacion_salidas.json", comparisons)
        write_json(run / "salidas_nuevas_sha256.json", manifest(fresh))
        print("Verificando integridad final de entradas y salidas originales...", flush=True)
        inputs_after = manifest(raw_files(repo))
        write_json(run / "datos_sha256_despues.json", inputs_after)
        final_outputs = manifest(outputs(repo))
        write_json(run / "salidas_originales_sha256_despues.json", final_outputs)
        geo_after = manifest({str(p.relative_to(geometry_root)): p
                              for p in geometry_root.rglob("*") if p.is_file()})
        write_json(run / "geometrias_sha256_despues.json", geo_after)
        sources_after = manifest({p.name: p for p in repo.glob("_*.py")})
        summary = {
            "steps_complete": len(state["steps"]), "outputs_compared": len(comparisons),
            "outputs_equal": sum(r["equal"] for r in comparisons),
            "differences": [r for r in comparisons if not r["equal"]],
            "raw_inputs_unchanged": inputs_before == inputs_after,
            "original_outputs_unchanged": before_outputs == final_outputs,
            "source_scripts_unchanged": source_manifest == sources_after,
            "geometry_inputs_unchanged": geo_before == geo_after,
        }
        summary["passed"] = (not summary["differences"] and all(summary[k] for k in (
            "raw_inputs_unchanged", "original_outputs_unchanged",
            "source_scripts_unchanged", "geometry_inputs_unchanged")))
        write_json(run / "resumen.json", summary)
        state.update(status="reproducido" if summary["passed"] else "diferencias",
                     finished_utc=datetime.now(timezone.utc).isoformat(), summary=summary)
        write_json(run / "estado.json", state)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0 if summary["passed"] else 2
    except Exception:
        state.update(status="fallido", error=traceback.format_exc(),
                     finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(run / "estado.json", state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
