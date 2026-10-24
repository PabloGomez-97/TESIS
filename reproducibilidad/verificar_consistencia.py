"""Comprueba relaciones entre salidas regeneradas, sin usar las de la raíz."""
import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ejecucion", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    root = args.ejecucion.resolve()
    checks = []

    def check(name, ok, detail=None):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(("OK " if ok else "FALLO ") + name, flush=True)

    def close(x, y, atol):
        return bool(np.allclose(x, y, atol=atol, rtol=0, equal_nan=True))

    def load_json(rel):
        return json.loads((root / rel).read_text())

    def normalize(s):
        return "".join(c for c in unicodedata.normalize("NFKD", s.lower())
                       if not unicodedata.combining(c)).replace(" ", "_")

    # Pickles locales generados por los scripts de esta ejecución.
    import pickle
    with (root / "_pregunta9_outputs/pdf_durations_i_global.pkl").open("rb") as f:
        pools = pickle.load(f)
    with (root / "_pregunta9_calibracion_outputs/spell_durations_sensibilidad_4cuencas.pkl").open("rb") as f:
        sens = pickle.load(f)
    table = pd.read_csv(root / "resultados_tesis/tabla3_estadisticos_ventanas.csv")
    names = {"CR2MET (hist)": "CR2MET hist", "ALADIN (hist)": "ALADIN hist",
             "ALADIN (futuro)": "ALADIN fut"}
    for basin, datasets in pools.items():
        edges = np.unique(np.round(np.logspace(
            0, np.log10(max(np.max(a) for a in datasets.values()) + 1), 26))) - 0.5
        for dataset, a in datasets.items():
            check(f"Duraciones positivas enteras: {basin}/{dataset}",
                  a.ndim == 1 and (a >= 1).all() and (a == np.floor(a)).all())
            row = table[(table.ventana == basin) &
                        (table.producto == names[dataset])].iloc[0]
            expected = [a.size, a.mean(), np.median(a), *np.percentile(a, [90, 99]),
                        a.max(), np.sum(a >= 20), 100 * np.mean(a >= 20)]
            observed = row[["n_spells", "media", "mediana", "p90", "p99",
                            "max", "n_ge20", "frac_ge20_pct"]].to_numpy(float)
            check(f"Tabla 3 desde duraciones: {basin}/{dataset}", close(
                expected, observed, 0.00051))
            counts, _ = np.histogram(a, bins=edges)
            check(f"Histograma conserva eventos: {basin}/{dataset}",
                  counts.sum() == a.size)
        for main, sensitivity in (("CR2MET (hist)", "CR2MET 1 mm"),
                                   ("ALADIN (hist)", "ALADIN R0 global")):
            check(f"PDF principal y sensibilidad coinciden: {basin}/{main}",
                  np.array_equal(datasets[main], sens[basin][sensitivity]))

    rr_values = {}
    for suffix in ("", "_1mm", "_ssp370"):
        folder = "outputs_pregunta5_aladin" + suffix
        result = load_json(folder + "/results.json")
        rr_values[suffix] = result
        for region, curve in result["curves"].items():
            spells = pd.read_csv(root / folder / ("spells_" + normalize(region) + ".csv"))
            check(f"Meses permitidos RR: {suffix}/{region}",
                  spells.start_month.between(3, 11).all())
            good = True
            for item in curve:
                threshold = item["Umbral (dias)"]
                years = []
                for first, last in ((1980, 2014), (2040, 2074)):
                    selected = spells[spells.start_year.between(first, last)]
                    counts = selected[selected.duration >= threshold].groupby("start_year").size()
                    years.append(counts.reindex(range(first, last + 1), fill_value=0).to_numpy())
                h, f = years
                good &= h.sum() == item["N1"] and f.sum() == item["N2"]
                expected_rr = f.sum() / h.sum() if h.sum() else np.nan
                good &= close(expected_rr, item["RR"], 0.000051)
                rng = np.random.default_rng(42)
                boot = []
                for _ in range(1000):
                    draw_h = rng.choice(35, 35, replace=True)
                    draw_f = rng.choice(35, 35, replace=True)
                    den = h[draw_h].sum()
                    if den:
                        boot.append(f[draw_f].sum() / den)
                quantiles = np.percentile(boot, [2.5, 50, 97.5]) if boot else np.full(3, np.nan)
                good &= close(quantiles, [item["IC95 inferior"],
                                          item["RR bootstrap mediano"],
                                          item["IC95 superior"]], 0.000051)
                good &= item["Replicas descartadas"] == 1000 - len(boot)
            check(f"10 RR y bootstrap independiente por conteos anuales: {suffix}/{region}", good)

    z = np.load(root / "_pregunta7_outputs/metrics_integrado_cache.npz")
    check("Máscara Chile de 3587 celdas", z["chile_mask"].sum() == 3587)
    for metric in ("mean", "t99"):
        c, a = z[metric + "_cr2"], z[metric + "_ala"]
        valid = z["chile_mask"] & np.isfinite(c) & np.isfinite(a)
        b = np.load(root / f"resultados_tesis/bootstrap_por_celda_A_opcion_B_integrado_{metric}.npz")
        check("Máscara común principal " + metric,
              valid.sum() == 3526 and np.array_equal(valid, b["valida"]))
        check("Delta mapa y bootstrap " + metric,
              close((a - c)[valid], b["delta"][valid], 0.0002))
    boot_summary = pd.read_csv(root / "outputs_pregunta7_bootstrap/bootstrap_summary.csv")
    for criterion, prefix in (("A_opcion_B_integrado", "Forma A"),
                               ("B_mismo_umbral", "Forma B"),
                               ("C_umbral_local", "Forma C")):
        for metric, label in (("mean", "Duracion media"), ("t99", "t99")):
            b = np.load(root / f"resultados_tesis/bootstrap_por_celda_{criterion}_{metric}.npz")
            valid = b["valida"]
            finite = np.isfinite(b["ic_low"]) & np.isfinite(b["ic_high"])
            sig_lo = finite & (b["ic_high"] < 0)
            sig_hi = finite & (b["ic_low"] > 0)
            check(f"Significancia desde intervalos: {criterion}/{metric}",
                  np.array_equal(b["mas_corto"], sig_lo) and
                  np.array_equal(b["mas_largo"], sig_hi) and
                  np.array_equal(b["significativo"], sig_lo | sig_hi))
            row = boot_summary[boot_summary.Escenario.str.startswith(prefix) &
                               boot_summary.Metrica.str.startswith(label)].iloc[0]
            expected = [b["delta"][valid].mean(), np.median(b["delta"][valid]),
                        100 * (sig_lo | sig_hi)[valid].mean(),
                        100 * sig_hi[valid].mean(), 100 * sig_lo[valid].mean(), valid.sum()]
            observed = row[["Delta medio espacial (dias)", "Delta mediano espacial (dias)",
                            "% celdas significativas", "% ALADIN > CR2MET (sig.)",
                            "% ALADIN < CR2MET (sig.)", "Celdas validas"]].to_numpy(float)
            check(f"Resumen desde NPZ: {criterion}/{metric}", close(expected, observed, 0.00051))

    markov = pd.read_csv(root / "resultados_tesis/descomposicion_markov_rr_regiones.csv")
    curve = {(r, item["Umbral (dias)"]): item
             for r, items in rr_values[""]["curves"].items() for item in items}
    check("Conteos independientes Markov y RR",
          all(row.N_hist_obs == curve[row.region, row.D]["N1"] and
              row.N_fut_obs == curve[row.region, row.D]["N2"]
              for row in markov.itertuples()))
    selected = markov[markov.N_hist_obs >= 100]
    check("31 casos compatibles con solo frecuencia",
          len(selected) == 31 and selected.solo_frec_dentro_IC95_obs.all())
    persistence = pd.read_csv(root / "resultados_tesis/contraste_persistencia_regiones.csv")
    check("IC95 de delta rho incluye cero en las cuatro regiones",
          len(persistence) == 4 and persistence.rho_IC_incluye_0.all() and
          (persistence.rho_delta_IC95_inf <= 0).all() and
          (persistence.rho_delta_IC95_sup >= 0).all())
    censored = pd.read_csv(root / "_test_regrillado/rachas_censuradas_por_cuenca.csv")
    labels = {"CR2MET 1980-2014 (1 mm)": "CR2MET (hist)",
              "ALADIN 1980-2014 (R0)": "ALADIN (hist)",
              "ALADIN 2040-2074 (R0)": "ALADIN (futuro)"}
    check("Rachas completas en diagnóstico de censura y PDFs",
          all(row.rachas_completas == pools[row.cuenca][labels[row.serie]].size
              for row in censored.itertuples()))
    calibration = load_json("resultados_tesis/calibracion_R0.json")
    check("R0 aplicado es el redondeo de la calibración",
          round(calibration["R0_biseccion_mm"], 3) == calibration["R0_usado_mm"] == 5.285)
    result = {"passed": all(x["passed"] for x in checks),
              "checks": len(checks), "failures": [x for x in checks if not x["passed"]],
              "details": checks}
    args.salida.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "checks", "failures")}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
