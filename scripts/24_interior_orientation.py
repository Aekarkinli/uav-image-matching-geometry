#!/usr/bin/env python3
"""What the self-calibration recovered, and where it went in the surface.

The interpretation offered for the near-nadir block rests on the coupling
between interior and exterior orientation that a weak network cannot separate.
That coupling is measurable here, because every reconstruction adjusted its own
interior orientation and the reference solution reports one of its own. This
script collects the estimated parameters, expresses them in the same convention
as the reference calibration, and relates the recovered principal distance to
the height offset of the delivered surface, which is the observable the
coupling predicts.

For a nadir view over flat terrain a relative error in principal distance
scales the reconstruction, and after an alignment that fixes the camera
positions the residual appears as a height offset of about the flying height
times the relative error, with the opposite sign.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT, SFM_REPORTS_DIR  # noqa: E402

ANALYSIS = PROJECT_ROOT / "results" / "analysis"
SURFACE_SUMMARY = PROJECT_ROOT / "results" / "tables" / "surface_summary.json"
EVALUATION = PROJECT_ROOT / "results" / "evaluation"
DATASETS = ["D1_154_building", "D2_111_nadir"]
# the flying height above the terrain of each block, from the reference geometry
_GEOMETRY = json.loads((PROJECT_ROOT / "results" / "reference_geometry"
                        / "network_geometry_summary.json").read_text(encoding="utf-8"))
FLYING_HEIGHT = {entry["dataset"]: entry["flying_height_above_terrain_m"]
                 for entry in _GEOMETRY.values()}

LABEL = {
    "rootsift-lightglue": "RootSIFT",
    "rootsift-nn_ratio": "RootSIFT, nearest neighbour",
    "doghardnet-lightglue": "DoG + HardNet",
    "superpoint-lightglue": "SuperPoint",
    "aliked-lightglue": "ALIKED",
    "disk-lightglue": "DISK",
    "loftr-dense": "LoFTR",
}


def reference_calibration(dataset: str) -> dict:
    path = PROJECT_ROOT / dataset / "exports" / "sensor_calibration.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    row = max(rows, key=lambda r: float(r["f"]))
    width, height = float(row["width"]), float(row["height"])
    return {
        "c": float(row["f"]),
        "x0": width / 2.0 + float(row["cx"]),
        "y0": height / 2.0 + float(row["cy"]),
        "k1": float(row["k1"]), "k2": float(row["k2"]),
        # decentring terms in the OpenCV order of Equation 1
        "p1": float(row["p1"]), "p2": float(row["p2"]),
        "pixel_mm": float(row["pixel_width"]) * 1000.0,
    }


def estimated(dataset: str, config: str) -> dict | None:
    path = SFM_REPORTS_DIR / dataset / f"{config}.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    cameras = report.get("cameras") or {}
    if not cameras:
        return None
    params = list(cameras.values())[0]["params"]
    fx, fy, x0, y0, k1, k2, p1, p2 = params
    return {"fx": fx, "fy": fy, "c": 0.5 * (fx + fy), "x0": x0, "y0": y0,
            "k1": k1, "k2": k2, "p1": p1, "p2": p2,
            "registered": report.get("registered_images"),
            "components": report.get("model_component_count")}


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def main() -> int:
    surface = json.loads(SURFACE_SUMMARY.read_text(encoding="utf-8"))
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    rows, relation = [], {}
    for dataset in DATASETS:
        reference = reference_calibration(dataset)
        records = [record for record in surface.values()
                   if record["dataset"] == dataset]
        for record in sorted(records, key=lambda r: r["config"]):
            values = estimated(dataset, record["config"])
            if values is None:
                continue
            network = json.loads(
                (EVALUATION / dataset / f"{record['config']}.json")
                .read_text(encoding="utf-8"))
            rows.append({
                "dataset": dataset,
                "config": record["config"],
                "method": LABEL["-".join(record["config"].split("-")[:2])],
                "resolution": record["resolution"],
                "registered": values["registered"],
                "components": values["components"],
                "c_px": values["c"],
                "fx_minus_fy_px": values["fx"] - values["fy"],
                "c_minus_reference_px": values["c"] - reference["c"],
                "x0_minus_reference_px": values["x0"] - reference["x0"],
                "y0_minus_reference_px": values["y0"] - reference["y0"],
                "k1": values["k1"], "k2": values["k2"],
                "p1": values["p1"], "p2": values["p2"],
                "held_out_rmse_m": network["similarity_held_out"]["rmse"],
                "surface_median_m": record["core_median"],
                "trend_amplitude_m": record["trend_amplitude"],
                "detrended_nmad_m": record["detrended_nmad"],
            })

        block = [row for row in rows if row["dataset"] == dataset]
        core = [row for row in block if row["method"] != "LoFTR"]
        height = FLYING_HEIGHT[dataset]
        for name, subset in (("all", block), ("without LoFTR", core)):
            if len(subset) < 3:
                continue
            c = np.array([row["c_px"] for row in subset])
            median = np.array([row["surface_median_m"] for row in subset])
            slope, intercept = np.polyfit(c, median, 1)

            # with five or six points a single slope says little on its own, so
            # every configuration is dropped in turn
            dropped_slopes, dropped_rho = [], []
            for omit in range(len(subset)):
                keep = [i for i in range(len(subset)) if i != omit]
                if len(keep) < 3:
                    continue
                dropped_slopes.append(float(np.polyfit(c[keep], median[keep], 1)[0]))
                dropped_rho.append(spearman(c[keep], median[keep]))

            relation[f"{dataset} | {name}"] = {
                "n": len(subset),
                "configurations": [row["method"] for row in subset],
                "principal_distance_definition":
                    "mean of the two estimated principal distances",
                "slope_leave_one_out_min": min(dropped_slopes) if dropped_slopes
                    else float("nan"),
                "slope_leave_one_out_max": max(dropped_slopes) if dropped_slopes
                    else float("nan"),
                "spearman_leave_one_out_min": min(dropped_rho) if dropped_rho
                    else float("nan"),
                "spearman_leave_one_out_max": max(dropped_rho) if dropped_rho
                    else float("nan"),
                "flying_height_m": height,
                "reference_principal_distance_px": reference["c"],
                "spearman_c_vs_surface_median": spearman(c, median),
                "slope_m_per_px": float(slope),
                "predicted_slope_m_per_px": -height / reference["c"],
                "spearman_absolute_c_error_vs_held_out":
                    spearman([abs(row["c_minus_reference_px"]) for row in subset],
                             [row["held_out_rmse_m"] for row in subset]),
                "spearman_held_out_vs_trend":
                    spearman([row["held_out_rmse_m"] for row in subset],
                             [row["trend_amplitude_m"] for row in subset]),
                "spearman_held_out_vs_detrended":
                    spearman([row["held_out_rmse_m"] for row in subset],
                             [row["detrended_nmad_m"] for row in subset]),
                "c_spread_px": float(c.max() - c.min()),
            }

    # the spread of the interior orientation over repeated runs, for scale
    repeatability = {}
    for dataset in DATASETS:
        for stem in ("rootsift-lightglue-r2048-k8192",
                     "superpoint-lightglue-r2048-k8192",
                     "aliked-lightglue-r2048-k8192"):
            values = []
            for suffix in ("", "__seed1", "__seed2", "__seed3", "__seed4"):
                entry = estimated(dataset, stem + suffix)
                if entry:
                    values.append(entry["c"])
            if len(values) > 1:
                repeatability[f"{dataset} | {stem}"] = {
                    "runs": len(values),
                    "c_range_px": float(max(values) - min(values)),
                }

    out = ANALYSIS / "interior_orientation.json"
    out.write_text(json.dumps(
        {"rows": rows, "relation": relation, "repeatability": repeatability,
         "reference": {d: reference_calibration(d) for d in DATASETS}},
        indent=2), encoding="utf-8")

    with (ANALYSIS / "interior_orientation.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out.name} and interior_orientation.csv ({len(rows)} rows)")
    for key, entry in relation.items():
        print(f"  {key:34s} n={entry['n']}  rho={entry['spearman_c_vs_surface_median']:+.3f} "
              f"[{entry['spearman_leave_one_out_min']:+.3f},"
              f"{entry['spearman_leave_one_out_max']:+.3f}]  "
              f"slope={entry['slope_m_per_px']:+.4f} "
              f"[{entry['slope_leave_one_out_min']:+.4f},"
              f"{entry['slope_leave_one_out_max']:+.4f}] m/px  "
              f"predicted={entry['predicted_slope_m_per_px']:+.4f}  "
              f"c spread={entry['c_spread_px']:.2f} px")
    for key, entry in repeatability.items():
        print(f"  repeat {key:52s} range {entry['c_range_px']:.4f} px "
              f"over {entry['runs']} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
