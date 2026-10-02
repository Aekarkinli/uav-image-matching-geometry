#!/usr/bin/env python3
"""Sensitivity of the surface statistics to the gate threshold.

The primary surface comparison removes cells whose height difference reaches
one metre. This recomputes the median, the robust dispersion, the trend
amplitude and the local component at half a metre, one metre and two metres,
and at no gate at all, and reports how the ordering of the configurations
moves.

Writes results/analysis/surface_gate_sensitivity.json and .csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT  # noqa: E402

ANALYSIS = PROJECT_ROOT / "results" / "analysis"
SURFACE = PROJECT_ROOT / "results" / "surface"
SUMMARY = PROJECT_ROOT / "results" / "tables" / "surface_summary.json"
DATASETS = ["D1_154_building", "D2_111_nadir"]
GATES = [0.5, 1.0, 2.0, float("inf")]
SUBSAMPLE = 200_000
SEED = 0

LABEL = {
    "rootsift-lightglue": "RootSIFT",
    "doghardnet-lightglue": "DoG + HardNet",
    "superpoint-lightglue": "SuperPoint",
    "aliked-lightglue": "ALIKED",
    "disk-lightglue": "DISK",
    "loftr-dense": "LoFTR",
}


def nmad(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def quadratic(grid: np.ndarray, mask: np.ndarray, rng) -> tuple[float, float]:
    """Amplitude of a fitted quadratic and the robust dispersion left after it.

    The fit uses the same subsample and the same normalisation as the reported
    statistics, so the numbers here are comparable with those of the article.
    """
    indices = np.flatnonzero(mask.ravel())
    if indices.size < 100:
        return float("nan"), float("nan")
    if indices.size > SUBSAMPLE:
        indices = rng.choice(indices, size=SUBSAMPLE, replace=False)
    rows, cols = np.unravel_index(indices, grid.shape)
    values = grid.ravel()[indices]
    y = rows / grid.shape[0]
    x = cols / grid.shape[1]
    design = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    return (float(np.percentile(fitted, 95) - np.percentile(fitted, 5)),
            nmad(values - fitted))


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows, report = [], {}

    for dataset in DATASETS:
        records = sorted((r for r in summary.values() if r["dataset"] == dataset),
                         key=lambda r: r["config"])
        for record in records:
            path = Path(record["raster"])
            if not path.exists():
                path = SURFACE / dataset / f"{record['config']}_difference.tif"
            with rasterio.open(path) as source:
                data = source.read(1).astype(np.float32)
            finite = np.isfinite(data)
            method = LABEL.get("-".join(record["config"].split("-")[:2]),
                               record["config"])
            for gate in GATES:
                rng = np.random.default_rng(SEED)
                mask = finite & (np.abs(data) < gate)
                values = data[mask]
                trend, detrended = quadratic(data, mask, rng)
                rows.append({
                    "dataset": dataset,
                    "config": record["config"],
                    "method": method,
                    "resolution": record["resolution"],
                    "gate_m": "none" if gate == float("inf") else f"{gate:g}",
                    "cells": int(mask.sum()),
                    "kept_pct": 100.0 * float(mask.sum()) / max(int(finite.sum()), 1),
                    "median_m": float(np.median(values)) if values.size else float("nan"),
                    "nmad_m": nmad(values),
                    "trend_amplitude_m": trend,
                    "detrended_nmad_m": detrended,
                })

    # how far the ordering moves between gates, on the detector-based set the
    # article reports the factor over
    for dataset in DATASETS:
        entry = {}
        for quantity in ("median_m", "trend_amplitude_m", "detrended_nmad_m"):
            at = {}
            for gate in GATES:
                key = "none" if gate == float("inf") else f"{gate:g}"
                selected = [r for r in rows if r["dataset"] == dataset
                            and r["gate_m"] == key and r["method"] != "LoFTR"]
                selected.sort(key=lambda r: r["config"])
                at[key] = [r[quantity] for r in selected]
            reference = at["1"]
            entry[quantity] = {
                "order_against_one_metre": {
                    key: spearman(reference, values)
                    for key, values in at.items() if key != "1"},
                "values_at_one_metre": reference,
                "configurations": [r["config"] for r in rows
                                   if r["dataset"] == dataset
                                   and r["gate_m"] == "1"
                                   and r["method"] != "LoFTR"],
            }
        # the ratio the article quotes, at each gate
        entry["local_component_factor"] = {}
        for gate in GATES:
            key = "none" if gate == float("inf") else f"{gate:g}"
            values = [r["detrended_nmad_m"] for r in rows
                      if r["dataset"] == dataset and r["gate_m"] == key
                      and r["method"] != "LoFTR"]
            entry["local_component_factor"][key] = max(values) / min(values)
        report[dataset] = entry

    (ANALYSIS / "surface_gate_sensitivity.json").write_text(
        json.dumps({"summary": report, "rows": rows}, indent=2), encoding="utf-8")
    with (ANALYSIS / "surface_gate_sensitivity.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("wrote surface_gate_sensitivity.json and .csv")
    for dataset, entry in report.items():
        print(f"  {dataset}")
        for quantity, values in entry.items():
            if quantity == "local_component_factor":
                print("    local component factor  "
                      + "  ".join(f"{k}: {v:.3f}" for k, v in values.items()))
                continue
            print(f"    {quantity:20s} rank correlation against the one-metre "
                  "gate  "
                  + "  ".join(f"{k}: {v:+.3f}"
                              for k, v in values["order_against_one_metre"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
