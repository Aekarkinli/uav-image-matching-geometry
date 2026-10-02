#!/usr/bin/env python3
"""The surface comparison on a set of cells common to every configuration.

Each difference grid covers the cells its own dense cloud reached, and those
cells differ between configurations, so the statistics reported per
configuration are not taken over the same ground. The grids of a block share an
origin and a cell size, which makes the intersection straightforward to build.

Coverage is limited by the density of the fused cloud rather than by its extent,
so the strict cell-by-cell intersection is small and scattered. Both are
therefore reported: the strict intersection, and an aggregation to a coarser
cell in which a configuration contributes the median of its own valid cells,
which is the same aggregation the surface figures already display.
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
CORE_LIMIT = 1.0            # the gate the reported statistics already apply
BLOCK = 4                   # the aggregation the surface maps are drawn at

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


def trend_amplitude(grid: np.ndarray, mask: np.ndarray) -> float:
    """Peak to peak of a quadratic surface fitted in normalised grid indices."""
    rows, cols = np.nonzero(mask)
    if rows.size < 100:
        return float("nan")
    values = grid[rows, cols]
    y = (rows - rows.mean()) / max(rows.std(), 1.0)
    x = (cols - cols.mean()) / max(cols.std(), 1.0)
    design = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coefficients
    return float(np.percentile(fitted, 95) - np.percentile(fitted, 5))


def coarsen(grid: np.ndarray, factor: int) -> np.ndarray:
    rows = grid.shape[0] // factor * factor
    cols = grid.shape[1] // factor * factor
    trimmed = grid[:rows, :cols].reshape(
        rows // factor, factor, cols // factor, factor)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(trimmed, axis=(1, 3))


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    report, rows = {}, []

    for dataset in DATASETS:
        records = sorted((r for r in summary.values() if r["dataset"] == dataset),
                         key=lambda r: r["config"])
        grids, raw, shape = {}, {}, None
        for record in records:
            path = Path(record["raster"])
            if not path.exists():
                path = SURFACE / dataset / f"{record['config']}_difference.tif"
            with rasterio.open(path) as source:
                data = source.read(1).astype(np.float32)
            shape = data.shape if shape is None else shape
            if data.shape != shape:
                print(f"  shape mismatch for {record['config']}")
                continue
            raw[record["config"]] = data.copy()
            gated = data.copy()
            gated[np.abs(gated) >= CORE_LIMIT] = np.nan
            grids[record["config"]] = gated

        strict = np.ones(shape, dtype=bool)
        for data in grids.values():
            strict &= np.isfinite(data)

        # the same intersection built before the gate
        strict_ungated = np.ones(shape, dtype=bool)
        for data in raw.values():
            strict_ungated &= np.isfinite(data)

        coarse = {config: coarsen(data, BLOCK) for config, data in grids.items()}
        coarse_mask = np.ones(next(iter(coarse.values())).shape, dtype=bool)
        for data in coarse.values():
            coarse_mask &= np.isfinite(data)

        coarse_raw = {config: coarsen(data, BLOCK) for config, data in raw.items()}
        coarse_mask_ungated = np.ones(next(iter(coarse_raw.values())).shape, dtype=bool)
        for data in coarse_raw.values():
            coarse_mask_ungated &= np.isfinite(data)

        for config, data in grids.items():
            own = np.isfinite(data)
            finite = np.isfinite(raw[config])
            magnitude = np.abs(raw[config][finite])
            entry = {
                "dataset": dataset,
                "config": config,
                "method": LABEL.get("-".join(config.split("-")[:2]), config),
                "cells_before_gate": int(finite.sum()),
                "kept_by_gate_pct": 100.0 * float(own.sum()) / max(int(finite.sum()), 1),
                "beyond_0p5m_pct": 100.0 * float(np.mean(magnitude > 0.5)),
                "beyond_1m_pct": 100.0 * float(np.mean(magnitude > 1.0)),
                "beyond_2m_pct": 100.0 * float(np.mean(magnitude > 2.0)),
                "median_before_gate_m": float(np.median(raw[config][finite])),
                "nmad_before_gate_m": nmad(raw[config][finite]),
                "cells_own": int(own.sum()),
                "cells_strict_common": int(strict.sum()),
                "cells_coarse_common": int(coarse_mask.sum()),
                "cells_coarse_common_before_gate": int(coarse_mask_ungated.sum()),
                "median_coarse_common_before_gate_m": float(np.median(
                    coarse_raw[config][coarse_mask_ungated])),
                "detrended_nmad_coarse_common_before_gate_m": float("nan"),
                "median_own_m": float(np.median(data[own])),
                "nmad_own_m": nmad(data[own]),
                "trend_own_m": trend_amplitude(data, own),
                "detrended_nmad_own_m": float("nan"),
                "median_common_m": float(np.median(data[strict])) if strict.any()
                    else float("nan"),
                "nmad_common_m": nmad(data[strict]),
                "median_coarse_common_m": float(
                    np.median(coarse[config][coarse_mask])),
                "nmad_coarse_common_m": nmad(coarse[config][coarse_mask]),
                "trend_coarse_common_m": trend_amplitude(coarse[config], coarse_mask),
            }
            # the local component is the dispersion left after the trend
            for scope, grid, mask, key in (
                    ("own", data, own, "detrended_nmad_own_m"),
                    ("coarse", coarse[config], coarse_mask,
                     "detrended_nmad_coarse_common_m"),
                    ("coarse_raw", coarse_raw[config], coarse_mask_ungated,
                     "detrended_nmad_coarse_common_before_gate_m")):
                indices = np.nonzero(mask)
                if indices[0].size < 100:
                    entry[key] = float("nan")
                    continue
                values = grid[indices]
                y = (indices[0] - indices[0].mean()) / max(indices[0].std(), 1.0)
                x = (indices[1] - indices[1].mean()) / max(indices[1].std(), 1.0)
                design = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
                coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
                entry[key] = nmad(values - design @ coefficients)
            rows.append(entry)

        core = [row for row in rows
                if row["dataset"] == dataset and row["method"] != "LoFTR"]

        # how much of the mapped area the aggregated common set spans; an
        # aggregated cell covers BLOCK squared cells of the reference grid
        first = records[0]
        reference_valued = int(round(
            int(np.isfinite(raw[first["config"]]).sum())
            / first["compared_fraction_of_reference"]))
        span = BLOCK * BLOCK / reference_valued
        report[dataset] = {
            "cells_strict_common": int(strict.sum()),
            "cells_strict_common_before_gate": int(strict_ungated.sum()),
            "cells_coarse_common": int(coarse_mask.sum()),
            "cells_coarse_common_before_gate": int(coarse_mask_ungated.sum()),
            "reference_valued_cells": reference_valued,
            "aggregation_block": BLOCK,
            "footprint_share_coarse_common":
                min(1.0, coarse_mask.sum() * span),
            "footprint_share_coarse_common_before_gate":
                min(1.0, coarse_mask_ungated.sum() * span),
            "reference_cells": int(np.isfinite(next(iter(raw.values()))).size),
            "local_component_factor_own": (
                max(r["detrended_nmad_own_m"] for r in core)
                / min(r["detrended_nmad_own_m"] for r in core)),
            "local_component_factor_coarse_common": (
                max(r["detrended_nmad_coarse_common_m"] for r in core)
                / min(r["detrended_nmad_coarse_common_m"] for r in core)),
            "local_component_factor_coarse_common_before_gate": (
                max(r["detrended_nmad_coarse_common_before_gate_m"] for r in core)
                / min(r["detrended_nmad_coarse_common_before_gate_m"] for r in core)),
        }

    with (ANALYSIS / "common_surface_mask.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (ANALYSIS / "common_surface_mask.json").write_text(
        json.dumps({"summary": report, "rows": rows}, indent=2), encoding="utf-8")

    print("wrote common_surface_mask.csv and .json")
    for dataset, entry in report.items():
        print(f"  {dataset}: strict common cells {entry['cells_strict_common']}, "
              f"aggregated common cells {entry['cells_coarse_common']}")
        print(f"    local component factor  own masks "
              f"{entry['local_component_factor_own']:.3f}   "
              f"common aggregated {entry['local_component_factor_coarse_common']:.3f}"
              f"   common before the gate "
              f"{entry['local_component_factor_coarse_common_before_gate']:.3f}")
        print(f"    common cells before the gate "
              f"{entry['cells_coarse_common_before_gate']}, after "
              f"{entry['cells_coarse_common']}")
        print(f"    footprint spanned: before the gate "
              f"{100 * entry['footprint_share_coarse_common_before_gate']:.1f} %, "
              f"after {100 * entry['footprint_share_coarse_common']:.1f} % of "
              f"{entry['reference_valued_cells']} reference cells")
        for row in rows:
            if row["dataset"] != dataset:
                continue
            print(f"    {row['method']:16s} own {1000*row['detrended_nmad_own_m']:6.1f} mm "
                  f"({row['cells_own']:8d} cells)   common "
                  f"{1000*row['detrended_nmad_coarse_common_m']:6.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
