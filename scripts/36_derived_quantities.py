#!/usr/bin/env python3
"""Derived quantities that no other script records.

The first is how many of its sixteen constituents an aggregated cell of the
common surface set carries, per configuration. The second is the ratio of the
vertical to the horizontal camera-centre discrepancy over the whole design. The
third is the composition of the two models the detector-free run of the
convergent block returned, whose image sets overlap. The fourth is the best and
worst configuration of each family in the most convergent strata over every
extraction resolution and keypoint limit.

Writes results/analysis/common_surface_constituents.json,
       results/analysis/vertical_to_horizontal_ratio.json,
       results/analysis/detector_free_model_split.json and
       results/analysis/family_scope_convergence.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    EVALUATION_DIR,
    PAIRWISE_DIR,
    PROJECT_ROOT,
    REFERENCE_DIR,
    SFM_DIR,
)

ANALYSIS = PROJECT_ROOT / "results" / "analysis"
SURFACE = PROJECT_ROOT / "results" / "surface"
SUMMARY = PROJECT_ROOT / "results" / "tables" / "surface_summary.json"
DATASETS = ["D1_154_building", "D2_111_nadir"]
CORE_LIMIT = 1.0            # the gate the reported statistics already apply
BLOCK = 4                   # the aggregation the surface maps are drawn at
SEED = re.compile(r"__seed\d+$")
VARIANT = re.compile(r"-c\d+$")
DRAWN = ["rootsift-lightglue-r2048-k8192", "superpoint-lightglue-r2048-k8192",
         "disk-lightglue-r2048-k8192", "aliked-lightglue-r2048-k8192"]
DETECTOR_FREE = "loftr-dense-r1024-kall"


def blocks(grid: np.ndarray, factor: int) -> np.ndarray:
    rows = grid.shape[0] // factor * factor
    cols = grid.shape[1] // factor * factor
    return grid[:rows, :cols].reshape(rows // factor, factor,
                                      cols // factor, factor)


def constituents() -> dict:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    report = {}
    for dataset in DATASETS:
        records = sorted((r for r in summary.values() if r["dataset"] == dataset),
                         key=lambda r: r["config"])
        counts, shape = {}, None
        for record in records:
            path = Path(record["raster"])
            if not path.exists():
                path = SURFACE / dataset / f"{record['config']}_difference.tif"
            with rasterio.open(path) as source:
                data = source.read(1).astype(np.float32)
            shape = data.shape if shape is None else shape
            if data.shape != shape:
                continue
            data[np.abs(data) >= CORE_LIMIT] = np.nan
            counts[record["config"]] = np.isfinite(
                blocks(data, BLOCK)).sum(axis=(1, 3))

        mask = np.ones(next(iter(counts.values())).shape, dtype=bool)
        for count in counts.values():
            mask &= count > 0
        per_configuration = {config: float(np.median(count[mask]))
                             for config, count in sorted(counts.items())}
        stacked = np.stack([counts[c][mask] for c in sorted(counts)])
        report[dataset] = {
            "aggregation_block": BLOCK,
            "common_cells": int(mask.sum()),
            "median_by_configuration": per_configuration,
            "median_per_configuration": float(np.median(
                list(per_configuration.values()))),
            "median_of_the_thinnest_configuration": min(
                per_configuration.values()),
            "median_of_the_per_cell_minimum": float(
                np.median(stacked.min(axis=0))),
        }
        print(f"  {dataset}: median "
              f"{report[dataset]['median_per_configuration']:.0f} of "
              f"{BLOCK * BLOCK} constituents, thinnest configuration "
              f"{report[dataset]['median_of_the_thinnest_configuration']:.0f}")
    return report


def vertical_to_horizontal() -> dict:
    everything, drawn = {}, {}
    for dataset in DATASETS:
        for path in sorted((EVALUATION_DIR / dataset).glob("*.json")):
            config = path.stem
            if SEED.search(config) or VARIANT.search(config):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            cameras = record.get("per_camera")
            if not cameras:
                continue
            east = np.array([c["residual_east"] for c in cameras.values()])
            north = np.array([c["residual_north"] for c in cameras.values()])
            up = np.array([c["residual_up"] for c in cameras.values()])
            ratio = float(np.median(np.abs(up))
                          / np.median(np.hypot(east, north)))
            everything[f"{dataset}/{config}"] = ratio
            if config in DRAWN:
                drawn[f"{dataset}/{config}"] = ratio

    report = {}
    for name, group in (("all_configurations", everything),
                        ("drawn_in_the_residual_figure", drawn)):
        report[name] = {
            "count": len(group),
            "minimum": min(group.values()),
            "maximum": max(group.values()),
            "at_the_minimum": min(group, key=group.get),
            "at_the_maximum": max(group, key=group.get),
            "by_configuration": group,
        }
        print(f"  {name}: {report[name]['minimum']:.4f} to "
              f"{report[name]['maximum']:.4f} over {len(group)} configurations")
    return report


def model_split() -> dict:
    import pycolmap

    root = SFM_DIR / "D1_154_building" / DETECTOR_FREE
    kept = {image.name for image
            in pycolmap.Reconstruction(str(root)).images.values()}
    other = max(
        ({image.name for image in pycolmap.Reconstruction(str(sub)).images.values()}
         for sub in sorted((root / "models").iterdir())
         if (sub / "images.bin").exists() and sub.name != "0"),
        key=len)
    report = {
        "config": DETECTOR_FREE,
        "larger": len(kept),
        "smaller": len(other),
        "shared": len(kept & other),
        "only_in_the_smaller": len(other - kept),
        "union": len(kept | other),
    }
    print(f"  detector-free D1: {report['larger']} kept, {report['smaller']} in "
          f"the second model, {report['only_in_the_smaller']} of those absent "
          f"from the first")
    return report


def family_scope() -> dict:
    """Best and worst configuration of each family over the whole sweep.

    The convergent block at 35 to 50 degrees, over every extraction resolution
    and keypoint limit, beside the values at the adopted setting.
    """
    import csv

    geometry = {}
    path = REFERENCE_DIR / "D1_154_building_pair_geometry.csv"
    for row in csv.DictReader(path.open(encoding="utf-8")):
        geometry[(row["image1"], row["image2"])] = float(
            row["convergence_angle_deg"])

    families = {"difference_of_gaussians": ("rootsift", "doghardnet"),
                "jointly_learned": ("superpoint", "aliked", "disk")}
    worst = {name: (-1.0, "", 0) for name in families}
    best = {name: (101.0, "", 0) for name in families}
    for source in sorted((PAIRWISE_DIR / "D1_154_building").glob("*.csv")):
        if SEED.search(source.stem) or VARIANT.search(source.stem):
            continue
        failed = total = 0
        for row in csv.DictReader(source.open(encoding="utf-8")):
            angle = geometry.get((row["image1"], row["image2"]))
            if angle is None or not 35.0 <= angle < 50.0:
                continue
            total += 1
            if str(row["failed_pair"]).lower() in ("true", "1"):
                failed += 1
        if not total:
            continue
        share = 100.0 * failed / total
        for name, prefixes in families.items():
            if source.stem.split("-")[0] not in prefixes:
                continue
            if share > worst[name][0]:
                worst[name] = (share, source.stem, total - failed)
            if share < best[name][0]:
                best[name] = (share, source.stem, total - failed)

    report = {}
    for name in families:
        report[name] = {
            "class": "35 to 50 degrees of convergence, D1",
            "worst_35_50_pct": worst[name][0],
            "worst_35_50_config": worst[name][1],
            "worst_35_50_recovered": worst[name][2],
            "best_35_50_pct": best[name][0],
            "best_35_50_config": best[name][1],
            "best_35_50_recovered": best[name][2],
        }
        print(f"  {name}: worst {worst[name][0]:.1f} per cent "
              f"({worst[name][1]}), best recovers {best[name][2]} of the class")
    return report


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for name, build in (("common_surface_constituents", constituents),
                        ("vertical_to_horizontal_ratio", vertical_to_horizontal),
                        ("detector_free_model_split", model_split),
                        ("family_scope_convergence", family_scope)):
        print(name)
        (ANALYSIS / f"{name}.json").write_text(
            json.dumps(build(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
