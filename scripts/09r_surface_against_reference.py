#!/usr/bin/env python3
"""Compare the stored dense surfaces against the current reference solution.

Dense matching does not depend on the reference: each fused cloud is computed
in the frame of its own sparse reconstruction and only then georeferenced
through the reference camera positions and rasterised onto the reference
surface grid. This stage repeats those last two steps for every surface that
the dense stage stored, against whichever reference sits in exports/, so a
change of reference does not require dense stereo to be run again. The
georeferencing, rasterisation and statistics are those of the dense stage.

The sparse model is read from the reconstruction store; image_undistorter
copies its poses unchanged, so it shares the datum of the fused cloud. The
point count and timing of each fused cloud are read from the report the dense
stage wrote, which this stage then rewrites.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DENSE_DIR, PROJECT_ROOT, SFM_DIR  # noqa: E402

SURFACE_DIR = PROJECT_ROOT / "results" / "surface"
ORIGINAL_SURFACE_DIR = SURFACE_DIR

spec = importlib.util.spec_from_file_location(
    "dense", Path(__file__).resolve().parent / "09_dense_and_surface.py")
dense = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dense)


def compare(key: str, config: str) -> dict:
    block = dense.DATASETS[key]
    label = block["label"]
    fused = DENSE_DIR / label / config / "fused.ply"
    model = SFM_DIR / label / config
    original = json.loads((ORIGINAL_SURFACE_DIR / label / f"{config}.json")
                          .read_text(encoding="utf-8"))

    reconstruction = pycolmap.Reconstruction(str(model))
    reference = dense.reference_centres_projected(block["xml"])
    names, source, target = [], [], []
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        stem = Path(image.name).stem
        if stem not in reference:
            continue
        names.append(stem)
        source.append(np.asarray(image.projection_center()))
        target.append(reference[stem])
    scale, rotation, translation = dense.umeyama(np.array(source), np.array(target))
    residual = (scale * (rotation @ np.array(source).T)).T + translation - np.array(target)

    points = dense.read_ply(fused)
    if len(points) != original["fused_points"]:
        raise SystemExit(f"{config}: fused cloud holds {len(points)} points, "
                         f"the dense stage recorded {original['fused_points']}")
    points = (scale * (rotation @ points.T)).T + translation

    with rasterio.open(block["dsm"]) as reference_raster:
        bounds = reference_raster.bounds
        inside = ((points[:, 0] >= bounds.left) & (points[:, 0] <= bounds.right)
                  & (points[:, 1] >= bounds.bottom) & (points[:, 1] <= bounds.top))
        surface = dense.rasterise_max(points[inside], reference_raster.transform,
                                      reference_raster.width, reference_raster.height)
        truth = reference_raster.read(1, masked=True).filled(np.nan)
        profile = reference_raster.profile

    valid = np.isfinite(surface) & np.isfinite(truth)
    differences = (surface - truth)[valid]
    out_raster = SURFACE_DIR / label / f"{config}_difference.tif"
    out_raster.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    with rasterio.open(out_raster, "w", **profile) as handle:
        handle.write(np.where(valid, surface - truth, np.nan).astype("float32"), 1)

    report = {
        "dataset": label,
        "config": config,
        "dense_max_image_size": original["dense_max_image_size"],
        "dense_seconds": original["dense_seconds"],
        "fused_points": int(len(points)),
        "georeference_scale": scale,
        "georeference_camera_rmse_m": float(np.sqrt((residual ** 2).sum(1).mean())),
        "cameras_used": len(names),
        "reference_grid_resolution_m": float(abs(profile["transform"][0])),
        "coverage_reference_fraction": float(np.isfinite(truth).mean()),
        "coverage_model_fraction": float(np.isfinite(surface).mean()),
        "compared_cells": int(valid.sum()),
        "compared_fraction_of_reference": float(valid.sum() / max(np.isfinite(truth).sum(), 1)),
        "height_difference_m": dense.robust(differences),
        "difference_raster": str(out_raster),
    }
    (SURFACE_DIR / label / f"{config}.json").write_text(json.dumps(report, indent=2),
                                                         encoding="utf-8")
    return report


def main() -> int:
    keys = {block["label"]: key for key, block in dense.DATASETS.items()}
    for path in sorted(ORIGINAL_SURFACE_DIR.glob("*/*.json")):
        label, config = path.parent.name, path.stem
        report = compare(keys[label], config)
        stats = report["height_difference_m"]
        print(f"{label} {config}: {report['compared_cells']} cells, median "
              f"{stats['median']:+.3f} m, NMAD {stats['nmad']:.3f} m, georeference "
              f"RMSE {report['georeference_camera_rmse_m']:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
