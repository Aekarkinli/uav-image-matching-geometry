#!/usr/bin/env python3
"""Carry selected configurations through to a surface model and compare them.

Camera-network agreement describes the orientation of the block. It does not
say whether the differences between front ends survive into the product a
survey actually delivers. This stage runs dense image matching on a chosen
sparse reconstruction, georeferences the fused point cloud through the
similarity transform recovered from the reference camera positions, rasterises
it onto the grid of the reference surface model, and reports the height
differences with robust statistics.

Coverage is reported alongside the differences, because a front end that
produces a confident surface over a smaller part of the block is not
equivalent to one that covers all of it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pycolmap
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DENSE_DIR, EVALUATION_DIR, PROJECT_ROOT, SFM_DIR  # noqa: E402

# the COLMAP build that ran dense stereo sat in external/COLMAP; elsewhere the
# binary is taken from the COLMAP_EXECUTABLE environment variable or the path
_BUNDLED = PROJECT_ROOT / "external" / "COLMAP" / "bin" / "colmap.exe"
COLMAP = os.environ.get("COLMAP_EXECUTABLE") or (
    str(_BUNDLED) if _BUNDLED.exists() else "colmap")
REFERENCE_CRS = "EPSG:5256"
ECEF_CRS = "EPSG:4978"

DATASETS = {
    "D1": {
        "label": "D1_154_building",
        "images": PROJECT_ROOT / "D1_154_building" / "D1_images",
        "xml": PROJECT_ROOT / "D1_154_building" / "exports" / "reference_cameras.xml",
        "dsm": PROJECT_ROOT / "D1_154_building" / "exports" / "D1_DSM.tif",
    },
    "D2": {
        "label": "D2_111_nadir",
        "images": PROJECT_ROOT / "D2_111_nadir" / "D2_images",
        "xml": PROJECT_ROOT / "D2_111_nadir" / "exports" / "reference_cameras.xml",
        "dsm": PROJECT_ROOT / "D2_111_nadir" / "exports" / "D2_DSM.tif",
    },
}

SURFACE_DIR = PROJECT_ROOT / "results" / "surface"


def reference_centres_projected(xml_path: Path) -> dict:
    """Reference camera centres in the projected reference system."""
    root = ET.parse(xml_path).getroot()
    chunk = root.find("chunk")
    transform = chunk.find("transform")
    rotation = np.array([float(v) for v in transform.find("rotation").text.split()]).reshape(3, 3)
    translation = np.array([float(v) for v in transform.find("translation").text.split()])
    scale = float(transform.find("scale").text)

    labels, ecef = [], []
    for element in chunk.find("cameras").iter("camera"):
        node = element.find("transform")
        if node is None:
            continue
        local = np.array([float(v) for v in node.text.split()]).reshape(4, 4)
        ecef.append(scale * rotation @ local[:3, 3] + translation)
        labels.append(element.get("label"))
    ecef = np.array(ecef)
    transformer = Transformer.from_crs(ECEF_CRS, REFERENCE_CRS, always_xy=True)
    x, y, z = transformer.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    return {label: np.array([a, b, c]) for label, a, b, c in zip(labels, x, y, z)}


def umeyama(source: np.ndarray, target: np.ndarray):
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    u, singular, vt = np.linalg.svd(xt.T @ xs / len(source))
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1
    rotation = u @ correction @ vt
    scale = float(np.trace(np.diag(singular) @ correction) / ((xs**2).sum() / len(source)))
    return scale, rotation, mu_t - scale * rotation @ mu_s


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"{command[1]} failed, see {log}")


def read_ply(path: Path) -> np.ndarray:
    type_map = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1",
                "int": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}
    with path.open("rb") as handle:
        header = b""
        while b"end_header" not in header:
            header += handle.readline()
        text = header.decode("ascii", errors="replace")
        count = int([l for l in text.splitlines() if l.startswith("element vertex")][0].split()[-1])
        dtype = [(p[2], type_map[p[1]]) for p in
                 (l.split() for l in text.splitlines() if l.startswith("property"))]
        data = np.fromfile(handle, dtype=np.dtype(dtype), count=count)
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(float)


def rasterise_max(points: np.ndarray, transform, width: int, height: int) -> np.ndarray:
    """Highest point per cell, the usual surface-model convention."""
    inverse = ~transform
    cols, rows = inverse * (points[:, 0], points[:, 1])
    cols = np.floor(cols).astype(int)
    rows = np.floor(rows).astype(int)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    grid = np.full((height, width), -np.inf)
    np.maximum.at(grid, (rows[inside], cols[inside]), points[inside, 2])
    return np.where(np.isinf(grid), np.nan, grid)


def robust(values: np.ndarray) -> dict:
    median = float(np.median(values))
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": median,
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "nmad": float(1.4826 * np.median(np.abs(values - median))),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    label = spec["label"]
    report_path = SURFACE_DIR / label / f"{args.config}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists() and not args.force:
        print(f"[skip] {label} {args.config} already done")
        return 0

    sparse = SFM_DIR / label / args.config
    if not (sparse / "images.bin").exists():
        print(f"[error] no sparse model at {sparse}")
        return 2

    workspace = DENSE_DIR / label / args.config
    # Dense stereo caches one depth and normal map per image, and skips any
    # that already exist, so an interrupted run resumes where it stopped.
    resuming = workspace.exists() and (workspace / "stereo").exists()
    if workspace.exists() and args.force:
        shutil.rmtree(workspace)
        resuming = False
    workspace.mkdir(parents=True, exist_ok=True)
    logs = workspace / "logs"

    started = time.perf_counter()
    print(f"[dense] {label} {args.config} at {args.max_image_size} px"
          + (" (resuming)" if resuming else ""))
    if not resuming:
        run([str(COLMAP), "image_undistorter",
             "--image_path", str(spec["images"]),
             "--input_path", str(sparse),
             "--output_path", str(workspace),
             "--output_type", "COLMAP",
             "--max_image_size", str(args.max_image_size)], logs / "undistort.log")
    run([str(COLMAP), "patch_match_stereo",
         "--workspace_path", str(workspace),
         "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "true"], logs / "stereo.log")
    fused = workspace / "fused.ply"
    run([str(COLMAP), "stereo_fusion",
         "--workspace_path", str(workspace),
         "--workspace_format", "COLMAP",
         "--input_type", "geometric",
         "--output_path", str(fused)], logs / "fusion.log")
    dense_seconds = time.perf_counter() - started
    print(f"[dense] fused in {dense_seconds/60:.1f} min")

    # Georeference through the camera positions of the reference solution.
    # The transform is fitted from the model that the depth maps were actually
    # computed from, which image_undistorter copied into the workspace. Reading
    # the input model instead would silently produce a wrong surface if that
    # model were replaced between undistortion and fusion, because two runs of
    # incremental mapping do not share an arbitrary datum.
    undistorted_model = workspace / "sparse"
    if not (undistorted_model / "images.bin").exists():
        print(f"[error] undistorted model missing at {undistorted_model}")
        return 3
    reconstruction = pycolmap.Reconstruction(str(undistorted_model))
    reference = reference_centres_projected(spec["xml"])
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
    scale, rotation, translation = umeyama(np.array(source), np.array(target))
    residual = (scale * (rotation @ np.array(source).T)).T + translation - np.array(target)
    print(f"[georeference] {len(names)} cameras, scale {scale:.6f}, "
          f"camera RMSE {np.sqrt((residual**2).sum(1).mean()):.4f} m")

    points = read_ply(fused)
    points = (scale * (rotation @ points.T)).T + translation

    with rasterio.open(spec["dsm"]) as reference_raster:
        bounds = reference_raster.bounds
        inside = (
            (points[:, 0] >= bounds.left) & (points[:, 0] <= bounds.right)
            & (points[:, 1] >= bounds.bottom) & (points[:, 1] <= bounds.top)
        )
        surface = rasterise_max(points[inside], reference_raster.transform,
                                reference_raster.width, reference_raster.height)
        truth = reference_raster.read(1, masked=True).filled(np.nan)
        profile = reference_raster.profile

    valid = np.isfinite(surface) & np.isfinite(truth)
    differences = (surface - truth)[valid]
    coverage_reference = float(np.isfinite(truth).mean())
    coverage_model = float(np.isfinite(surface).mean())

    out_raster = SURFACE_DIR / label / f"{args.config}_difference.tif"
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    with rasterio.open(out_raster, "w", **profile) as handle:
        handle.write(np.where(valid, surface - truth, np.nan).astype("float32"), 1)

    report = {
        "dataset": label,
        "config": args.config,
        "dense_max_image_size": args.max_image_size,
        "dense_seconds": dense_seconds,
        "fused_points": int(len(points)),
        "georeference_scale": scale,
        "georeference_camera_rmse_m": float(np.sqrt((residual**2).sum(1).mean())),
        "cameras_used": len(names),
        "reference_grid_resolution_m": float(abs(profile["transform"][0])),
        "coverage_reference_fraction": coverage_reference,
        "coverage_model_fraction": coverage_model,
        "compared_cells": int(valid.sum()),
        "compared_fraction_of_reference": float(valid.sum() / max(np.isfinite(truth).sum(), 1)),
        "height_difference_m": robust(differences),
        "difference_raster": str(out_raster),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.keep_workspace:
        for name in ("images", "stereo", "sparse"):
            shutil.rmtree(workspace / name, ignore_errors=True)
        # fused.ply is kept, so a later re-analysis does not repeat dense stereo

    stats = report["height_difference_m"]
    print(f"[surface] {args.config}: {report['compared_cells']} cells, "
          f"median {stats['median']:+.3f} m, NMAD {stats['nmad']:.3f} m, "
          f"RMSE {stats['rmse']:.3f} m, coverage {100*report['compared_fraction_of_reference']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
