#!/usr/bin/env python3
"""Sparse reconstruction under a single fixed bundle adjustment configuration.

Every configuration is reconstructed with the same camera model, the same
self-calibration settings, the same mapper thresholds and the same image pair
list, so that differences between runs can be attributed to the correspondences
rather than to the reconstruction back end. The random seed is an explicit
argument so that run to run variability can be measured.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pycolmap

from hloc import reconstruction as hloc_reconstruction

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    FEATURES_DIR,
    MATCHES_DIR,
    PROJECT_ROOT,
    SFM_DIR,
    SFM_REPORTS_DIR,
)

DATASETS = {
    "D1": {
        "label": "D1_154_building",
        "images": PROJECT_ROOT / "D1_154_building" / "D1_images",
        "pairs": PROJECT_ROOT / "results" / "pairs" / "D1_154_building_pairs.txt",
    },
    "D2": {
        "label": "D2_111_nadir",
        "images": PROJECT_ROOT / "D2_111_nadir" / "D2_images",
        "pairs": PROJECT_ROOT / "results" / "pairs" / "D2_111_nadir_pairs.txt",
    },
}

# Identical for every configuration. The camera model matches the distortion
# terms carried by the reference bundle adjustment, and interior orientation is
# recovered by self-calibration from a single shared camera per block.
CAMERA_MODEL = "OPENCV"
CAMERA_MODE = pycolmap.CameraMode.SINGLE
MAPPER_OPTIONS = {
    "ba_refine_focal_length": True,
    "ba_refine_principal_point": True,
    "ba_refine_extra_params": True,
    "min_num_matches": 15,
    "multiple_models": True,
    "min_model_size": 3,
}


def summarise(reconstruction, total_images: int) -> dict:
    points = reconstruction.points3D
    track_lengths = np.array([len(p.track.elements) for p in points.values()])
    errors = np.array([p.error for p in points.values()])
    registered = reconstruction.num_reg_images()
    cameras = {}
    for camera in reconstruction.cameras.values():
        cameras[camera.camera_id] = {
            "model": camera.model.name,
            "width": camera.width,
            "height": camera.height,
            "params": [float(v) for v in camera.params],
        }
    return {
        "registered_images": int(registered),
        "total_images": int(total_images),
        "registration_rate": float(registered / total_images),
        "sparse_points": int(len(points)),
        "observations": int(track_lengths.sum()) if len(track_lengths) else 0,
        "mean_track_length": float(track_lengths.mean()) if len(track_lengths) else 0.0,
        "median_track_length": float(np.median(track_lengths))
        if len(track_lengths)
        else 0.0,
        "mean_reprojection_error_px": float(errors.mean()) if len(errors) else 0.0,
        "median_reprojection_error_px": float(np.median(errors)) if len(errors) else 0.0,
        "cameras": cameras,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--config", required=True, help="configuration identifier")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-database", action="store_true")
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    label = spec["label"]
    run_id = args.config if args.seed == 0 else f"{args.config}__seed{args.seed}"

    stem = args.config
    for token in ("-lightglue-", "-nn_ratio-", "-dense-"):
        stem = stem.replace(token, "-")
    features = FEATURES_DIR / label / f"{stem}.h5"
    matches = MATCHES_DIR / label / f"{args.config}.h5"
    sfm_dir = SFM_DIR / label / run_id
    report_path = SFM_REPORTS_DIR / label / f"{run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if report_path.exists() and not args.force:
        print(f"[skip] {label} {run_id} already reconstructed")
        return 0
    for path in (features, matches):
        if not path.exists():
            print(f"[error] missing input {path}")
            return 2

    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)
    sfm_dir.mkdir(parents=True, exist_ok=True)

    pycolmap.set_random_seed(args.seed)
    options = dict(MAPPER_OPTIONS)
    options["mapper"] = {"random_seed": args.seed}

    print(f"[sfm] {label} {run_id}")
    start = time.perf_counter()
    reconstruction = hloc_reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=spec["images"],
        pairs=spec["pairs"],
        features=features,
        matches=matches,
        camera_mode=CAMERA_MODE,
        image_options={"camera_model": CAMERA_MODEL},
        mapper_options=options,
        verbose=False,
    )
    elapsed = time.perf_counter() - start

    total_images = len(list(spec["images"].glob("*.JPG")))
    # The largest model is moved to the run directory, so its source folder is
    # left empty. Remaining folders are the smaller disconnected components.
    models = (
        sorted(p for p in (sfm_dir / "models").glob("*") if p.is_dir())
        if (sfm_dir / "models").exists()
        else []
    )
    component_sizes = [int(reconstruction.num_reg_images())]
    for model_dir in models:
        if not any(model_dir.glob("*.bin")):
            continue
        try:
            component_sizes.append(
                int(pycolmap.Reconstruction(str(model_dir)).num_reg_images())
            )
        except Exception:
            continue

    report = {
        "dataset": label,
        "config": args.config,
        "run_id": run_id,
        "seed": args.seed,
        "camera_model": CAMERA_MODEL,
        "camera_mode": CAMERA_MODE.name,
        "mapper_options": MAPPER_OPTIONS,
        "reconstruction_seconds": elapsed,
        "model_component_count": max(len(models), 1),
        "component_registered_images": sorted(component_sizes, reverse=True),
        **summarise(reconstruction, total_images),
        "pycolmap": pycolmap.__version__,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.keep_database:
        database = sfm_dir / "database.db"
        if database.exists():
            database.unlink()

    print(
        f"[done] {run_id}: registered {report['registered_images']}/{total_images}, "
        f"{report['sparse_points']} points, "
        f"reprojection {report['mean_reprojection_error_px']:.3f} px, "
        f"{elapsed / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
