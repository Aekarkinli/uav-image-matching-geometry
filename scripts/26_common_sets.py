#!/usr/bin/env python3
"""Comparisons on sets every configuration shares.

Two of the comparisons are taken over samples that differ between
configurations. The camera-network error is taken over the
cameras each reconstruction registered, and detector-free matching registers
fewer of them in the convergent block. The extraction resolution is the one each
family was run at, and the detector-free method was run only at the smallest of
them. Both are recomputed here on a shared basis.

The camera-network error is recomputed after realigning every reconstruction on
the cameras that every configuration registered, so that the number compared is
a property of the same object in each case. The comparison at a common working
resolution simply reads the runs that exist at 1024 pixels for every method.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT, REFERENCE_DIR, SFM_REPORTS_DIR  # noqa: E402

ANALYSIS = PROJECT_ROOT / "results" / "analysis"
EVALUATION = PROJECT_ROOT / "results" / "evaluation"
PAIRWISE = PROJECT_ROOT / "results" / "pairwise"
DATASETS = ["D1_154_building", "D2_111_nadir"]
SEED = 12345
REPEATS = 200

SETS = {
    "operating": [
        "rootsift-lightglue-r2048-k8192",
        "rootsift-nn_ratio-r2048-k8192",
        "doghardnet-lightglue-r2048-k8192",
        "superpoint-lightglue-r2048-k8192",
        "aliked-lightglue-r2048-k8192",
        "disk-lightglue-r2048-k8192",
        "loftr-dense-r1024-kall",
    ],
    "common1024": [
        "rootsift-lightglue-r1024-k8192",
        "rootsift-nn_ratio-r1024-k8192",
        "doghardnet-lightglue-r1024-k8192",
        "superpoint-lightglue-r1024-k8192",
        "aliked-lightglue-r1024-k8192",
        "disk-lightglue-r1024-k8192",
        "loftr-dense-r1024-kall",
    ],
}
LABEL = {
    "rootsift-lightglue": "RootSIFT",
    "rootsift-nn_ratio": "RootSIFT, nearest neighbour",
    "doghardnet-lightglue": "DoG + HardNet",
    "superpoint-lightglue": "SuperPoint",
    "aliked-lightglue": "ALIKED",
    "disk-lightglue": "DISK",
    "loftr-dense": "LoFTR",
}


def reference_centres(dataset: str) -> dict:
    path = REFERENCE_DIR / f"{dataset}_reference_poses.csv"
    return {row["camera_label"]: np.array(
        [float(row["east_m"]), float(row["north_m"]), float(row["up_m"])])
        for row in csv.DictReader(path.open(encoding="utf-8"))}


def umeyama(source: np.ndarray, target: np.ndarray):
    centroid_source = source.mean(axis=0)
    centroid_target = target.mean(axis=0)
    a, b = source - centroid_source, target - centroid_target
    u, singular, vt = np.linalg.svd(b.T @ a / len(a))
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.trace(np.diag(singular) @ correction) / ((a ** 2).sum() / len(a)))
    return scale, rotation, centroid_target - scale * rotation @ centroid_source


def held_out(source: np.ndarray, target: np.ndarray) -> float:
    rng = np.random.default_rng(SEED)
    half = len(source) // 2
    pooled = []
    for _ in range(REPEATS):
        order = rng.permutation(len(source))
        scale, rotation, translation = umeyama(source[order[:half]], target[order[:half]])
        moved = scale * (rotation @ source[order[half:]].T).T + translation
        pooled.append(np.linalg.norm(moved - target[order[half:]], axis=1))
    stacked = np.concatenate(pooled)
    return float(np.sqrt((stacked ** 2).mean()))


def recovered_centres(dataset: str, config: str, centres: dict):
    path = EVALUATION / dataset / f"{config}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    per_camera = record.get("per_camera") or {}
    return {label: centres[label] + np.array([
        values["residual_east"], values["residual_north"], values["residual_up"]])
        for label, values in per_camera.items() if label in centres}


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    report = {}

    for tag, configs in SETS.items():
        for dataset in DATASETS:
            centres = reference_centres(dataset)
            recovered = {config: recovered_centres(dataset, config, centres)
                         for config in configs}
            recovered = {k: v for k, v in recovered.items() if v}
            if not recovered:
                continue
            shared = set.intersection(*(set(v) for v in recovered.values()))
            order = [label for label in centres if label in shared]
            target = np.array([centres[label] for label in order])

            rows = []
            for config in configs:
                if config not in recovered:
                    continue
                own = recovered[config]
                own_labels = [label for label in centres if label in own]
                own_source = np.array([own[label] for label in own_labels])
                own_target = np.array([centres[label] for label in own_labels])
                shared_source = np.array([own[label] for label in order])

                sfm = json.loads((SFM_REPORTS_DIR / dataset / f"{config}.json")
                                 .read_text(encoding="utf-8"))
                pairwise = list(csv.DictReader(
                    (PAIRWISE / dataset / f"{config}.csv").open(encoding="utf-8")))
                verified = np.array([int(r["verified_inliers"]) for r in pairwise])
                network = json.loads((EVALUATION / dataset / f"{config}.json")
                                     .read_text(encoding="utf-8"))

                rows.append({
                    "dataset": dataset,
                    "config": config,
                    "method": LABEL["-".join(config.split("-")[:2])],
                    "resolution": 1024 if "r1024" in config else (
                        "native" if "native" in config else 2048),
                    "registered_images": sfm.get("registered_images"),
                    "model_components": sfm.get("model_component_count"),
                    "images_in_largest_model": sfm.get("component_registered_images"),
                    "cameras_own": len(own_labels),
                    "cameras_shared": len(order),
                    "median_verified_correspondences": float(np.median(verified)),
                    "failed_pairs_pct": 100.0 * float(np.mean(verified < 15)),
                    "mean_track_length": sfm.get("mean_track_length"),
                    "reprojection_error_px": sfm.get("mean_reprojection_error_px"),
                    "held_out_rmse_own_m": held_out(own_source, own_target),
                    "held_out_rmse_shared_m": held_out(shared_source, target),
                    "relative_rotation_median_deg":
                        network["relative_rotation_error_deg"]["median"],
                    "relative_rotation_pairs":
                        network["relative_rotation_error_deg"]["count"],
                })
            report[f"{tag}|{dataset}"] = rows

    path = ANALYSIS / "common_sets.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    flat = [row for rows in report.values() for row in rows]
    with (ANALYSIS / "common_sets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)

    print(f"wrote {path.name} and common_sets.csv")
    for key, rows in report.items():
        print(f"  {key}   shared cameras {rows[0]['cameras_shared']}")
        for row in rows:
            print(f"    {row['method']:28s} own {1000*row['held_out_rmse_own_m']:8.1f} mm "
                  f"on {row['cameras_own']:3d} cameras   shared "
                  f"{1000*row['held_out_rmse_shared_m']:8.1f} mm   "
                  f"failed {row['failed_pairs_pct']:5.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
