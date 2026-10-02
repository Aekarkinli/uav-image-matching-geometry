#!/usr/bin/env python3
"""Reference epipolar support of every candidate pair.

A pair enters the block when fifteen correspondences survive the robust
estimator. That test asks only whether a self-consistent epipolar geometry
exists among the putative matches; it cannot tell a correct geometry from a
consistent wrong one. This script answers the question directly and without
estimating anything: every putative correspondence of every candidate pair is
measured against the epipolar geometry implied by the reference exterior
orientations and the reference calibration, and a correspondence is called
reference consistent when its Sampson distance falls within two pixels, the
threshold the geometric verification also uses.

A pair is then

  accepted    when the pipeline retained at least fifteen verified matches,
  supportable when at least fifteen of its putative matches are reference
              consistent,

and the two classifications give a precision and a recall for the acceptance
rule, per method and per class of pair geometry. The same pass writes the
per-pair counts, so the continuous failure curve and the image-clustered
bootstrap can be built from them without touching the stores again.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    PAIRWISE_DIR,
    PROJECT_ROOT,
    REFERENCE_DIR,
    WORKSPACE,
)

OUT = PROJECT_ROOT / "results" / "analysis"
DATASETS = ["D1_154_building", "D2_111_nadir"]

# the seven configurations that make up the comparison at the adopted setting,
# and the same seven at the one extraction resolution every method was run at
OPERATING_POINT = [
    "rootsift-lightglue-r2048-k8192",
    "rootsift-nn_ratio-r2048-k8192",
    "doghardnet-lightglue-r2048-k8192",
    "superpoint-lightglue-r2048-k8192",
    "aliked-lightglue-r2048-k8192",
    "disk-lightglue-r2048-k8192",
    "loftr-dense-r1024-kall",
]
COMMON_RESOLUTION = [
    "rootsift-lightglue-r1024-k8192",
    "rootsift-nn_ratio-r1024-k8192",
    "doghardnet-lightglue-r1024-k8192",
    "superpoint-lightglue-r1024-k8192",
    "aliked-lightglue-r1024-k8192",
    "disk-lightglue-r1024-k8192",
    "loftr-dense-r1024-kall",
]

SAMPSON_PX = 2.0          # the threshold the pipeline applies to its own estimate
SUPPORT_MIN = 15          # the count at which the pipeline accepts a pair


def reference_calibration(dataset: str) -> dict:
    """The adjusted interior orientation of the reference solution."""
    path = PROJECT_ROOT / dataset / "exports" / "sensor_calibration.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    # the first row of D1 is an empty placeholder sensor carrying no images
    row = max(rows, key=lambda r: float(r["f"]))
    width, height = float(row["width"]), float(row["height"])
    return {
        "f": float(row["f"]),
        "cx": width / 2.0 + float(row["cx"]),
        "cy": height / 2.0 + float(row["cy"]),
        "k1": float(row["k1"]), "k2": float(row["k2"]), "k3": float(row["k3"]),
        # decentring terms in the OpenCV order of Equation 1
        "p1": float(row["p1"]), "p2": float(row["p2"]),
    }


def normalise(points: np.ndarray, cal: dict) -> np.ndarray:
    """Pixel coordinates to distortion-free normalised camera coordinates."""
    x = (points[:, 0] - cal["cx"]) / cal["f"]
    y = (points[:, 1] - cal["cy"]) / cal["f"]
    u, v = x.copy(), y.copy()
    for _ in range(12):                       # fixed point inversion of the model
        r2 = u * u + v * v
        radial = 1.0 + cal["k1"] * r2 + cal["k2"] * r2 * r2 + cal["k3"] * r2 ** 3
        dx = 2.0 * cal["p1"] * u * v + cal["p2"] * (r2 + 2.0 * u * u)
        dy = cal["p1"] * (r2 + 2.0 * v * v) + 2.0 * cal["p2"] * u * v
        u = (x - dx) / radial
        v = (y - dy) / radial
    return np.stack([u, v], axis=1)


def reference_poses(dataset: str) -> dict:
    path = REFERENCE_DIR / f"{dataset}_reference_poses.csv"
    poses = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        rotation = np.array([[float(row[f"r{i}{j}"]) for j in range(3)]
                             for i in range(3)])
        centre = np.array([float(row["east_m"]), float(row["north_m"]),
                           float(row["up_m"])])
        poses[row["camera_label"]] = (rotation, centre)
    return poses


def essential(pose_i, pose_k) -> np.ndarray:
    """Essential matrix taking a normalised point of image i to image k."""
    (r_i, c_i), (r_k, c_k) = pose_i, pose_k
    relative_rotation = r_k.T @ r_i
    translation = r_k.T @ (c_i - c_k)
    skew = np.array([[0.0, -translation[2], translation[1]],
                     [translation[2], 0.0, -translation[0]],
                     [-translation[1], translation[0], 0.0]])
    return skew @ relative_rotation


def sampson(points_i: np.ndarray, points_k: np.ndarray,
            matrix: np.ndarray) -> np.ndarray:
    a = np.concatenate([points_i, np.ones((len(points_i), 1))], axis=1)
    b = np.concatenate([points_k, np.ones((len(points_k), 1))], axis=1)
    line_k = a @ matrix.T
    line_i = b @ matrix
    numerator = np.einsum("ij,ij->i", b, line_k) ** 2
    denominator = (line_k[:, 0] ** 2 + line_k[:, 1] ** 2
                   + line_i[:, 0] ** 2 + line_i[:, 1] ** 2)
    return numerator / np.maximum(denominator, 1e-24)


def feature_store(dataset: str, config: str) -> Path:
    stem = config.replace("-lightglue", "").replace("-nn_ratio", "")
    stem = stem.replace("-dense", "")
    return WORKSPACE / "features" / dataset / f"{stem}.h5"


def matched_points(group, load, first: str, second: str):
    """The matched pixel coordinates of one pair, in the order (first, second).

    Both families are read the same way. The compaction step gives every method
    one keypoint list per image in native image coordinates and an assignment
    from the first image into the second, which is also the convention the
    per-pair statistics of the experiment were written with.
    """
    assignment = group["matches0"][()]
    keep = np.flatnonzero(assignment > -1)
    if keep.size == 0:
        return None, None
    source = load(first)
    target = load(second)
    return source[keep, :2], target[assignment[keep], :2]


def analyse(dataset: str, configs: list[str], tag: str) -> None:
    calibration = reference_calibration(dataset)
    poses = reference_poses(dataset)
    geometry = {}
    for row in csv.DictReader(
            (REFERENCE_DIR / f"{dataset}_pair_geometry.csv").open(encoding="utf-8")):
        key = tuple(sorted((Path(row["image1"]).stem, Path(row["image2"]).stem)))
        geometry[key] = row

    records = []
    for config in configs:
        pairwise = PAIRWISE_DIR / dataset / f"{config}.csv"
        if not pairwise.exists():
            print(f"  missing {pairwise.name}")
            continue
        rows = list(csv.DictReader(pairwise.open(encoding="utf-8")))
        matches_path = WORKSPACE / "matches" / dataset / f"{config}.h5"
        features_path = feature_store(dataset, config)
        if not matches_path.exists() or not features_path.exists():
            print(f"  missing store for {config}")
            continue

        with h5py.File(features_path, "r") as features, \
                h5py.File(matches_path, "r") as matches:
            keypoints = {}

            def load(name: str) -> np.ndarray:
                if name not in keypoints:
                    keypoints[name] = features[name]["keypoints"][()].astype(float)
                return keypoints[name]

            for row in rows:
                first, second = row["image1"], row["image2"]
                label_i, label_k = Path(first).stem, Path(second).stem
                if label_i not in poses or label_k not in poses:
                    continue
                group = matches.get(first, {}).get(second) if first in matches else None
                if group is None and second in matches:
                    group = matches[second].get(first)
                    swap = True
                else:
                    swap = False
                if group is None:
                    continue

                stored_first, stored_second = (second, first) if swap else (first, second)
                source, target = matched_points(group, load, stored_first, stored_second)
                if source is None:
                    consistent, putative_count = 0, 0
                else:
                    pose_a = poses[Path(stored_first).stem]
                    pose_b = poses[Path(stored_second).stem]
                    a = normalise(source, calibration)
                    b = normalise(target, calibration)
                    matrix = essential(pose_a, pose_b)
                    distance = sampson(a, b, matrix) * calibration["f"] ** 2
                    consistent = int(np.count_nonzero(distance <= SAMPSON_PX ** 2))
                    putative_count = int(len(source))

                pair = geometry.get(tuple(sorted((label_i, label_k))), {})
                records.append({
                    "dataset": dataset,
                    "config": config,
                    "pair_index": int(row["pair_index"]),
                    "image1": label_i,
                    "image2": label_k,
                    "baseline_m": float(pair.get("baseline_m", "nan")),
                    "convergence_angle_deg": float(
                        pair.get("convergence_angle_deg", "nan")),
                    "mean_camera_height_m": float(
                        pair.get("mean_camera_height_m", "nan")),
                    "putative": putative_count,
                    "verified_inliers": int(row["verified_inliers"]),
                    "reference_consistent": consistent,
                    "accepted": int(row["verified_inliers"]) >= SUPPORT_MIN,
                    "supportable": consistent >= SUPPORT_MIN,
                })
        print(f"  {dataset} {config}: {len(rows)} pairs")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"pair_pose_support_{tag}_{dataset}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"  wrote {path.name} ({len(records)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["operating", "common", "both"],
                        default="both")
    args = parser.parse_args()

    plans = []
    if args.set in ("operating", "both"):
        plans.append((OPERATING_POINT, "operating"))
    if args.set in ("common", "both"):
        plans.append((COMMON_RESOLUTION, "common1024"))

    for configs, tag in plans:
        for dataset in DATASETS:
            analyse(dataset, configs, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
