#!/usr/bin/env python3
"""Detector-free matching, evaluated under the same protocol as the rest.

A detector-free matcher produces correspondences directly from an image pair
without committing to a keypoint set beforehand, so the keypoint budget of the
other configurations has no counterpart here. Everything else is held fixed:
the same image pair list, the same geometric verification with the same
threshold, and the same bundle adjustment afterwards.

The extraction resolution cannot be held fixed either. The attention over dense
feature maps does not fit in the available graphics memory beyond a long side
of about 1024 pixels, so this family is compared at its own operating point
rather than at the reference one.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from hloc import match_dense

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    FEATURES_DIR,
    MANIFESTS_DIR,
    MATCHES_DIR,
    PAIRWISE_DIR,
    PROJECT_ROOT,
    ensure_workspace,
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

VERIFIER = {
    "method": "USAC_MAGSAC fundamental matrix",
    "threshold_px": 2.0,
    "confidence": 0.9999,
    "max_iters": 10000,
    "min_matches": 8,
}
FAILED_PAIR_INLIER_THRESHOLD = 15
COVERAGE_GRID = 8


def verify(points0: np.ndarray, points1: np.ndarray):
    if len(points0) < VERIFIER["min_matches"]:
        return np.zeros(len(points0), dtype=bool), "too_few_matches"
    try:
        matrix, mask = cv2.findFundamentalMat(
            points0.astype(np.float64), points1.astype(np.float64),
            cv2.USAC_MAGSAC, VERIFIER["threshold_px"],
            VERIFIER["confidence"], VERIFIER["max_iters"])
    except cv2.error:
        return np.zeros(len(points0), dtype=bool), "degenerate"
    if matrix is None or mask is None:
        return np.zeros(len(points0), dtype=bool), "no_model"
    return mask.ravel().astype(bool), "ok"


def coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) == 0:
        return 0.0
    col = np.clip((points[:, 0] / width * COVERAGE_GRID).astype(int), 0, COVERAGE_GRID - 1)
    row = np.clip((points[:, 1] / height * COVERAGE_GRID).astype(int), 0, COVERAGE_GRID - 1)
    return float(len(set(zip(row.tolist(), col.tolist()))) / COVERAGE_GRID**2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--resize", type=int, default=1024)
    # A detector-free matcher has no keypoint budget. Capping the keypoints an
    # image accumulates over its pairs discards most of the correspondences it
    # produced, so the cap is off by default and the absence of a counterpart to
    # the budget is reported instead of imposing one.
    parser.add_argument("--max-keypoints", type=int, default=0,
                        help="0 keeps every keypoint, which is the intended setting")
    # A detector-free matcher re-detects points independently in every pair, so
    # the toolkit merges them onto one keypoint set per image within a spatial
    # tolerance. That tolerance governs whether a scene point seen in many
    # images becomes one long track or several short ones, so it is exposed
    # here rather than left at its default.
    parser.add_argument("--cell-size", type=int, default=1)
    parser.add_argument("--max-error", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_workspace()
    spec = DATASETS[args.dataset]
    label = spec["label"]
    cap = args.max_keypoints if args.max_keypoints > 0 else None
    budget_tag = "kall" if cap is None else f"k{cap}"
    merge_tag = "" if args.cell_size == 1 and args.max_error == 1.0 else f"-c{args.cell_size}"
    config = f"loftr-dense-r{args.resize}-{budget_tag}{merge_tag}"
    features_path = FEATURES_DIR / label / f"loftr-r{args.resize}-{budget_tag}{merge_tag}.h5"
    matches_path = MATCHES_DIR / label / f"{config}.h5"
    metrics_path = PAIRWISE_DIR / label / f"{config}.csv"
    manifest_path = MANIFESTS_DIR / label / f"{config}.json"
    for path in (features_path, matches_path, metrics_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() and not args.force:
        print(f"[skip] {label} {config} already complete")
        return 0

    conf = dict(match_dense.confs["loftr"])
    conf["preprocessing"] = {**conf["preprocessing"], "resize_max": args.resize}
    conf["cell_size"] = args.cell_size
    conf["max_error"] = args.max_error

    def outputs_complete() -> bool:
        if not (features_path.exists() and matches_path.exists()):
            return False
        try:
            with h5py.File(features_path, "r") as handle:
                first = next(iter(handle), None)
                return first is not None and "keypoints" in handle[first]
        except OSError:
            return False

    print(f"[dense match] {label} {config}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if outputs_complete() and not args.force:
        print("[dense match] reusing the existing correspondences")
        elapsed = float("nan")
        peak = float("nan")
    else:
        match_dense.main(
            conf=conf,
            pairs=spec["pairs"],
            image_dir=spec["images"],
            features=features_path,
            matches=matches_path,
            max_kps=cap,
            overwrite=True,
        )
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"[dense match] finished in {elapsed/60:.1f} min, peak {peak:.2f} GB")

    # Same verification and the same per-pair statistics as every other run.
    pairs = [line.split() for line in spec["pairs"].read_text().splitlines() if line.strip()]
    rows = []
    # The dense store holds only keypoints, so the frame size is read once from
    # the imagery. Every image in both blocks comes from the same sensor.
    probe = cv2.imread(str(next(iter(spec["images"].glob("*.JPG")))), cv2.IMREAD_REDUCED_COLOR_8)
    frame = (probe.shape[1] * 8, probe.shape[0] * 8)
    with h5py.File(features_path, "r") as features, h5py.File(matches_path, "r") as matches:
        keypoints = {name: features[name]["keypoints"][()] for name in features}
        sizes = {name: frame for name in features}
        for index, (name0, name1) in enumerate(pairs, start=1):
            group = matches[name0][name1] if name0 in matches and name1 in matches[name0] else None
            if group is None:
                continue
            record = group["matches0"][()]
            scores = group["matching_scores0"][()]
            valid = np.flatnonzero(record != -1)
            kp0, kp1 = keypoints[name0], keypoints[name1]
            points0 = kp0[valid]
            points1 = kp1[record[valid]]
            inlier_mask, status = verify(points0, points1)
            inliers = int(inlier_mask.sum())
            width0, height0 = int(sizes[name0][0]), int(sizes[name0][1])
            width1, height1 = int(sizes[name1][0]), int(sizes[name1][1])
            rows.append({
                "dataset": label, "config": config, "front_end": "loftr",
                "matcher": "dense", "working_resolution": args.resize,
                "keypoint_budget": cap if cap is not None else "", "pair_index": index,
                "image1": name0, "image2": name1,
                "keypoints_image1": len(kp0), "keypoints_image2": len(kp1),
                "raw_matches": int(len(valid)), "verified_inliers": inliers,
                "inlier_ratio": (inliers / len(valid)) if len(valid) else 0.0,
                "inliers_per_1000_keypoints": (
                    1000.0 * inliers / min(len(kp0), len(kp1)) if min(len(kp0), len(kp1)) else 0.0),
                "coverage_image1": coverage(points0[inlier_mask], width0, height0) if len(valid) else 0.0,
                "coverage_image2": coverage(points1[inlier_mask], width1, height1) if len(valid) else 0.0,
                "mean_match_score": float(np.mean(scores[valid])) if len(valid) else 0.0,
                "failed_pair": inliers < FAILED_PAIR_INLIER_THRESHOLD,
                "verification_status": status,
                "seconds": elapsed / max(len(pairs), 1),
            })

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    inliers = np.array([r["verified_inliers"] for r in rows], dtype=float)
    statuses: dict = {}
    for row in rows:
        statuses[row["verification_status"]] = statuses.get(row["verification_status"], 0) + 1
    manifest = {
        "config": config, "dataset": label, "front_end": "loftr", "matcher": "dense",
        "working_resolution": args.resize, "keypoint_budget": cap,
        "detector_free": True,
        "keypoint_budget_note": "not applicable to a detector-free matcher",
        "merge_cell_size_px": args.cell_size,
        "merge_max_error_px": args.max_error,
        "verifier": VERIFIER,
        "failed_pair_inlier_threshold": FAILED_PAIR_INLIER_THRESHOLD,
        "pairs": len(pairs),
        "extraction": {"reused": False, "keypoints_median": float(
            np.median([len(v) for v in keypoints.values()]))},
        "matching_seconds_total": elapsed,
        "matching_seconds_per_pair_median": elapsed / max(len(pairs), 1),
        "matching_peak_gpu_mb": peak * 1000.0,
        "frame_size": list(frame),
        "median_verified_inliers": float(np.median(inliers)),
        "mean_verified_inliers": float(np.mean(inliers)),
        "failed_pair_fraction": float(np.mean(inliers < FAILED_PAIR_INLIER_THRESHOLD)),
        "verification_status_counts": statuses,
        "features_path": str(features_path), "matches_path": str(matches_path),
        "metrics_path": str(metrics_path),
        "torch": torch.__version__, "opencv": cv2.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] {config}: median inliers {manifest['median_verified_inliers']:.0f}, "
          f"failed pairs {100*manifest['failed_pair_fraction']:.1f}%, "
          f"{manifest['matching_seconds_per_pair_median']:.3f} s/pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
