#!/usr/bin/env python3
"""Where the on-board exposure positions and the reference centres disagree.

Scoring against raw on-board positions can reverse an ordering when those
positions carry errors with spatial structure. The on-board positions and the
reference perspective centres are both stored, so their difference can be
decomposed into a constant offset, a component shared by the tilted subset and
a remainder, and the contribution of that subset to the comparison can be
isolated by removing it.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT, REFERENCE_DIR  # noqa: E402

ANALYSIS = PROJECT_ROOT / "results" / "analysis"
EVALUATION = PROJECT_ROOT / "results" / "evaluation"
DATASETS = ["D1_154_building", "D2_111_nadir"]
TILT_THRESHOLD = 20.0

# the tolerance sensitivity run carries a trailing marker and is not one of the
# thirty-six configurations of the design, so it stays out of the ordering
VARIANT = re.compile(r"-c\d+$")


def poses(dataset: str) -> list[dict]:
    path = REFERENCE_DIR / f"{dataset}_reference_poses.csv"
    return list(csv.DictReader(path.open(encoding="utf-8")))


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def umeyama(source: np.ndarray, target: np.ndarray):
    centroid_source = source.mean(axis=0)
    centroid_target = target.mean(axis=0)
    a = source - centroid_source
    b = target - centroid_target
    covariance = b.T @ a / len(a)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    variance = (a ** 2).sum() / len(a)
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    translation = centroid_target - scale * rotation @ centroid_source
    return scale, rotation, translation


def held_out_rmse(source: np.ndarray, target: np.ndarray, seed: int = 12345,
                  repeats: int = 200) -> float:
    rng = np.random.default_rng(seed)
    residuals = []
    half = len(source) // 2
    for _ in range(repeats):
        order = rng.permutation(len(source))
        fit, test = order[:half], order[half:]
        scale, rotation, translation = umeyama(source[fit], target[fit])
        moved = scale * (rotation @ source[test].T).T + translation
        residuals.append(np.linalg.norm(moved - target[test], axis=1))
    pooled = np.concatenate(residuals)
    return float(np.sqrt((pooled ** 2).mean()))


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    report = {}
    per_configuration = []

    for dataset in DATASETS:
        rows = poses(dataset)
        labels = [row["camera_label"] for row in rows]
        reference = np.array([[float(row["east_m"]), float(row["north_m"]),
                               float(row["up_m"])] for row in rows])
        onboard = np.array([[float(row["rtk_east_m"]), float(row["rtk_north_m"]),
                             float(row["rtk_up_m"])] for row in rows])
        tilt = np.array([float(row["tilt_from_nadir_deg"]) for row in rows])
        azimuth = np.array([float(row["viewing_azimuth_deg"]) for row in rows])

        difference = onboard - reference
        constant = difference.mean(axis=0)
        residual = difference - constant
        total = float((difference ** 2).sum())
        tilted = tilt >= TILT_THRESHOLD

        group_bias = np.zeros_like(residual)
        for member in (tilted, ~tilted):
            if member.any():
                group_bias[member] = residual[member].mean(axis=0)
        local = residual - group_bias
        entry = {
            "cameras": len(rows),
            "rms_3d_mm": 1000.0 * float(np.sqrt((difference ** 2).sum(axis=1).mean())),
            "constant_offset_mm": (1000.0 * constant).round(2).tolist(),
            "share_constant": float((len(rows) * (constant ** 2).sum()) / total),
            "share_tilt_group": float((group_bias ** 2).sum() / total),
            "share_local": float((local ** 2).sum() / total),
            "tilted_cameras": int(tilted.sum()),
            "share_of_squared_difference_from_tilted":
                float((difference[tilted] ** 2).sum() / total) if tilted.any() else 0.0,
            "median_difference_tilted_mm":
                1000.0 * float(np.median(np.linalg.norm(difference[tilted], axis=1)))
                if tilted.any() else float("nan"),
            "median_difference_nadir_mm":
                1000.0 * float(np.median(np.linalg.norm(difference[~tilted], axis=1))),
        }

        # how much of the ordering reversal survives when the tilted images go
        index = {label: position for position, label in enumerate(labels)}
        against_reference, against_onboard, names, camera_counts = [], [], [], []
        against_reference_nadir, against_onboard_nadir = [], []
        for path in sorted((EVALUATION / dataset).glob("*.json")):
            if "__seed" in path.name or VARIANT.search(path.stem):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            per_camera = record.get("per_camera") or {}
            # a configuration that registered only part of the block is kept,
            # scored on the cameras it has, with that count recorded
            if len(per_camera) < 0.5 * len(rows):
                continue
            order = [label for label in labels if label in per_camera]
            recovered = np.array([
                reference[index[label]] + np.array([
                    per_camera[label]["residual_east"],
                    per_camera[label]["residual_north"],
                    per_camera[label]["residual_up"]])
                for label in order])
            # the residual is the aligned centre minus the reference centre, so
            # adding it back recovers the aligned reconstruction, which is then
            # realigned freely
            target_reference = np.array([reference[index[l]] for l in order])
            target_onboard = np.array([onboard[index[l]] for l in order])
            nadir = np.array([tilt[index[l]] < TILT_THRESHOLD for l in order])
            names.append(path.stem)
            camera_counts.append(len(order))
            against_reference.append(held_out_rmse(recovered, target_reference))
            against_onboard.append(held_out_rmse(recovered, target_onboard))
            if nadir.sum() > 20:
                against_reference_nadir.append(
                    held_out_rmse(recovered[nadir], target_reference[nadir]))
                against_onboard_nadir.append(
                    held_out_rmse(recovered[nadir], target_onboard[nadir]))

        # the per-configuration values, so the ranking figures can use one measure
        per_configuration.extend(
            {"dataset": dataset, "config": name,
             "held_out_against_reference_m": against,
             "held_out_against_onboard_m": onboard_value,
             "cameras": cameras}
            for name, against, onboard_value, cameras
            in zip(names, against_reference, against_onboard, camera_counts))

        entry["configurations_compared"] = len(names)
        entry["spearman_reference_vs_onboard"] = spearman(
            against_reference, against_onboard)
        if against_reference_nadir:
            entry["spearman_reference_vs_onboard_nadir_only"] = spearman(
                against_reference_nadir, against_onboard_nadir)
            entry["nadir_cameras_used"] = int(
                sum(tilt[index[l]] < TILT_THRESHOLD for l in labels))
        report[dataset] = entry

    path = ANALYSIS / "onboard_decomposition.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    axes = ANALYSIS / "held_out_axes.csv"
    with axes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_configuration[0].keys()))
        writer.writeheader()
        writer.writerows(per_configuration)

    print(f"wrote {path.name} and {axes.name} "
          f"({len(per_configuration)} configurations)")
    for dataset, entry in report.items():
        print(f"  {dataset}: RMS {entry['rms_3d_mm']:.1f} mm over "
              f"{entry['cameras']} cameras")
        print(f"    shares  constant {entry['share_constant']:.3f}  "
              f"tilt group {entry['share_tilt_group']:.3f}  "
              f"local {entry['share_local']:.3f}")
        print(f"    tilted {entry['tilted_cameras']} cameras carry "
              f"{entry['share_of_squared_difference_from_tilted']:.3f} of the "
              f"squared difference; median tilted "
              f"{entry['median_difference_tilted_mm']:.0f} mm vs nadir "
              f"{entry['median_difference_nadir_mm']:.0f} mm")
        print(f"    rho(reference, on-board) over "
              f"{entry['configurations_compared']} configurations = "
              f"{entry['spearman_reference_vs_onboard']:+.3f}"
              + (f", nadir cameras only "
                 f"{entry['spearman_reference_vs_onboard_nadir_only']:+.3f}"
                 if "spearman_reference_vs_onboard_nadir_only" in entry else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
