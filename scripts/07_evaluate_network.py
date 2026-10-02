#!/usr/bin/env python3
"""Evaluate recovered camera networks against the reference orientation.

Three views of the same reconstruction are reported, deliberately chosen so
that the weaknesses of each are covered by another.

The first is alignment free. Relative rotation and relative translation
direction are compared pair by pair against the reference exterior
orientation, which requires neither a datum nor a scale and therefore cannot
absorb error into alignment parameters.

The second is a spatial similarity alignment of the camera positions. The
conventional in-sample residual is reported for comparability, together with a
held-out residual in which the transformation is estimated on one half of the
cameras and evaluated on the other. Because the scale is fixed by the fitting
half, a scale inconsistency remains visible in the held-out residual instead of
being absorbed.

The third describes the shape of the residual field. A low order polynomial
surface is fitted to the residual components, and the detrended residual is
reported alongside the fraction of variance the trend explains. A residual
dominated by a smooth trend is a property of the block or of the reference, not
a property of the matcher.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    EVALUATION_DIR,
    PROJECT_ROOT,
    REFERENCE_DIR,
    SFM_DIR,
    SFM_REPORTS_DIR,
)

DATASET_PAIRS = {
    "D1_154_building": PROJECT_ROOT / "results" / "pairs" / "D1_154_building_pairs.txt",
    "D2_111_nadir": PROJECT_ROOT / "results" / "pairs" / "D2_111_nadir_pairs.txt",
}

SPLIT_REPEATS = 200
TREND_DEGREE = 2
RANDOM_SEED = 12345


def load_reference(label: str):
    """Reference camera centres and orientations in the local frame.

    Two position references are returned. The adjusted centres come from the
    reference bundle adjustment, which carried the on-board positions as
    weighted observations. The kinematic centres are the raw exposure
    positions, which differ from the perspective centres by the lever arm.
    """
    path = REFERENCE_DIR / f"{label}_reference_poses.csv"
    adjusted, kinematic, rotations = {}, {}, {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["camera_label"]
            adjusted[name] = np.array(
                [float(row["east_m"]), float(row["north_m"]), float(row["up_m"])]
            )
            try:
                kinematic[name] = np.array(
                    [
                        float(row["rtk_east_m"]),
                        float(row["rtk_north_m"]),
                        float(row["rtk_up_m"]),
                    ]
                )
            except (KeyError, ValueError):
                pass
            rotations[name] = np.array(
                [[float(row[f"r{i}{j}"]) for j in range(3)] for i in range(3)]
            )
    return adjusted, kinematic, rotations


def umeyama(source: np.ndarray, target: np.ndarray):
    """Similarity transform taking source onto target, with scale."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    covariance = xt.T @ xs / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1
    rotation = u @ correction @ vt
    variance = (xs**2).sum() / len(source)
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    translation = mu_t - scale * rotation @ mu_s
    return scale, rotation, translation


def apply_similarity(points, scale, rotation, translation):
    return (scale * (rotation @ points.T)).T + translation


def robust_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    nmad = float(1.4826 * np.median(np.abs(values - median)))
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mean": float(values.mean()),
        "median": median,
        "nmad": nmad,
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "count": int(len(values)),
    }


def rotation_angle(matrix: np.ndarray) -> float:
    trace = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def relative_orientation_error(model_poses, reference_rotations, reference_centres, pairs):
    """Compare relative pose per image pair without aligning the two networks."""
    rotation_errors, direction_errors = [], []
    for name0, name1 in pairs:
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        if stem0 not in model_poses or stem1 not in model_poses:
            continue
        if stem0 not in reference_rotations or stem1 not in reference_rotations:
            continue
        r0, c0 = model_poses[stem0]
        r1, c1 = model_poses[stem1]
        # reference stores camera-to-local, the world-to-camera form is its transpose
        q0 = reference_rotations[stem0].T
        q1 = reference_rotations[stem1].T
        d0, d1 = reference_centres[stem0], reference_centres[stem1]

        rotation_errors.append(rotation_angle((r1 @ r0.T) @ (q1 @ q0.T).T))

        baseline_model = r1 @ (c0 - c1)
        baseline_reference = q1 @ (d0 - d1)
        n0 = np.linalg.norm(baseline_model)
        n1 = np.linalg.norm(baseline_reference)
        if n0 < 1e-9 or n1 < 1e-9:
            continue
        cosine = float(np.clip(np.dot(baseline_model / n0, baseline_reference / n1), -1, 1))
        direction_errors.append(float(np.degrees(np.arccos(cosine))))
    return np.array(rotation_errors), np.array(direction_errors)


def polynomial_design(points: np.ndarray, degree: int) -> np.ndarray:
    east, north = points[:, 0], points[:, 1]
    columns = [np.ones_like(east)]
    for total in range(1, degree + 1):
        for power in range(total + 1):
            columns.append((east**(total - power)) * (north**power))
    return np.stack(columns, axis=1)


def trend_decomposition(reference_points, residuals, degree=TREND_DEGREE):
    design = polynomial_design(reference_points, degree)
    report = {}
    detrended = np.empty_like(residuals)
    for index, axis in enumerate(("east", "north", "up")):
        values = residuals[:, index]
        solution, *_ = np.linalg.lstsq(design, values, rcond=None)
        fitted = design @ solution
        detrended[:, index] = values - fitted
        variance = float(np.var(values))
        report[f"trend_fraction_{axis}"] = (
            float(np.var(fitted) / variance) if variance > 1e-12 else 0.0
        )
    return detrended, report


def evaluate(model_dir: Path, label: str, pairs) -> dict:
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    centres_reference, centres_kinematic, rotations_reference = load_reference(label)

    model_poses = {}
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        pose = image.cam_from_world()
        rotation = pose.rotation.matrix()
        centre = np.asarray(image.projection_center())
        model_poses[Path(image.name).stem] = (rotation, centre)

    shared = sorted(set(model_poses) & set(centres_reference))
    if len(shared) < 10:
        return {"status": "too_few_shared_cameras", "shared": len(shared)}

    source = np.array([model_poses[name][1] for name in shared])
    target = np.array([centres_reference[name] for name in shared])

    kinematic_block = None
    usable = [n for n in shared if n in centres_kinematic
              and np.isfinite(centres_kinematic[n]).all()]
    if len(usable) >= 10:
        src_k = np.array([model_poses[n][1] for n in usable])
        tgt_k = np.array([centres_kinematic[n] for n in usable])
        s_k, r_k, t_k = umeyama(src_k, tgt_k)
        res_k = apply_similarity(src_k, s_k, r_k, t_k) - tgt_k
        kinematic_block = {
            "cameras_compared": len(usable),
            "similarity_scale": s_k,
            "similarity_in_sample": robust_stats(np.linalg.norm(res_k, axis=1)),
        }

    scale, rotation, translation = umeyama(source, target)
    aligned = apply_similarity(source, scale, rotation, translation)
    residuals = aligned - target
    distances = np.linalg.norm(residuals, axis=1)

    rng = np.random.default_rng(RANDOM_SEED)
    held_out = []
    scales = []
    count = len(shared)
    half = count // 2
    for _ in range(SPLIT_REPEATS):
        order = rng.permutation(count)
        fit_index, test_index = order[:half], order[half:]
        s, r, t = umeyama(source[fit_index], target[fit_index])
        scales.append(s)
        predicted = apply_similarity(source[test_index], s, r, t)
        held_out.append(np.linalg.norm(predicted - target[test_index], axis=1))
    held_out = np.concatenate(held_out)

    detrended, trend = trend_decomposition(target, residuals)
    detrended_distance = np.linalg.norm(detrended, axis=1)

    # A single grossly misoriented camera moves the root mean square residual
    # a long way while leaving the robust descriptors untouched, so the two
    # phenomena are reported separately rather than one masking the other.
    median = float(np.median(distances))
    nmad = float(1.4826 * np.median(np.abs(distances - median)))
    gross = {
        "beyond_3_nmad": int((distances > median + 3 * nmad).sum()),
        "beyond_10_nmad": int((distances > median + 10 * nmad).sum()),
        "worst_camera": shared[int(np.argmax(distances))],
        "worst_residual_m": float(distances.max()),
    }

    rotation_errors, direction_errors = relative_orientation_error(
        model_poses, rotations_reference, centres_reference, pairs
    )

    return {
        "status": "ok",
        "cameras_compared": len(shared),
        "similarity_scale": scale,
        "similarity_in_sample": robust_stats(distances),
        "gross_errors": gross,
        "against_kinematic_positions": kinematic_block,
        "similarity_held_out": robust_stats(held_out),
        "held_out_scale_spread": float(np.std(scales) / np.mean(scales)),
        "residual_detrended": robust_stats(detrended_distance),
        **trend,
        "relative_rotation_error_deg": robust_stats(rotation_errors)
        if len(rotation_errors)
        else None,
        "relative_direction_error_deg": robust_stats(direction_errors)
        if len(direction_errors)
        else None,
        "per_camera": {
            name: {
                "residual_east": float(residuals[i, 0]),
                "residual_north": float(residuals[i, 1]),
                "residual_up": float(residuals[i, 2]),
                "residual_3d": float(distances[i]),
            }
            for i, name in enumerate(shared)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    pair_cache = {
        label: [line.split() for line in path.read_text().splitlines() if line.strip()]
        for label, path in DATASET_PAIRS.items()
    }

    reports = sorted(SFM_REPORTS_DIR.glob("*/*.json"))
    print(f"{len(reports)} reconstructions to evaluate")
    done = 0
    for report_path in reports:
        label = report_path.parent.name
        run_id = report_path.stem
        out_path = EVALUATION_DIR / label / f"{run_id}.json"
        if out_path.exists() and not args.force:
            continue
        model_dir = SFM_DIR / label / run_id
        if not (model_dir / "images.bin").exists():
            print(f"[skip] {label} {run_id}: no model on disk")
            continue
        result = evaluate(model_dir, label, pair_cache[label])
        result["dataset"] = label
        result["run_id"] = run_id
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        done += 1
        if result["status"] == "ok":
            print(
                f"[ok] {label} {run_id}: "
                f"held-out RMSE {result['similarity_held_out']['rmse']:.3f} m, "
                f"in-sample {result['similarity_in_sample']['rmse']:.3f} m, "
                f"detrended {result['residual_detrended']['rmse']:.3f} m, "
                f"rel. rotation median {result['relative_rotation_error_deg']['median']:.3f} deg"
            )
        else:
            print(f"[warn] {label} {run_id}: {result['status']}")

    print(f"\nevaluated {done} reconstructions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
