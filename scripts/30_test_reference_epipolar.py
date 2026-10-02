#!/usr/bin/env python3
"""Check the reference epipolar geometry against a case whose answer is known.

The reference-consistency test rests on one matrix. Two things have to be true
for it to mean anything: the rotations stored with the reference poses must be
in the sense the formula assumes, and the formula must then reproduce an exact
epipolar constraint. Both are checked here, the first against the archive and
the second against a synthetic pair of cameras and a cloud of points whose
geometry is constructed rather than measured.

Run it directly. It prints the result of each check and exits non-zero if
either fails.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import REFERENCE_DIR  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "pair_support", HERE / "22_pair_pose_support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)

DATASETS = ["D1_154_building", "D2_111_nadir"]
TOLERANCE = 1e-12


def rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    skew = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    return (np.eye(3) + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * skew @ skew)


def synthetic() -> tuple[bool, str]:
    """Two constructed cameras and a cloud of points, with no measurement in it.

    The rotations are built as camera-to-world, which is the sense the archive
    stores, so a point seen by both cameras must satisfy the epipolar
    constraint exactly under the matrix the pipeline builds.
    """
    rng = np.random.default_rng(7)
    r_i = rotation_from_axis_angle(np.array([0.2, -0.5, 1.0]), 0.7)
    r_k = rotation_from_axis_angle(np.array([-0.7, 0.3, 0.4]), -0.45)
    c_i = np.array([12.0, -4.0, 60.0])
    c_k = np.array([-7.0, 9.0, 63.0])

    # build the cloud in the first camera's own frame, then place it in world
    forward = np.hstack([rng.normal(scale=0.25, size=(400, 2)),
                         rng.uniform(40.0, 90.0, size=(400, 1))])
    forward[:, :2] *= forward[:, 2:3]
    points = (r_i @ forward.T).T + c_i

    # camera-to-world means X = R x + C, so the projection is the transpose
    local_i = (r_i.T @ (points - c_i).T).T
    local_k = (r_k.T @ (points - c_k).T).T
    ahead = (local_i[:, 2] > 1.0) & (local_k[:, 2] > 1.0)
    local_i, local_k = local_i[ahead], local_k[ahead]
    if len(local_i) < 50:
        return False, "the synthetic cloud did not fall in front of both cameras"

    normalised_i = local_i[:, :2] / local_i[:, 2:3]
    normalised_k = local_k[:, :2] / local_k[:, 2:3]

    matrix = support.essential((r_i, c_i), (r_k, c_k))
    homogeneous_i = np.hstack([normalised_i, np.ones((len(normalised_i), 1))])
    homogeneous_k = np.hstack([normalised_k, np.ones((len(normalised_k), 1))])
    residual = np.abs(np.einsum("ij,jk,ik->i", homogeneous_k, matrix, homogeneous_i))
    worst = float(residual.max())

    # the other convention, to show the test can fail
    wrong = support.essential((r_i.T, c_i), (r_k.T, c_k))
    wrong_residual = float(np.abs(np.einsum(
        "ij,jk,ik->i", homogeneous_k, wrong, homogeneous_i)).max())

    ok = worst < TOLERANCE
    return ok, (f"synthetic pair, {len(local_i)} points: "
                f"worst |x_k^T E x_i| = {worst:.3e} "
                f"(the transposed convention gives {wrong_residual:.3e})")


def stored_convention(dataset: str) -> tuple[bool, str]:
    """The third column of a stored rotation should be the optical axis."""
    path = REFERENCE_DIR / f"{dataset}_reference_poses.csv"
    worst_column, worst_row, worst_orthogonality = 0.0, 0.0, 0.0
    count = 0
    for row in csv.DictReader(path.open(encoding="utf-8")):
        rotation = np.array([[float(row[f"r{i}{j}"]) for j in range(3)]
                             for i in range(3)])
        axis = np.array([float(row["axis_e"]), float(row["axis_n"]),
                         float(row["axis_u"])])
        worst_column = max(worst_column,
                           float(np.abs(rotation[:, 2] - axis).max()))
        worst_row = max(worst_row, float(np.abs(rotation[2, :] - axis).max()))
        worst_orthogonality = max(worst_orthogonality, float(
            np.abs(rotation @ rotation.T - np.eye(3)).max()))
        count += 1
    ok = worst_column < 1e-9 and worst_orthogonality < 1e-9
    return ok, (f"{dataset}: {count} poses, orthogonality {worst_orthogonality:.2e}, "
                f"third column matches the optical axis to {worst_column:.2e} "
                f"(third row would be {worst_row:.2e})")


def against_the_archive(dataset: str) -> tuple[bool, str]:
    """The reference poses must satisfy their own epipolar constraint exactly."""
    poses = support.reference_poses(dataset)
    labels = list(poses)[:40]
    worst = 0.0
    rng = np.random.default_rng(11)
    for first, second in zip(labels[:-1], labels[1:]):
        (r_i, c_i), (r_k, c_k) = poses[first], poses[second]
        # a point on the ground below the pair, seen by construction
        ground = 0.5 * (c_i + c_k) + np.array([0.0, 0.0, -70.0])
        cloud = ground + rng.normal(scale=20.0, size=(60, 3))
        local_i = (r_i.T @ (cloud - c_i).T).T
        local_k = (r_k.T @ (cloud - c_k).T).T
        keep = (local_i[:, 2] > 1.0) & (local_k[:, 2] > 1.0)
        if keep.sum() < 5:
            continue
        a = local_i[keep][:, :2] / local_i[keep][:, 2:3]
        b = local_k[keep][:, :2] / local_k[keep][:, 2:3]
        matrix = support.essential(poses[first], poses[second])
        distance = support.sampson(a, b, matrix)
        worst = max(worst, float(np.sqrt(distance).max()))
    ok = worst < 1e-9
    return ok, (f"{dataset}: worst Sampson distance over consecutive reference "
                f"pairs, in normalised units, {worst:.2e}")


def main() -> int:
    results = [synthetic()]
    for dataset in DATASETS:
        results.append(stored_convention(dataset))
        results.append(against_the_archive(dataset))

    for ok, message in results:
        print(f"  {'pass' if ok else 'FAIL'}  {message}")
    failed = [message for ok, message in results if not ok]
    print(f"{len(results) - len(failed)} of {len(results)} checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
