#!/usr/bin/env python3
"""Turn the per-pair support record into the quantities the article reports.

Three things come out of it. The acceptance rule of the experiment is scored
against the reference geometry, so that a pair the pipeline keeps can be
separated into one that carries the relative orientation it claims and one that
does not. The proportions are given an interval that respects the fact that
candidate pairs share images, by resampling images rather than pairs. And the
dependence of recovery on viewing-direction difference is fitted continuously,
so the angle at which a method loses half of its pairs is estimated rather than
read off a class boundary.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT  # noqa: E402

ANALYSIS = PROJECT_ROOT / "results" / "analysis"
DATASETS = ["D1_154_building", "D2_111_nadir"]
CONVERGENCE_EDGES = [0.0, 2.0, 10.0, 25.0, 35.0, 50.0, np.inf]
BASELINE_EDGES = [0.0, 20.0, 40.0, 60.0, 80.0, 120.0, np.inf]
DRAWS = 4000
SEED = 20260914
HALF_DRAWS = 600          # the logistic is refitted on every draw, so fewer

LABEL = {
    "rootsift-lightglue": "RootSIFT",
    "rootsift-nn_ratio": "RootSIFT, nearest neighbour",
    "doghardnet-lightglue": "DoG + HardNet",
    "superpoint-lightglue": "SuperPoint",
    "aliked-lightglue": "ALIKED",
    "disk-lightglue": "DISK",
    "loftr-dense": "LoFTR",
}


def method_of(config: str) -> str:
    return "-".join(config.split("-")[:2])


def load(tag: str, dataset: str) -> list[dict]:
    path = ANALYSIS / f"pair_pose_support_{tag}_{dataset}.csv"
    rows = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        row["accepted"] = row["accepted"] == "True"
        row["supportable"] = row["supportable"] == "True"
        for key in ("baseline_m", "convergence_angle_deg"):
            row[key] = float(row[key])
        for key in ("putative", "verified_inliers", "reference_consistent"):
            row[key] = int(row[key])
        rows.append(row)
    return rows


def clustered_interval(rows: list[dict], flag: np.ndarray, rng) -> tuple[float, float]:
    """Resample images and keep the pairs among them, so a pair is not a trial.

    Every candidate pair shares each of its images with many others, so an
    interval built by resampling pairs treats dependent observations as
    independent. Resampling images instead, and weighting a pair by the product
    of the multiplicities of its two images, respects that structure.
    """
    if not rows:
        return float("nan"), float("nan")
    images = sorted({row["image1"] for row in rows} | {row["image2"] for row in rows})
    index = {name: position for position, name in enumerate(images)}
    first = np.array([index[row["image1"]] for row in rows])
    second = np.array([index[row["image2"]] for row in rows])
    flag = flag.astype(float)

    draws = np.empty(DRAWS)
    for draw in range(DRAWS):
        multiplicity = np.bincount(
            rng.integers(0, len(images), len(images)), minlength=len(images))
        weight = multiplicity[first] * multiplicity[second]
        total = weight.sum()
        draws[draw] = (weight @ flag) / total if total else np.nan
    draws = draws[np.isfinite(draws)]
    if draws.size == 0 or draws.max() - draws.min() < 1e-12:
        # every draw agrees, which happens when the class is entirely recovered
        # or entirely lost; the resampling carries no width and the score
        # interval is quoted instead
        successes = int(round(flag.sum()))
        centre, low, high = wilson(successes, len(flag))
        return low, high
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def wilson(successes: int, total: int, z: float = 1.96):
    """Score interval, used only where the image resampling degenerates."""
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denominator
    spread = z * ((p * (1 - p) / total
                   + z ** 2 / (4 * total ** 2)) ** 0.5) / denominator
    return p, max(centre - spread, 0.0), min(centre + spread, 1.0)


def half_angle_interval(rows, recovered: np.ndarray, rng):
    """Image-resampled interval for the half-loss angle, and the observed span.

    The logistic is refitted on every draw. A draw whose crossing falls outside
    the angles the block contains is recorded as unreached, and the share of
    such draws is reported beside the interval.
    """
    angles = np.array([row["convergence_angle_deg"] for row in rows])
    ceiling = float(angles.max())
    images = sorted({row["image1"] for row in rows} | {row["image2"] for row in rows})
    index = {name: position for position, name in enumerate(images)}
    first = np.array([index[row["image1"]] for row in rows])
    second = np.array([index[row["image2"]] for row in rows])

    crossings, unreached = [], 0
    for _ in range(HALF_DRAWS):
        multiplicity = np.bincount(
            rng.integers(0, len(images), len(images)), minlength=len(images))
        weight = (multiplicity[first] * multiplicity[second]).astype(float)
        if weight.sum() <= 0:
            continue
        value = logistic_half(angles, recovered, weight)
        if not np.isfinite(value) or value > ceiling or value < 0.0:
            unreached += 1
            continue
        crossings.append(value)
    if len(crossings) < 0.5 * HALF_DRAWS:
        return {"reached": False, "ceiling_deg": ceiling,
                "unreached_share": unreached / max(HALF_DRAWS, 1)}
    return {"reached": True, "ceiling_deg": ceiling,
            "low_deg": float(np.percentile(crossings, 2.5)),
            "high_deg": float(np.percentile(crossings, 97.5)),
            "unreached_share": unreached / max(HALF_DRAWS, 1)}


def logistic_half(angles: np.ndarray, recovered: np.ndarray,
                  weight: np.ndarray | None = None) -> float:
    """Angle at which a fitted logistic crosses one half, by Newton descent."""
    if recovered.all() or not recovered.any():
        return float("nan")
    x = (angles - angles.mean()) / max(angles.std(), 1e-9)
    beta = np.zeros(2)
    design = np.stack([np.ones_like(x), x], axis=1)
    prior = np.ones_like(x) if weight is None else weight
    for _ in range(200):
        eta = design @ beta
        probability = 1.0 / (1.0 + np.exp(-eta))
        gradient = design.T @ (prior * (recovered - probability))
        weights = prior * probability * (1.0 - probability) + 1e-9
        hessian = design.T @ (design * weights[:, None])
        step = np.linalg.solve(hessian + 1e-8 * np.eye(2), gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    if abs(beta[1]) < 1e-9:
        return float("nan")
    return float(-beta[0] / beta[1] * max(angles.std(), 1e-9) + angles.mean())


def strata(rows: list[dict], field: str, edges: list[float]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        position = int(np.digitize(row[field], edges[1:-1], right=False))
        grouped[position] = grouped[position] + [row]
    return grouped


def summarise(tag: str) -> dict:
    rng = np.random.default_rng(SEED)
    report = {}
    for dataset in DATASETS:
        rows = load(tag, dataset)
        by_config = defaultdict(list)
        for row in rows:
            by_config[row["config"]].append(row)

        block = {}
        for config, entries in by_config.items():
            accepted = np.array([row["accepted"] for row in entries])
            supportable = np.array([row["supportable"] for row in entries])
            both = int(np.count_nonzero(accepted & supportable))
            precision = both / max(int(accepted.sum()), 1)
            recall = both / max(int(supportable.sum()), 1)
            low, high = clustered_interval(entries, ~accepted, rng)

            angles = np.array([row["convergence_angle_deg"] for row in entries])
            ceiling = float(angles.max())
            half_accepted = logistic_half(angles, accepted.astype(float))
            half_supported = logistic_half(angles, supportable.astype(float))
            half_interval = half_angle_interval(entries, accepted.astype(float), rng)
            reached = (np.isfinite(half_accepted) and half_accepted <= ceiling
                       and half_interval["reached"])

            classes = {}
            for name, field, edges in (
                    ("convergence", "convergence_angle_deg", CONVERGENCE_EDGES),
                    ("baseline", "baseline_m", BASELINE_EDGES)):
                classes[name] = []
                grouped = strata(entries, field, edges)
                for position in range(len(edges) - 1):
                    members = grouped.get(position, [])
                    if not members:
                        classes[name].append({"count": 0})
                        continue
                    failed = sum(0 if row["accepted"] else 1 for row in members)
                    unsupported = sum(1 for row in members
                                      if row["accepted"] and not row["supportable"])
                    interval = clustered_interval(
                        members,
                        np.array([not row["accepted"] for row in members]), rng)
                    classes[name].append({
                        "count": len(members),
                        "failed_pct": 100.0 * failed / len(members),
                        "failed_low": 100.0 * interval[0],
                        "failed_high": 100.0 * interval[1],
                        "unsupported_of_accepted_pct":
                            100.0 * unsupported / max(len(members) - failed, 1),
                        "supportable_pct": 100.0 * float(
                            np.mean([row["supportable"] for row in members])),
                    })

            block[config] = {
                "method": LABEL[method_of(config)],
                "pairs": len(entries),
                "accepted_pct": 100.0 * float(accepted.mean()),
                "failed_pct": 100.0 * float(1.0 - accepted.mean()),
                "failed_low": 100.0 * low,
                "failed_high": 100.0 * high,
                "supportable_pct": 100.0 * float(supportable.mean()),
                "precision": precision,
                "recall": recall,
                "half_angle_accepted_deg": half_accepted if reached else None,
                "half_angle_supported_deg": half_supported,
                "half_angle_reached": bool(reached),
                "half_angle_ceiling_deg": ceiling,
                "half_angle_interval": half_interval,
                "half_angle_unconstrained_fit_deg": half_accepted,
                "classes": classes,
            }
        report[dataset] = block
    return report


def main() -> int:
    for tag in ("operating", "common1024"):
        report = summarise(tag)
        path = ANALYSIS / f"pair_support_summary_{tag}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {path.name}")
        for dataset, block in report.items():
            print(f"  {dataset}")
            for config, entry in block.items():
                print(f"    {entry['method']:28s} failed {entry['failed_pct']:5.1f} "
                      f"[{entry['failed_low']:5.1f},{entry['failed_high']:5.1f}]  "
                      f"supportable {entry['supportable_pct']:5.1f}  "
                      f"precision {entry['precision']:.3f}  "
                      + (f"half {entry['half_angle_accepted_deg']:5.1f} "
                         f"[{entry['half_angle_interval']['low_deg']:.1f},"
                         f"{entry['half_angle_interval']['high_deg']:.1f}] deg"
                         if entry["half_angle_reached"]
                         else f"half not reached within "
                              f"{entry['half_angle_ceiling_deg']:.0f} deg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
