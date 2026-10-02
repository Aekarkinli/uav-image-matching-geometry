#!/usr/bin/env python3
"""Controlled feature extraction and matching for one operating point.

One run covers a single combination of local feature, matcher, working image
resolution and keypoint budget. Keypoints are always returned in native image
coordinates and stored in single precision, the image pair list is identical
across runs, and geometric verification for the reported correspondence
statistics uses one estimator with one threshold for every configuration.

Outputs follow the layout expected by the reconstruction stage, so the same
features and matches feed both the correspondence analysis and the bundle
adjustment.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from lightglue import ALIKED, DISK, SIFT, DoGHardNet, LightGlue, SuperPoint
from lightglue.utils import load_image

import sys

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

# grayscale flag and the LightGlue weight name for each local feature
FRONT_ENDS = {
    "sift": (SIFT, {"rootsift": False}, True, "sift"),
    "rootsift": (SIFT, {"rootsift": True}, True, "sift"),
    "doghardnet": (DoGHardNet, {}, True, "doghardnet"),
    "superpoint": (SuperPoint, {}, True, "superpoint"),
    "aliked": (ALIKED, {}, False, "aliked"),
    "disk": (DISK, {}, False, "disk"),
}

# Geometric verification applied identically to every configuration.
VERIFIER = {
    "method": "USAC_MAGSAC fundamental matrix",
    "threshold_px": 2.0,
    "confidence": 0.9999,
    "max_iters": 10000,
    "min_matches": 8,
}
FAILED_PAIR_INLIER_THRESHOLD = 15
COVERAGE_GRID = 8

# For the learned detectors the keypoint budget is a strict limit on the final
# detections, so keeping the strongest n of a larger store reproduces a direct
# extraction exactly. The difference-of-Gaussians front ends pass the budget to the
# underlying implementation, which applies it before duplicate suppression, so a
# direct extraction returns fewer keypoints than the budget while a truncation
# returns exactly the budget. Those front ends are therefore always extracted.
BUDGET_APPLIED_BEFORE_FILTERING = {"sift", "rootsift", "doghardnet"}


def config_id(front_end: str, matcher: str, resize, budget: int) -> str:
    res = "native" if resize is None else f"r{resize}"
    return f"{front_end}-{matcher}-{res}-k{budget}"


def feature_id(front_end: str, resize, budget: int) -> str:
    res = "native" if resize is None else f"r{resize}"
    return f"{front_end}-{res}-k{budget}"


def build_extractor(front_end: str, budget: int, device):
    cls, conf, grayscale, lg_name = FRONT_ENDS[front_end]
    extractor = cls(max_num_keypoints=budget, **conf).eval().to(device)
    return extractor, grayscale, lg_name


def extract_all(
    front_end: str, resize, budget: int, image_paths, device, out_path: Path
) -> dict:
    """Extract features for every image and store them in native coordinates."""
    extractor, grayscale, _ = build_extractor(front_end, budget, device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timings, counts = [], []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with h5py.File(out_path, "w") as handle:
        for path in image_paths:
            image = load_image(path, resize=None)
            if grayscale and image.shape[0] == 3:
                image = 0.299 * image[0:1] + 0.587 * image[1:2] + 0.114 * image[2:3]
            image = image.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                feats = extractor.extract(image[None], resize=resize)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)

            keypoints = feats["keypoints"][0].detach().cpu().numpy().astype(np.float32)
            descriptors = feats["descriptors"][0].detach().cpu().numpy()
            scores_key = (
                "keypoint_scores" if "keypoint_scores" in feats else "scores"
            )
            scores = (
                feats[scores_key][0].detach().cpu().numpy().astype(np.float32)
                if scores_key in feats
                else np.zeros(len(keypoints), dtype=np.float32)
            )
            width, height = (
                int(feats["image_size"][0, 0].item()),
                int(feats["image_size"][0, 1].item()),
            )
            counts.append(len(keypoints))

            group = handle.create_group(path.name)
            group.create_dataset("keypoints", data=keypoints)  # float32, native px
            group.create_dataset(
                "descriptors", data=descriptors.astype(np.float16)
            )
            group.create_dataset("scores", data=scores)
            group.create_dataset(
                "image_size", data=np.array([width, height], dtype=np.int32)
            )
            # Detector scale and orientation, required by the LightGlue weights
            # trained on scale-invariant detectors. Stored exactly as produced,
            # which is the convention those weights were trained under.
            for extra in ("scales", "oris"):
                if extra in feats:
                    group.create_dataset(
                        extra,
                        data=feats[extra][0].detach().cpu().numpy().astype(np.float32),
                    )
            del image, feats

    peak = (
        torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "images": len(image_paths),
        "seconds_total": float(np.sum(timings)),
        "seconds_per_image_median": float(np.median(timings)),
        "keypoints_median": float(np.median(counts)),
        "keypoints_mean": float(np.mean(counts)),
        "keypoints_min": int(np.min(counts)),
        "keypoints_max": int(np.max(counts)),
        "peak_gpu_mb": float(peak),
    }


def derive_from_larger_budget(
    target: Path, front_end: str, resize, budget: int, label: str
) -> bool:
    """Build a smaller keypoint budget by truncating an existing larger one.

    The extractors retain the highest scoring detections, so truncating a
    larger store by score is identical to re-running extraction with the
    smaller budget, and it avoids repeating the costly dense forward pass.
    """
    res = "native" if resize is None else f"r{resize}"
    directory = FEATURES_DIR / label
    if not directory.exists():
        return False
    candidates = []
    for path in directory.glob(f"{front_end}-{res}-k*.h5"):
        try:
            other = int(path.stem.rsplit("-k", 1)[1])
        except (IndexError, ValueError):
            continue
        if other <= budget:
            continue
        # Only a store that still carries descriptors can seed a matching run.
        try:
            with h5py.File(path, "r") as probe:
                first = next(iter(probe), None)
                if first is None or "descriptors" not in probe[first]:
                    continue
        except OSError:
            continue
        candidates.append((other, path))
    if not candidates:
        return False
    _, source = min(candidates)

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.h5")
    with h5py.File(source, "r") as src, h5py.File(partial, "w") as dst:
        for name in src:
            group = src[name]
            scores = group["scores"][()]
            keypoints = group["keypoints"][()]
            descriptors = group["descriptors"][()]
            keep = None
            if len(keypoints) > budget:
                keep = np.argsort(-scores)[:budget]
                keep.sort()
                keypoints, descriptors, scores = (
                    keypoints[keep],
                    descriptors[keep],
                    scores[keep],
                )
            out = dst.create_group(name)
            out.create_dataset("keypoints", data=keypoints)
            out.create_dataset("descriptors", data=descriptors)
            out.create_dataset("scores", data=scores)
            out.create_dataset("image_size", data=group["image_size"][()])
            for extra in ("scales", "oris"):
                if extra in group:
                    values = group[extra][()]
                    if keep is not None:
                        values = values[keep]
                    out.create_dataset(extra, data=values)
    partial.replace(target)
    return True


def usable_store(path: Path) -> bool:
    """A store left behind by an interrupted write holds no images."""
    try:
        with h5py.File(path, "r") as handle:
            return next(iter(handle), None) is not None
    except OSError:
        return False


def load_features(path: Path):
    store = {}
    with h5py.File(path, "r") as handle:
        for name in handle:
            group = handle[name]
            entry = {
                "keypoints": group["keypoints"][()],
                "descriptors": group["descriptors"][()],
                "image_size": group["image_size"][()],
            }
            for extra in ("scales", "oris"):
                if extra in group:
                    entry[extra] = group[extra][()]
            store[name] = entry
    return store


def to_lightglue(entry, device):
    data = {
        "keypoints": torch.from_numpy(entry["keypoints"]).float()[None].to(device),
        "descriptors": torch.from_numpy(entry["descriptors"]).float()[None].to(device),
        "image_size": torch.from_numpy(entry["image_size"]).float()[None].to(device),
    }
    for extra in ("scales", "oris"):
        if extra in entry:
            data[extra] = torch.from_numpy(entry[extra]).float()[None].to(device)
    return data


def nn_ratio_match(desc0: torch.Tensor, desc1: torch.Tensor, ratio: float = 0.8):
    """Mutual nearest neighbour matching with Lowe's ratio test."""
    d0 = torch.nn.functional.normalize(desc0, dim=-1)
    d1 = torch.nn.functional.normalize(desc1, dim=-1)
    sim = d0 @ d1.t()
    if sim.shape[1] < 2 or sim.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)
    top0 = sim.topk(2, dim=1)
    top1 = sim.topk(2, dim=0)
    nn0 = top0.indices[:, 0]
    nn1 = top1.indices[0, :]
    idx = torch.arange(len(nn0), device=sim.device)
    mutual = nn1[nn0] == idx
    # cosine similarity to squared euclidean distance for the ratio test
    dist0 = (2.0 - 2.0 * top0.values.clamp(-1, 1)).clamp(min=1e-12).sqrt()
    ratio_ok = dist0[:, 0] <= ratio * dist0[:, 1]
    keep = mutual & ratio_ok
    matches = torch.stack([idx[keep], nn0[keep]], dim=1)
    scores = top0.values[keep, 0]
    return matches.cpu().numpy(), scores.cpu().numpy().astype(np.float32)


def verify(points0: np.ndarray, points1: np.ndarray):
    """Common geometric verification for the reported statistics."""
    if len(points0) < VERIFIER["min_matches"]:
        return np.zeros(len(points0), dtype=bool), "too_few_matches"
    try:
        matrix, mask = cv2.findFundamentalMat(
            points0.astype(np.float64),
            points1.astype(np.float64),
            cv2.USAC_MAGSAC,
            VERIFIER["threshold_px"],
            VERIFIER["confidence"],
            VERIFIER["max_iters"],
        )
    except cv2.error:
        # The estimator raises on degenerate configurations, for example a
        # near planar scene or a pair separated by an almost pure rotation,
        # where no epipolar geometry can be recovered from the putative set.
        return np.zeros(len(points0), dtype=bool), "degenerate"
    if matrix is None or mask is None:
        return np.zeros(len(points0), dtype=bool), "no_model"
    return mask.ravel().astype(bool), "ok"


def coverage(points: np.ndarray, width: int, height: int) -> float:
    """Fraction of an 8 by 8 image grid that contains at least one point."""
    if len(points) == 0:
        return 0.0
    col = np.clip((points[:, 0] / width * COVERAGE_GRID).astype(int), 0, COVERAGE_GRID - 1)
    row = np.clip((points[:, 1] / height * COVERAGE_GRID).astype(int), 0, COVERAGE_GRID - 1)
    return float(len(set(zip(row.tolist(), col.tolist()))) / COVERAGE_GRID**2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--front-end", required=True, choices=sorted(FRONT_ENDS))
    parser.add_argument("--matcher", required=True, choices=["lightglue", "nn_ratio"])
    parser.add_argument("--resize", default="native")
    parser.add_argument("--budget", type=int, default=8192)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    resize = None if str(args.resize).lower() == "native" else int(args.resize)
    spec = DATASETS[args.dataset]
    label = spec["label"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cid = config_id(args.front_end, args.matcher, resize, args.budget)
    fid = feature_id(args.front_end, resize, args.budget)
    ensure_workspace()
    features_path = FEATURES_DIR / label / f"{fid}.h5"
    matches_path = MATCHES_DIR / label / f"{cid}.h5"
    metrics_path = PAIRWISE_DIR / label / f"{cid}.csv"
    manifest_path = MANIFESTS_DIR / label / f"{cid}.json"
    for path in (features_path, matches_path, metrics_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() and not args.force:
        print(f"[skip] {label} {cid} already complete")
        return 0

    image_paths = sorted(spec["images"].glob("*.JPG"))
    pairs = [
        line.split() for line in spec["pairs"].read_text().splitlines() if line.strip()
    ]

    if features_path.exists() and not usable_store(features_path):
        print(f"[features] discarding unusable {features_path.name}")
        features_path.unlink()

    if features_path.exists() and not args.force:
        extraction = {"reused": True}
        print(f"[features] reusing {features_path.name}")
    elif (
        not args.force
        and args.front_end not in BUDGET_APPLIED_BEFORE_FILTERING
        and derive_from_larger_budget(
            features_path, args.front_end, resize, args.budget, label
        )
    ):
        extraction = {"reused": True, "derived_by_score_truncation": True}
        print(f"[features] derived {features_path.name} from a larger budget")
    else:
        print(f"[features] {label} {fid} over {len(image_paths)} images")
        extraction = extract_all(
            args.front_end, resize, args.budget, image_paths, device, features_path
        )
        extraction["reused"] = False
        print(
            f"[features] median {extraction['keypoints_median']:.0f} keypoints, "
            f"{extraction['seconds_per_image_median']:.2f} s/image, "
            f"peak {extraction['peak_gpu_mb']:.0f} MB"
        )

    store = load_features(features_path)
    matcher = None
    if args.matcher == "lightglue":
        lg_name = FRONT_ENDS[args.front_end][3]
        matcher = LightGlue(features=lg_name).eval().to(device)

    rows = []
    match_times = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_all = time.perf_counter()

    with h5py.File(matches_path, "w") as handle:
        for index, (name0, name1) in enumerate(pairs, start=1):
            entry0, entry1 = store[name0], store[name1]
            feats0 = to_lightglue(entry0, device)
            feats1 = to_lightglue(entry1, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            if args.matcher == "lightglue":
                with torch.no_grad():
                    out = matcher({"image0": feats0, "image1": feats1})
                matches = out["matches"][0].detach().cpu().numpy()
                scores = out["scores"][0].detach().cpu().numpy().astype(np.float32)
            else:
                matches, scores = nn_ratio_match(
                    feats0["descriptors"][0], feats1["descriptors"][0], args.ratio
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            match_times.append(elapsed)

            kp0, kp1 = entry0["keypoints"], entry1["keypoints"]
            width0, height0 = int(entry0["image_size"][0]), int(entry0["image_size"][1])
            width1, height1 = int(entry1["image_size"][0]), int(entry1["image_size"][1])

            if len(matches):
                points0 = kp0[matches[:, 0]]
                points1 = kp1[matches[:, 1]]
                inlier_mask, status = verify(points0, points1)
            else:
                points0 = points1 = np.zeros((0, 2), dtype=np.float32)
                inlier_mask, status = np.zeros(0, dtype=bool), "no_matches"

            inliers = int(inlier_mask.sum())
            raw = int(len(matches))
            # hloc-compatible match record, restricted to verified inliers
            kept = matches[inlier_mask] if raw else matches
            kept_scores = scores[inlier_mask] if raw else scores
            matches0 = np.full(len(kp0), -1, dtype=np.int32)
            scores0 = np.zeros(len(kp0), dtype=np.float32)
            if len(kept):
                matches0[kept[:, 0]] = kept[:, 1].astype(np.int32)
                scores0[kept[:, 0]] = kept_scores
            group = handle.create_group(f"{name0}/{name1}")
            group.create_dataset("matches0", data=matches0)
            group.create_dataset("matching_scores0", data=scores0)

            rows.append(
                {
                    "dataset": label,
                    "config": cid,
                    "front_end": args.front_end,
                    "matcher": args.matcher,
                    "working_resolution": "native" if resize is None else resize,
                    "keypoint_budget": args.budget,
                    "pair_index": index,
                    "image1": name0,
                    "image2": name1,
                    "keypoints_image1": len(kp0),
                    "keypoints_image2": len(kp1),
                    "raw_matches": raw,
                    "verified_inliers": inliers,
                    "inlier_ratio": (inliers / raw) if raw else 0.0,
                    "inliers_per_1000_keypoints": (
                        1000.0 * inliers / min(len(kp0), len(kp1))
                        if min(len(kp0), len(kp1))
                        else 0.0
                    ),
                    "coverage_image1": coverage(points0[inlier_mask], width0, height0)
                    if raw
                    else 0.0,
                    "coverage_image2": coverage(points1[inlier_mask], width1, height1)
                    if raw
                    else 0.0,
                    "mean_match_score": float(np.mean(scores)) if len(scores) else 0.0,
                    "failed_pair": inliers < FAILED_PAIR_INLIER_THRESHOLD,
                    "verification_status": status,
                    "seconds": elapsed,
                }
            )
            if index % 250 == 0:
                print(
                    f"[match] {index}/{len(pairs)} "
                    f"median inliers {np.median([r['verified_inliers'] for r in rows]):.0f}"
                )

    total_seconds = time.perf_counter() - start_all
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    inliers = np.array([r["verified_inliers"] for r in rows], dtype=float)
    statuses = {}
    for row in rows:
        statuses[row["verification_status"]] = statuses.get(row["verification_status"], 0) + 1
    manifest = {
        "config": cid,
        "dataset": label,
        "front_end": args.front_end,
        "matcher": args.matcher,
        "working_resolution": "native" if resize is None else resize,
        "keypoint_budget": args.budget,
        "ratio_threshold": args.ratio if args.matcher == "nn_ratio" else None,
        "verifier": VERIFIER,
        "failed_pair_inlier_threshold": FAILED_PAIR_INLIER_THRESHOLD,
        "keypoint_storage_dtype": "float32",
        "descriptor_storage_dtype": "float16",
        "pairs": len(pairs),
        "extraction": extraction,
        "matching_seconds_total": total_seconds,
        "matching_seconds_per_pair_median": float(np.median(match_times)),
        "matching_peak_gpu_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else None
        ),
        "median_verified_inliers": float(np.median(inliers)),
        "mean_verified_inliers": float(np.mean(inliers)),
        "failed_pair_fraction": float(np.mean(inliers < FAILED_PAIR_INLIER_THRESHOLD)),
        "verification_status_counts": statuses,
        "features_path": str(features_path),
        "matches_path": str(matches_path),
        "metrics_path": str(metrics_path),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"[done] {cid}: median inliers {manifest['median_verified_inliers']:.0f}, "
        f"failed pairs {100 * manifest['failed_pair_fraction']:.1f}%, "
        f"{manifest['matching_seconds_per_pair_median']:.3f} s/pair, "
        f"{total_seconds / 60:.1f} min total"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
