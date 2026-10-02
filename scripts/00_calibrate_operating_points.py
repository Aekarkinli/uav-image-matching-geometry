#!/usr/bin/env python3
"""Calibrate runtime and GPU memory for the controlled matching experiment.

Measures, for each local-feature front end and each candidate working
resolution, the extraction time per image, the number of keypoints detected,
and the peak GPU memory. Also times LightGlue matching on a small number of
image pairs. The output is used to choose the operating points that the full
experiment can afford on the available hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from lightglue import ALIKED, DISK, SIFT, DoGHardNet, LightGlue, SuperPoint
from lightglue.utils import load_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
OUT_DIR = RESULTS / "calibration"

IMAGE_DIRS = {
    "D1": PROJECT_ROOT / "D1_154_building" / "D1_images",
    "D2": PROJECT_ROOT / "D2_111_nadir" / "D2_images",
}

RESOLUTIONS = [1024, 1600, 2048, 2560, 3200, 4096, None]  # None = native
KEYPOINT_BUDGET = 8192

EXTRACTORS = {
    "sift": (SIFT, {"max_num_keypoints": KEYPOINT_BUDGET, "rootsift": True}, True),
    "doghardnet": (DoGHardNet, {"max_num_keypoints": KEYPOINT_BUDGET}, True),
    "superpoint": (SuperPoint, {"max_num_keypoints": KEYPOINT_BUDGET}, True),
    "aliked": (ALIKED, {"max_num_keypoints": KEYPOINT_BUDGET}, False),
    "disk": (DISK, {"max_num_keypoints": KEYPOINT_BUDGET}, False),
}

LIGHTGLUE_FEATURES = {
    "sift": "sift",
    "doghardnet": "doghardnet",
    "superpoint": "superpoint",
    "aliked": "aliked",
    "disk": "disk",
}


def peak_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return float("nan")


def reset_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load(path: Path, grayscale: bool, device) -> torch.Tensor:
    """Load at native resolution. Resizing is delegated to the extractor so
    that keypoints are returned in native image coordinates."""
    image = load_image(path, resize=None)
    if grayscale and image.shape[0] == 3:
        image = 0.299 * image[0:1] + 0.587 * image[1:2] + 0.114 * image[2:3]
    return image.to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--dataset", default="D1")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = sorted(IMAGE_DIRS[args.dataset].glob("*.JPG"))[: args.images + 1]
    if len(image_paths) < 2:
        raise SystemExit(f"not enough images in {IMAGE_DIRS[args.dataset]}")

    rows = []
    for name, (cls, conf, grayscale) in EXTRACTORS.items():
        try:
            extractor = cls(**conf).eval().to(device)
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"[skip] {name}: cannot instantiate ({exc})")
            continue
        matcher = None
        for resize in RESOLUTIONS:
            label = "native" if resize is None else str(resize)
            reset_mem()
            times, counts, shapes = [], [], []
            status, detail = "OK", ""
            feats_cache = []
            try:
                for path in image_paths[: args.images]:
                    image = load(path, grayscale, device)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        feats = extractor.extract(image[None], resize=resize)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    times.append(time.perf_counter() - t0)
                    counts.append(int(feats["keypoints"].shape[1]))
                    working = (
                        image.shape[-2:]
                        if resize is None
                        else tuple(
                            int(round(s * resize / max(image.shape[-2:])))
                            for s in image.shape[-2:]
                        )
                    )
                    shapes.append(tuple(int(v) for v in working))
                    feats_cache.append({k: v for k, v in feats.items()})
                    del image
            except torch.cuda.OutOfMemoryError:
                status, detail = "OOM", "cuda out of memory during extraction"
            except Exception as exc:
                status, detail = "ERROR", f"{type(exc).__name__}: {exc}"

            extract_mem = peak_mem_mb()
            match_time, match_mem, match_count = "", "", ""
            if status == "OK" and len(feats_cache) >= 2:
                if matcher is None:
                    matcher = (
                        LightGlue(features=LIGHTGLUE_FEATURES[name]).eval().to(device)
                    )
                reset_mem()
                try:
                    mt = []
                    counts_m = []
                    for i in range(min(args.pairs, len(feats_cache) - 1)):
                        torch.cuda.synchronize() if device.type == "cuda" else None
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            out = matcher(
                                {"image0": feats_cache[i], "image1": feats_cache[i + 1]}
                            )
                        torch.cuda.synchronize() if device.type == "cuda" else None
                        mt.append(time.perf_counter() - t0)
                        counts_m.append(int(out["matches"][0].shape[0]))
                    match_time = f"{float(np.mean(mt)):.3f}"
                    match_count = f"{float(np.mean(counts_m)):.0f}"
                    match_mem = f"{peak_mem_mb():.0f}"
                except torch.cuda.OutOfMemoryError:
                    match_time, match_count, match_mem = "OOM", "", ""
                except Exception as exc:
                    match_time, match_count, match_mem = f"ERR:{type(exc).__name__}", "", ""

            del feats_cache
            reset_mem()

            row = {
                "front_end": name,
                "resize_max": label,
                "image_shape": shapes[0] if shapes else "",
                "status": status,
                "detail": detail,
                "extract_s_per_image": f"{float(np.mean(times)):.3f}" if times else "",
                "keypoints_median": f"{float(np.median(counts)):.0f}" if counts else "",
                "extract_peak_mem_mb": f"{extract_mem:.0f}",
                "match_s_per_pair": match_time,
                "match_peak_mem_mb": match_mem,
                "matches_mean": match_count,
            }
            rows.append(row)
            print(
                f"{name:11s} r={label:6s} {status:5s} "
                f"kp={row['keypoints_median']:>6s} "
                f"ext={row['extract_s_per_image']:>7s}s "
                f"mem={row['extract_peak_mem_mb']:>6s}MB "
                f"match={match_time:>7s}s {detail}"
            )
            if status == "OOM":
                break  # larger resolutions will also fail
        del extractor
        if matcher is not None:
            del matcher
        reset_mem()

    csv_path = OUT_DIR / f"operating_point_calibration_{args.dataset}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "total_vram_mb": (
            torch.cuda.get_device_properties(0).total_memory / 1024**2
            if torch.cuda.is_available()
            else None
        ),
        "torch": torch.__version__,
        "keypoint_budget": KEYPOINT_BUDGET,
        "dataset": args.dataset,
        "images_probed": args.images,
    }
    (OUT_DIR / f"calibration_meta_{args.dataset}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
