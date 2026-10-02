#!/usr/bin/env python3
"""Measure what each front end costs in graphics memory and time.

Records the peak allocation and the time per image for each front end and
extraction resolution. The full resolution sweep, and a repeat in half
precision, are measured for DISK, whose native-resolution run did not fit the
device.

A configuration that exhausts the device is recorded as such rather than
omitted, so the boundary of the design is part of the result. Each measurement
is written as it is taken, so an interrupted run still leaves what it measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT  # noqa: E402

BUDGET = 8192
REPEATS = 2
OUTPUT = PROJECT_ROOT / "results" / "tables" / "memory_feasibility.json"

# The full sweep is measured for the front end whose feasibility is in
# question; the others are measured at the reference and native resolutions for
# context.
FULL_SWEEP = {"disk"}
CONTEXT = [2048, None]
ALL_RESOLUTIONS = [1024, 1600, 2048, 3200, None]
DETECTORS = ["rootsift", "doghardnet", "superpoint", "aliked", "disk"]
DENSE_RESOLUTIONS = [1024, 1600, 2048]


def probe_detector(front_end: str, image_path: Path, resize, precision: str, device):
    from lightglue import ALIKED, DISK, SIFT, DoGHardNet, SuperPoint
    from lightglue.utils import load_image

    classes = {
        "rootsift": (SIFT, {"rootsift": True}, True),
        "doghardnet": (DoGHardNet, {}, True),
        "superpoint": (SuperPoint, {}, True),
        "aliked": (ALIKED, {}, False),
        "disk": (DISK, {}, False),
    }
    cls, conf, grayscale = classes[front_end]
    extractor = cls(max_num_keypoints=BUDGET, **conf).eval().to(device)
    if precision == "fp16":
        extractor = extractor.half()

    image = load_image(image_path, resize=None)
    if grayscale and image.shape[0] == 3:
        image = 0.299 * image[0:1] + 0.587 * image[1:2] + 0.114 * image[2:3]
    image = image.to(device)
    if precision == "fp16":
        image = image.half()

    torch.cuda.reset_peak_memory_stats()
    seconds, keypoints = [], None
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = extractor.extract(image[None], resize=resize)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
        keypoints = int(output["keypoints"].shape[1])
    peak = torch.cuda.max_memory_allocated() / 2**30
    del extractor, image
    torch.cuda.empty_cache()
    return {"peak_gib": peak, "seconds": min(seconds), "keypoints": keypoints}


def probe_dense(image_a: Path, image_b: Path, resize, device):
    """The detector-free matcher, which holds a correlation volume per pair."""
    import kornia
    import kornia.feature as KF
    from lightglue.utils import load_image

    matcher = KF.LoFTR(pretrained="outdoor").eval().to(device)
    tensors = []
    for path in (image_a, image_b):
        image = load_image(path, resize=resize)
        if image.shape[0] == 3:
            image = kornia.color.rgb_to_grayscale(image[None])[0]
        tensors.append(image[None].to(device))

    torch.cuda.reset_peak_memory_stats()
    seconds, matches = [], None
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = matcher({"image0": tensors[0], "image1": tensors[1]})
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
        matches = int(output["keypoints0"].shape[0])
    peak = torch.cuda.max_memory_allocated() / 2**30
    del matcher, tensors
    torch.cuda.empty_cache()
    return {"peak_gib": peak, "seconds": min(seconds), "matches": matches}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="D1_154_building")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no device available", flush=True)
        return 1
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    images = sorted((PROJECT_ROOT / args.dataset).glob("*images/*.JPG"))
    if len(images) < 2:
        print(f"no images under {args.dataset}", flush=True)
        return 1

    record = {
        "device": properties.name,
        "total_memory_gib": properties.total_memory / 2**30,
        "image": images[0].name,
        "keypoint_budget": BUDGET,
        "repeats": REPEATS,
        "detectors": [],
        "detector_free": [],
    }

    def save():
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(record, indent=2), encoding="utf-8")

    for front_end in DETECTORS:
        resolutions = ALL_RESOLUTIONS if front_end in FULL_SWEEP else CONTEXT
        precisions = ("fp32", "fp16") if front_end in FULL_SWEEP else ("fp32",)
        for resize in resolutions:
            for precision in precisions:
                try:
                    result, status = probe_detector(front_end, images[0], resize,
                                                    precision, device), "ok"
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    result, status = {}, "out of memory"
                except Exception as error:
                    torch.cuda.empty_cache()
                    result, status = {}, f"failed: {type(error).__name__}"
                record["detectors"].append({
                    "front_end": front_end, "resolution": resize or "native",
                    "precision": precision, "status": status, **result,
                })
                save()
                detail = (f"{result['peak_gib']:5.2f} GiB {result['seconds']:6.2f} s "
                          f"{result['keypoints']} keypoints" if result else "")
                print(f"  {front_end:11s} {str(resize or 'native'):>6} {precision}  "
                      f"{status:14s} {detail}", flush=True)

    for resize in DENSE_RESOLUTIONS:
        try:
            result, status = probe_dense(images[0], images[1], resize, device), "ok"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result, status = {}, "out of memory"
        except Exception as error:
            torch.cuda.empty_cache()
            result, status = {}, f"failed: {type(error).__name__}"
        record["detector_free"].append({
            "front_end": "loftr", "resolution": resize, "precision": "fp32",
            "status": status, **result,
        })
        save()
        detail = (f"{result['peak_gib']:5.2f} GiB {result['seconds']:6.2f} s "
                  f"{result['matches']} matches" if result else "")
        print(f"  loftr       {resize:>6} fp32  {status:14s} {detail}", flush=True)

    print(f"\nwritten to {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
