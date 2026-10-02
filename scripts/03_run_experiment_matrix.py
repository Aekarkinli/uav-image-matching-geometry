#!/usr/bin/env python3
"""Drive the full matching experiment matrix.

Four groups of runs are scheduled. The first varies the extraction
resolution at a fixed keypoint limit, the second adds the nearest-neighbour
matcher and the hybrid difference-of-Gaussians descriptor at two resolutions,
the third varies the keypoint limit at a fixed resolution, and the fourth
records which front ends can be run at native resolution on the available
hardware. Each cell is executed as a separate
process so that a failure in one configuration does not stop the others, and
completed cells are skipped on re-entry.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
WORKER = PROJECT_ROOT / "scripts" / "02_extract_and_match.py"
STATUS_PATH = RESULTS / "experiment_matrix_status.json"

DATASETS = ["D1", "D2"]
DATASET_LABELS = {"D1": "D1_154_building", "D2": "D2_111_nadir"}

RESOLUTIONS = [1024, 1600, 2048, 3200]
REFERENCE_RESOLUTION = 2048
REFERENCE_BUDGET = 8192
BUDGETS = [16384, 8192, 4096, 2048]

LEARNED = ["superpoint", "aliked", "disk"]
CLASSICAL = ["rootsift"]
HYBRID = ["doghardnet"]

# front ends that complete at native resolution within the GPU memory budget
NATIVE_CAPABLE = ["rootsift", "superpoint", "aliked"]


def cell(dataset, front_end, matcher, resize, budget):
    return {
        "dataset": dataset,
        "front_end": front_end,
        "matcher": matcher,
        "resize": resize,
        "budget": budget,
    }


def build_matrix():
    cells = []
    for dataset in DATASETS:
        # 1. resolution sweep at the reference keypoint budget
        for resize in RESOLUTIONS:
            for front_end in CLASSICAL + LEARNED:
                cells.append(
                    cell(dataset, front_end, "lightglue", resize, REFERENCE_BUDGET)
                )
        # 2. classical matcher baseline and hybrid descriptor at two resolutions
        for resize in (1024, REFERENCE_RESOLUTION):
            cells.append(cell(dataset, "rootsift", "nn_ratio", resize, REFERENCE_BUDGET))
            for front_end in HYBRID:
                cells.append(
                    cell(dataset, front_end, "lightglue", resize, REFERENCE_BUDGET)
                )
        # 3. keypoint budget sweep at the reference resolution
        for budget in BUDGETS:
            if budget == REFERENCE_BUDGET:
                continue
            for front_end in CLASSICAL + LEARNED:
                cells.append(
                    cell(dataset, front_end, "lightglue", REFERENCE_RESOLUTION, budget)
                )
        # 4. native resolution where the hardware allows it
        for front_end in NATIVE_CAPABLE:
            cells.append(
                cell(dataset, front_end, "lightglue", "native", REFERENCE_BUDGET)
            )
    return cells


# Matching cost grows sharply once a dense detector actually fills a large
# keypoint budget, because the matcher then runs its full depth on the hard
# pairs. Those cells are scheduled last so that the rest of the matrix is
# available early.
EXPENSIVE_FRONT_ENDS = {"aliked", "disk"}
EXPENSIVE_BUDGET = 16384


def sort_key(item):
    """Cheap cells first; within one feature store, the largest budget first so
    that smaller budgets are derived by score truncation instead of re-run."""
    resize = item["resize"]
    resize_rank = 10**6 if resize == "native" else int(resize)
    front_rank = {"rootsift": 0, "doghardnet": 1, "superpoint": 2, "aliked": 3, "disk": 4}
    expensive = (
        item["front_end"] in EXPENSIVE_FRONT_ENDS
        and int(item["budget"]) >= EXPENSIVE_BUDGET
    )
    return (
        1 if expensive else 0,
        resize_rank,
        front_rank.get(item["front_end"], 9),
        item["dataset"],
        -int(item["budget"]),
        item["matcher"],
    )


def config_id(item):
    res = "native" if item["resize"] == "native" else f"r{item['resize']}"
    return f"{item['front_end']}-{item['matcher']}-{res}-k{item['budget']}"


def is_complete(item) -> bool:
    path = (
        RESULTS
        / "manifests"
        / DATASET_LABELS[item["dataset"]]
        / f"{config_id(item)}.json"
    )
    return path.exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-dataset", choices=DATASETS)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cells = sorted(build_matrix(), key=sort_key)
    if args.only_dataset:
        cells = [c for c in cells if c["dataset"] == args.only_dataset]
    pending = [c for c in cells if not is_complete(c)]
    if args.limit:
        pending = pending[: args.limit]

    print(f"matrix: {len(cells)} cells, {len(pending)} pending")
    if args.dry_run:
        for item in pending:
            print(f"  {item['dataset']} {config_id(item)}")
        return 0

    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "cells": []}
    for index, item in enumerate(pending, start=1):
        command = [
            sys.executable,
            str(WORKER),
            "--dataset",
            item["dataset"],
            "--front-end",
            item["front_end"],
            "--matcher",
            item["matcher"],
            "--resize",
            str(item["resize"]),
            "--budget",
            str(item["budget"]),
        ]
        print(f"\n[{index}/{len(pending)}] {item['dataset']} {config_id(item)}", flush=True)
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        elapsed = time.perf_counter() - start
        status["cells"].append(
            {
                **item,
                "config": config_id(item),
                "returncode": completed.returncode,
                "seconds": elapsed,
            }
        )
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if completed.returncode != 0:
            print(f"  [warn] returncode {completed.returncode}, continuing", flush=True)

        # Discard descriptors that no later cell needs, so that the working set
        # on disk stays bounded as the matrix progresses.
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "05_compact_features.py")],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
        )

    failed = [c for c in status["cells"] if c["returncode"] != 0]
    print(f"\ncompleted {len(status['cells']) - len(failed)}/{len(status['cells'])} cells")
    if failed:
        print("failed cells:")
        for item in failed:
            print(f"  {item['dataset']} {item['config']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
