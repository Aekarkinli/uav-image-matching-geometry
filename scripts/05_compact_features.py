#!/usr/bin/env python3
"""Discard descriptors from feature stores that are no longer needed.

Descriptors are required only while a store is being matched, or while a
smaller keypoint budget is still to be derived from it. Once those steps are
complete only the keypoint coordinates, detector scores, scale and orientation
are needed, and those are two orders of magnitude smaller. Removing the
descriptors keeps the working set on disk manageable without losing anything
that the reconstruction or the analysis depends on.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
from collections import defaultdict
from pathlib import Path

import h5py

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FEATURES_DIR, PROJECT_ROOT  # noqa: E402
DATASET_LABELS = {"D1": "D1_154_building", "D2": "D2_111_nadir"}
KEEP = ("keypoints", "scores", "image_size", "scales", "oris")


def load_matrix_module():
    path = PROJECT_ROOT / "scripts" / "03_run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("experiment_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_store(path: Path):
    match = re.match(r"(.+)-(r\d+|native)-k(\d+)$", path.stem)
    if not match:
        return None
    return match.group(1), match.group(2), int(match.group(3))


def still_needed(matrix, label_to_key):
    """Feature stores that a pending matrix cell still requires descriptors for."""
    needed = set()
    groups = defaultdict(list)
    for item in matrix.build_matrix():
        res = "native" if item["resize"] == "native" else f"r{item['resize']}"
        label = DATASET_LABELS[item["dataset"]]
        key = (label, item["front_end"], res)
        groups[key].append(item)
    for key, items in groups.items():
        pending = [i for i in items if not matrix.is_complete(i)]
        if not pending:
            continue
        # the pending cells themselves, and every larger budget in the group,
        # since a smaller budget is derived by truncating one of those and that
        # needs the descriptors to still be present
        smallest_pending = min(int(i["budget"]) for i in pending)
        for item in items:
            if int(item["budget"]) >= smallest_pending:
                needed.add((*key, int(item["budget"])))
    return needed


def compact(path: Path) -> tuple[int, int]:
    before = path.stat().st_size
    temp = path.with_suffix(".compact.h5")
    with h5py.File(path, "r") as src, h5py.File(temp, "w") as dst:
        for name in src:
            group = src[name]
            out = dst.create_group(name)
            for key in KEEP:
                if key in group:
                    out.create_dataset(key, data=group[key][()])
    os.replace(temp, path)
    return before, path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="compact every store, including those a pending cell would use",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    matrix = load_matrix_module()
    needed = set() if args.all else still_needed(matrix, DATASET_LABELS)

    freed = 0
    compacted = 0
    for path in sorted(FEATURES_DIR.glob("*/*.h5")):
        parsed = parse_store(path)
        if parsed is None:
            continue
        label = path.parent.name
        key = (label, *parsed)
        with h5py.File(path, "r") as handle:
            first = next(iter(handle), None)
            if first is None or "descriptors" not in handle[first]:
                continue
        if key in needed:
            print(f"[keep] {label} {path.name} (still required)")
            continue
        if args.dry_run:
            print(f"[would compact] {label} {path.name} ({path.stat().st_size/1e6:.0f} MB)")
            continue
        before, after = compact(path)
        freed += before - after
        compacted += 1
        print(
            f"[compact] {label} {path.name}: "
            f"{before/1e6:.0f} MB -> {after/1e6:.0f} MB"
        )

    print(f"\ncompacted {compacted} stores, freed {freed/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
