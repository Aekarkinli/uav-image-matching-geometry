#!/usr/bin/env python3
"""Count the keypoints actually stored in every feature store.

The number of keypoints a front end delivers is not the number requested. Some
detectors saturate well below the budget, some apply the budget before their own
duplicate filtering, and the detector-free matcher has no budget at all. The
census reads the stores themselves rather than the extraction records, because a
store derived by truncating a larger budget carries no extraction record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FEATURES_DIR, PROJECT_ROOT  # noqa: E402

OUTPUT = PROJECT_ROOT / "results" / "tables" / "keypoint_census.json"


def main() -> int:
    import h5py

    census = {}
    for store in sorted(FEATURES_DIR.glob("*/*.h5")):
        dataset = store.parent.name
        stem = store.stem
        try:
            with h5py.File(store, "r") as handle:
                counts = [handle[name]["keypoints"].shape[0] for name in handle]
        except Exception as error:  # a store still being written, or truncated
            print(f"  [skip] {dataset}/{stem}: {error}")
            continue
        if not counts:
            print(f"  [skip] {dataset}/{stem}: empty")
            continue
        counts = np.array(counts, dtype=float)
        census[f"{dataset}/{stem}"] = {
            "dataset": dataset,
            "store": stem,
            "images": int(counts.size),
            "median": float(np.median(counts)),
            "mean": float(counts.mean()),
            "min": float(counts.min()),
            "max": float(counts.max()),
        }
        print(f"  {dataset}/{stem}: median {np.median(counts):.0f} over "
              f"{counts.size} images")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(census, indent=2), encoding="utf-8")
    print(f"\n{len(census)} stores written to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
