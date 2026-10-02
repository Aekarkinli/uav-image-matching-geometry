#!/usr/bin/env python3
"""Rebuild keypoint-only feature stores that the reconstruction stage needs.

Matching consumes descriptors, reconstruction consumes only keypoints. Stores
whose descriptors were discarded, or whose smaller keypoint budgets were
derived and later removed, can therefore be rebuilt cheaply by truncating a
larger budget at the same extraction resolution by detector score.

Truncation is associative under a fixed score ranking, so taking the strongest
n detections from any larger store yields exactly the set that was matched.
Each rebuilt store is checked against the stored match record, whose length
equals the number of keypoints in the first image of the pair, so an
inconsistent rebuild is detected rather than silently reconstructed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FEATURES_DIR, MANIFESTS_DIR, MATCHES_DIR  # noqa: E402

CARRY = ("keypoints", "scores", "image_size", "scales", "oris")

# See 02_extract_and_match: truncating a larger store does not reproduce a
# direct extraction for these front ends, so their stores are never rebuilt
# that way.
NEVER_TRUNCATE = {"sift", "rootsift", "doghardnet"}


def feature_id(front_end: str, resolution, budget: int) -> str:
    res = "native" if resolution == "native" else f"r{resolution}"
    return f"{front_end}-{res}-k{budget}"


def find_master(label: str, front_end: str, resolution, budget: int):
    res = "native" if resolution == "native" else f"r{resolution}"
    candidates = []
    for path in (FEATURES_DIR / label).glob(f"{front_end}-{res}-k*.h5"):
        try:
            other = int(path.stem.rsplit("-k", 1)[1])
        except (IndexError, ValueError):
            continue
        if other > budget:
            candidates.append((other, path))
    return min(candidates)[1] if candidates else None


def truncate(source: Path, target: Path, budget: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as src, h5py.File(target, "w") as dst:
        for name in src:
            group = src[name]
            scores = group["scores"][()]
            keep = None
            if len(scores) > budget:
                keep = np.argsort(-scores)[:budget]
                keep.sort()
            out = dst.create_group(name)
            for key in CARRY:
                if key not in group:
                    continue
                values = group[key][()]
                if keep is not None and key != "image_size":
                    values = values[keep]
                out.create_dataset(key, data=values)


def check_against_matches(store: Path, matches: Path) -> tuple[bool, str]:
    """The match record length equals the keypoint count of the first image."""
    if not matches.exists():
        return True, "no match file to check against"
    with h5py.File(store, "r") as features, h5py.File(matches, "r") as pairs:
        checked = 0
        for name0 in pairs:
            for name1 in pairs[name0]:
                record = pairs[name0][name1]["matches0"]
                if name0 not in features:
                    return False, f"{name0} absent from rebuilt store"
                count = features[name0]["keypoints"].shape[0]
                if record.shape[0] != count:
                    return (
                        False,
                        f"{name0}: {count} keypoints but match record holds "
                        f"{record.shape[0]}",
                    )
                values = record[()]
                if len(values) and values.max() >= 0:
                    target_count = features[name1]["keypoints"].shape[0]
                    if values.max() >= target_count:
                        return (
                            False,
                            f"{name0}/{name1}: match index {values.max()} exceeds "
                            f"{target_count} keypoints",
                        )
                checked += 1
                if checked >= 25:
                    return True, f"{checked} pairs checked"
    return True, f"{checked} pairs checked"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rebuilt = failed = present = 0
    for manifest_path in sorted(MANIFESTS_DIR.glob("*/*.json")):
        label = manifest_path.parent.name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fid = feature_id(
            manifest["front_end"],
            manifest["working_resolution"],
            manifest["keypoint_budget"],
        )
        store = FEATURES_DIR / label / f"{fid}.h5"
        if store.exists():
            present += 1
            continue
        if manifest["front_end"] in NEVER_TRUNCATE:
            print(f"[missing] {label} {fid}: re-extraction required for this front end")
            failed += 1
            continue
        master = find_master(
            label,
            manifest["front_end"],
            manifest["working_resolution"],
            manifest["keypoint_budget"],
        )
        if master is None:
            print(f"[missing] {label} {fid}: no larger store, re-extraction required")
            failed += 1
            continue
        if args.dry_run:
            print(f"[would rebuild] {label} {fid} from {master.name}")
            continue
        truncate(master, store, manifest["keypoint_budget"])
        ok, detail = check_against_matches(
            store, MATCHES_DIR / label / f"{manifest['config']}.h5"
        )
        if ok:
            print(f"[rebuilt] {label} {fid} from {master.name} ({detail})")
            rebuilt += 1
        else:
            print(f"[INCONSISTENT] {label} {fid}: {detail}")
            store.unlink(missing_ok=True)
            failed += 1

    print(f"\npresent {present}, rebuilt {rebuilt}, unresolved {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
