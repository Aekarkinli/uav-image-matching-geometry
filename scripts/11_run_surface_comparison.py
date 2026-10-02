#!/usr/bin/env python3
"""Drive the surface comparison over a chosen set of configurations.

The set spans the full range of camera-network agreement observed at the
reference operating point, so that the question the stage answers is whether
those differences survive into the delivered surface. One block additionally
carries the same local feature at two extraction resolutions, which separates the
effect of the operating point from the effect of the front end.

Dense stereo holds the graphics memory for the duration of a run, so the
configurations are processed one at a time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT  # noqa: E402

WORKER = PROJECT_ROOT / "scripts" / "09_dense_and_surface.py"
SURFACE_DIR = PROJECT_ROOT / "results" / "surface"
STATUS_PATH = PROJECT_ROOT / "results" / "surface_status.json"

LABELS = {"D1": "D1_154_building", "D2": "D2_111_nadir"}

# Spanning set at the reference operating point, plus a resolution contrast.
SELECTION = [
    ("D1", "rootsift-lightglue-r2048-k8192"),
    ("D1", "superpoint-lightglue-r2048-k8192"),
    ("D1", "disk-lightglue-r2048-k8192"),
    ("D1", "aliked-lightglue-r2048-k8192"),
    ("D2", "rootsift-lightglue-r2048-k8192"),
    ("D2", "superpoint-lightglue-r2048-k8192"),
    ("D2", "disk-lightglue-r2048-k8192"),
    ("D2", "aliked-lightglue-r2048-k8192"),
    ("D1", "superpoint-lightglue-r1024-k8192"),
    ("D1", "doghardnet-lightglue-r2048-k8192"),
    ("D2", "doghardnet-lightglue-r2048-k8192"),
    # the detector-free matcher, at the only extraction resolution it fits in
    ("D1", "loftr-dense-r1024-kall"),
    ("D2", "loftr-dense-r1024-kall"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-image-size", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    pending = [
        item for item in SELECTION
        if not (SURFACE_DIR / LABELS[item[0]] / f"{item[1]}.json").exists()
    ]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(SELECTION)} configurations selected, {len(pending)} pending")
    if args.dry_run:
        for dataset, config in pending:
            print(f"  {dataset} {config}")
        return 0

    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    for index, (dataset, config) in enumerate(pending, start=1):
        print(f"\n[{index}/{len(pending)}] {dataset} {config}", flush=True)
        start = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(WORKER), "--dataset", dataset, "--config", config,
             "--max-image-size", str(args.max_image_size)],
            cwd=PROJECT_ROOT,
        )
        status["runs"].append({
            "dataset": dataset, "config": config,
            "returncode": completed.returncode,
            "seconds": time.perf_counter() - start,
        })
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if completed.returncode != 0:
            print(f"  [warn] returncode {completed.returncode}, continuing", flush=True)

    failed = [r for r in status["runs"] if r["returncode"] != 0]
    print(f"\ncompleted {len(status['runs']) - len(failed)}/{len(status['runs'])} "
          f"in {sum(r['seconds'] for r in status['runs'])/3600:.1f} h")
    for item in failed:
        print(f"  failed: {item['dataset']} {item['config']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
