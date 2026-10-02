#!/usr/bin/env python3
"""Drive sparse reconstruction over every completed matching configuration.

Primary runs use a fixed random seed so that all configurations are compared
under identical conditions. A designated subset is additionally reconstructed
under several seeds, which measures the run to run variability of incremental
mapping and therefore sets the smallest difference the study can resolve.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import MANIFESTS_DIR, PROJECT_ROOT, SFM_REPORTS_DIR  # noqa: E402

WORKER = PROJECT_ROOT / "scripts" / "04_reconstruct.py"
STATUS_PATH = PROJECT_ROOT / "results" / "reconstruction_status.json"

DATASET_KEYS = {"D1_154_building": "D1", "D2_111_nadir": "D2"}

# Configurations repeated under several seeds to quantify mapper variability.
REPEATABILITY_CONFIGS = [
    "rootsift-lightglue-r2048-k8192",
    "superpoint-lightglue-r2048-k8192",
    "aliked-lightglue-r2048-k8192",
]
REPEATABILITY_SEEDS = [1, 2, 3, 4]


def completed_matching():
    """Every configuration whose matching run finished, newest cheapest first."""
    items = []
    for path in sorted(MANIFESTS_DIR.glob("*/*.json")):
        label = path.parent.name
        manifest = json.loads(path.read_text(encoding="utf-8"))
        items.append(
            {
                "dataset": DATASET_KEYS[label],
                "label": label,
                "config": manifest["config"],
                "front_end": manifest["front_end"],
                "resolution": manifest["working_resolution"],
                "budget": manifest["keypoint_budget"],
                "median_inliers": manifest["median_verified_inliers"],
            }
        )
    return items


def order_key(item):
    """Cheapest reconstructions first, so results appear early."""
    resolution = item["resolution"]
    rank = 10**6 if resolution == "native" else int(resolution)
    return (rank, item["median_inliers"], item["dataset"])


def is_done(label: str, config: str, seed: int) -> bool:
    run_id = config if seed == 0 else f"{config}__seed{seed}"
    return (SFM_REPORTS_DIR / label / f"{run_id}.json").exists()


def run(item, seed: int) -> int:
    command = [
        sys.executable,
        str(WORKER),
        "--dataset",
        item["dataset"],
        "--config",
        item["config"],
        "--seed",
        str(seed),
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--repeats-only",
        action="store_true",
        help="run only the multi seed repeatability set",
    )
    parser.add_argument(
        "--skip-repeats",
        action="store_true",
        help="run only the primary single seed reconstructions",
    )
    args = parser.parse_args()

    # Reconstruction needs the keypoints of every matched configuration. Stores
    # whose descriptors were discarded, or whose smaller budgets were derived
    # and later cleaned up, are restored here before the queue is built.
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "08_rebuild_keypoint_stores.py")],
        cwd=PROJECT_ROOT,
    )

    items = sorted(completed_matching(), key=order_key)
    jobs = []
    if not args.repeats_only:
        for item in items:
            if not is_done(item["label"], item["config"], 0):
                jobs.append((item, 0))
    if not args.skip_repeats:
        for item in items:
            if item["config"] not in REPEATABILITY_CONFIGS:
                continue
            for seed in REPEATABILITY_SEEDS:
                if not is_done(item["label"], item["config"], seed):
                    jobs.append((item, seed))
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"{len(items)} matched configurations, {len(jobs)} reconstructions pending")
    if args.dry_run:
        for item, seed in jobs:
            tag = "" if seed == 0 else f" seed {seed}"
            print(f"  {item['dataset']} {item['config']}{tag}")
        return 0

    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    for index, (item, seed) in enumerate(jobs, start=1):
        tag = "" if seed == 0 else f" seed {seed}"
        print(f"\n[{index}/{len(jobs)}] {item['dataset']} {item['config']}{tag}", flush=True)
        start = time.perf_counter()
        code = run(item, seed)
        elapsed = time.perf_counter() - start
        status["runs"].append(
            {
                "dataset": item["dataset"],
                "config": item["config"],
                "seed": seed,
                "returncode": code,
                "seconds": elapsed,
            }
        )
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if code != 0:
            print(f"  [warn] returncode {code}, continuing", flush=True)

    failed = [r for r in status["runs"] if r["returncode"] != 0]
    print(f"\ncompleted {len(status['runs']) - len(failed)}/{len(status['runs'])}")
    for item in failed:
        print(f"  failed: {item['dataset']} {item['config']} seed {item['seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
