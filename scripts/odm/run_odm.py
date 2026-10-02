#!/usr/bin/env python3
"""Compute the open-source reference solution of a block with OpenDroneMap.

The exposure positions of the timestamp file enter the adjustment as
observations with a ten-metre a priori accuracy, which leaves the shape of the
block to the imagery; the camera is self-calibrated with a Brown frame model;
and the surveyed targets of the convergent block are kept out of the adjustment
so that they remain an independent check. The surface model
is written without gap filling, at the cell size of the surface it replaces.

    python run_odm.py D1 [--fast]

Works in odm_runs/<block>, or wherever ODM_PROJECTS points;
--fast runs a quick, lower-quality pass to check the setup.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("ODM_PROJECTS", PROJECT / "odm_runs"))
IMAGE = "opendronemap/odm:latest"
# the CUDA build runs dense matching on the graphics card; features stay DSP-SIFT
GPU_IMAGE = "opendronemap/odm:gpu"
BLOCKS = {
    "D1": {"dataset": "D1_154_building", "images": "D1_images", "cell_cm": 8.74},
    "D2": {"dataset": "D2_111_nadir", "images": "D2_images", "cell_cm": 10.41},
}
POSITION_ACCURACY_M = 10.0


def write_geo(block: str, name: str) -> Path:
    """Exposure positions from the parsed timestamp records, WGS 84 ellipsoidal."""
    spec = BLOCKS[block]
    records = PROJECT / "results" / "parsed_inputs" / f"{spec['dataset']}_camera_records.csv"
    lines = ["EPSG:4326"]
    with records.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lines.append(f"{row['image_name']} {row['mrk_lon']} {row['mrk_lat']} "
                         f"{row['mrk_alt']}")
    path = WORK / name / "geo.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("block", choices=sorted(BLOCKS))
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    spec = BLOCKS[args.block]
    name = args.block + ("_fast" if args.fast else "")
    write_geo(args.block, name)
    images = PROJECT / spec["dataset"] / spec["images"]

    options = [
        "--geo", f"/datasets/{name}/geo.txt",
        "--gps-accuracy", str(POSITION_ACCURACY_M),
        "--camera-lens", "brown",
        "--dsm",
        "--dem-gapfill-steps", "0",
        "--skip-3dmodel",
        "--rerun-all",
    ]
    if args.fast:
        options += ["--feature-quality", "lowest", "--pc-quality", "lowest",
                    "--dem-resolution", "50", "--orthophoto-resolution", "50"]
    else:
        # features at the full image resolution, forty thousand per image
        options += ["--feature-quality", "ultra", "--min-num-features", "40000",
                    "--pc-quality", "high",
                    "--dem-resolution", str(spec["cell_cm"]),
                    "--orthophoto-resolution", str(spec["cell_cm"])]

    gpu = subprocess.run(["docker", "image", "inspect", GPU_IMAGE],
                         capture_output=True).returncode == 0
    options += ["--feature-type", "dspsift"]
    command = ["docker", "run", "--rm", *(["--gpus", "all"] if gpu else []),
               "-v", f"{WORK.as_posix()}:/datasets",
               "-v", f"{images.resolve().as_posix()}:/datasets/{name}/images:ro",
               GPU_IMAGE if gpu else IMAGE, "--project-path", "/datasets", name,
               *options]
    log = WORK / name / "odm.log"
    started = time.time()
    with log.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    print(f"{name}: exit {result.returncode} after {(time.time() - started) / 60:.1f} min,"
          f" log {log}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
