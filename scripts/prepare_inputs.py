#!/usr/bin/env python3
"""The inputs every later stage shares: camera records and candidate pairs.

Each block's exposures are read from the timestamp file the aircraft writes
beside its imagery, one line per exposure in the order they were taken, and are
joined to the images in filename order. The candidate pair list of a block is
then fixed once, before any matching, and every method is run on that same list:

    temporal pairs   each image with the next eight exposures of the flight
    spatial pairs    each image with its twelve nearest exposure positions,
                     by great-circle distance on the recorded positions

A pair found both ways is kept once. The list is written twice, as the plain
image-name pairs the matchers read and as a table carrying how each pair was
selected.

The image names are taken from the image directory when the imagery is present
and otherwise from the camera export of the reference solution, which lists the
same images, so the lists can be rebuilt without the imagery.

Writes results/parsed_inputs and results/pairs.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT  # noqa: E402

BLOCKS = {
    "D1_154_building": "D1_images",
    "D2_111_nadir": "D2_images",
}
TEMPORAL_GAP = 8
SPATIAL_NEIGHBOURS = 12
EARTH_RADIUS_M = 6_371_008.8

PARSED_DIR = PROJECT_ROOT / "results" / "parsed_inputs"
PAIRS_DIR = PROJECT_ROOT / "results" / "pairs"

RECORD_COLUMNS = ["dataset", "image_order_index", "image_name", "mrk_time",
                  "gps_week", "mrk_lat", "mrk_lon", "mrk_alt"]
PAIR_COLUMNS = ["dataset", "pair_id", "image1", "image2", "image1_order_index",
                "image2_order_index", "order_gap", "distance_m",
                "is_temporal_neighbor", "is_spatial_neighbor", "pair_source"]

# one exposure: index, GPS time of week, [week], three antenna offsets, then
# latitude, longitude and ellipsoidal height, each followed by its tag
EXPOSURE = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+\[(\d+)\].*?"
    r"([-+]?\d+\.\d+),Lat\s+([-+]?\d+\.\d+),Lon\s+([-+]?\d+\.\d+),Ellh")


def image_names(dataset: str) -> list[str]:
    folder = PROJECT_ROOT / dataset / BLOCKS[dataset]
    if folder.is_dir():
        names = [p.name for p in folder.iterdir() if p.suffix.lower() == ".jpg"]
    else:
        export = PROJECT_ROOT / dataset / "exports" / "cameras_estimated_and_reference.csv"
        with export.open(encoding="utf-8-sig") as handle:
            names = [f"{row['label']}.JPG" for row in csv.DictReader(handle)
                     if row.get("label")]
    return sorted(names, key=str.casefold)


def exposures(dataset: str) -> list[dict]:
    path = next(iter(sorted((PROJECT_ROOT / dataset / "RTK_data")
                            .glob("*Timestamp.MRK"))))
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EXPOSURE.match(line)
        if match:
            _, time, week, lat, lon, height = match.groups()
            records.append({"mrk_time": time, "gps_week": week, "mrk_lat": lat,
                            "mrk_lon": lon, "mrk_alt": height})
    return records


def great_circle(first: dict, second: dict) -> float:
    phi1, phi2 = math.radians(first["lat"]), math.radians(second["lat"])
    dphi = phi2 - phi1
    dlambda = math.radians(second["lon"] - first["lon"])
    a = (math.sin(dphi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def candidate_pairs(cameras: list[dict]) -> list[dict]:
    selected = {}

    def add(first: dict, second: dict, how: str) -> None:
        if first["order"] > second["order"]:
            first, second = second, first
        key = (first["order"], second["order"])
        entry = selected.setdefault(key, {"first": first, "second": second,
                                          "temporal": False, "spatial": False})
        entry[how] = True

    by_order = {camera["order"]: camera for camera in cameras}
    for camera in cameras:
        for gap in range(1, TEMPORAL_GAP + 1):
            if camera["order"] + gap in by_order:
                add(camera, by_order[camera["order"] + gap], "temporal")
    for camera in cameras:
        nearest = sorted((great_circle(camera, other), other["order"], other)
                         for other in cameras if other is not camera)
        for _, _, other in nearest[:SPATIAL_NEIGHBOURS]:
            add(camera, other, "spatial")
    return [selected[key] for key in sorted(selected)]


def write(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    for dataset in BLOCKS:
        names, records = image_names(dataset), exposures(dataset)
        if len(names) != len(records):
            raise SystemExit(f"{dataset}: {len(names)} images against "
                             f"{len(records)} exposures in the timestamp file")
        rows = [{"dataset": dataset, "image_order_index": order, "image_name": name,
                 **record}
                for order, (name, record) in enumerate(zip(names, records), start=1)]
        write(PARSED_DIR / f"{dataset}_camera_records.csv", rows, RECORD_COLUMNS)

        cameras = [{"order": row["image_order_index"], "name": row["image_name"],
                    "lat": float(row["mrk_lat"]), "lon": float(row["mrk_lon"])}
                   for row in rows]
        pairs = candidate_pairs(cameras)
        table = []
        for index, pair in enumerate(pairs, start=1):
            first, second = pair["first"], pair["second"]
            source = ("both" if pair["temporal"] and pair["spatial"]
                      else "temporal" if pair["temporal"] else "spatial")
            table.append({
                "dataset": dataset,
                "pair_id": f"{dataset}_pair_{index:06d}",
                "image1": first["name"], "image2": second["name"],
                "image1_order_index": first["order"],
                "image2_order_index": second["order"],
                "order_gap": second["order"] - first["order"],
                "distance_m": f"{great_circle(first, second):.3f}",
                "is_temporal_neighbor": pair["temporal"],
                "is_spatial_neighbor": pair["spatial"],
                "pair_source": source,
            })
        write(PAIRS_DIR / f"{dataset}_pairs.csv", table, PAIR_COLUMNS)
        (PAIRS_DIR / f"{dataset}_pairs.txt").write_text(
            "".join(f"{row['image1']} {row['image2']}\n" for row in table),
            encoding="utf-8")
        print(f"{dataset}: {len(rows)} exposures, {len(table)} candidate pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
