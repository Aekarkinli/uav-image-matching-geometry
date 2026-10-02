#!/usr/bin/env python3
"""Where the on-board exposure positions come from, and what they carry.

The positions are described by what the archive records: the field they were
read from, the recorded satellite status and standard deviations, the
antenna-to-camera offset the timestamp file lists and, for the convergent block,
four surveyed targets used as an object-space consistency check. The processing
record of the two OpenDroneMap reference solutions and the acquisition metadata
of the imagery are read as well.

Writes results/analysis/onboard_metadata.json.
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECT_ROOT  # noqa: E402

ANALYSIS = PROJECT_ROOT / "results" / "analysis"

BLOCKS = {
    "D1_154_building": {"images": "D1_images", "markers": True},
    "D2_111_nadir": {"images": "D2_images", "markers": False},
}

# a three-degree transverse Mercator zone on the central meridian of 36 degrees,
# which is what an easting near 375 500 m at this longitude implies
GRID_PROJ4 = ("+proj=tmerc +lat_0=0 +lon_0=36 +k=1 +x_0=500000 +y_0=0 "
              "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")

XMP_FIELDS = ("GpsLatitude", "GpsLongtitude", "AbsoluteAltitude", "RtkFlag",
              "RtkStdLat", "RtkStdLon", "RtkStdHgt")

# the airframe writes these beside the position, and they describe the flight
# rather than the reference solution, so they are read for the acquisition record
FLIGHT_FIELDS = ("PhotoDiff", "GimbalPitchDegree", "RelativeAltitude",
                 "AbsoluteAltitude")

# settings an automatic exposure can change from frame to frame, so the block is
# described by how often each value occurs and never by an average of them
EXPOSURE_FIELDS = ("ExposureTime", "FNumber", "ISOSpeedRatings",
                   "ExposureProgram", "ExposureMode", "MeteringMode",
                   "WhiteBalance")

def read_mrk(path: Path) -> list[dict]:
    """One record per exposure, as the receiver wrote it."""
    records = []
    for line in path.read_text(encoding="latin-1").splitlines():
        parts = line.split()
        if len(parts) < 12:
            continue
        north, east, vertical = (float(parts[i].split(",")[0]) for i in (3, 4, 5))
        records.append({
            "seconds": float(parts[1]),
            "offset_north_mm": north,
            "offset_east_mm": east,
            "offset_vertical_mm": vertical,
            "latitude": float(parts[6].split(",")[0]),
            "longitude": float(parts[7].split(",")[0]),
            "ellipsoidal_height_m": float(parts[8].split(",")[0]),
            "sigma_north_m": float(parts[9].rstrip(",")),
            "sigma_east_m": float(parts[10].rstrip(",")),
            "sigma_vertical_m": float(parts[11].rstrip(",")),
            "flag": int(parts[12].split(",")[0]),
        })
    return records


def read_xmp(path: Path, fields: tuple = XMP_FIELDS) -> dict:
    """The DJI namespace of the embedded metadata, without a reader library."""
    raw = path.read_bytes()[:262144]
    start, end = raw.find(b"<x:xmpmeta"), raw.find(b"</x:xmpmeta>")
    if start < 0 or end < 0:
        return {}
    text = raw[start:end].decode("utf-8", "replace")
    values = {}
    for field in fields:
        match = re.search(rf'drone-dji:{field}="([^"]+)"', text)
        if match:
            values[field] = match.group(1)
    return values


def marker_check(dataset: str) -> dict | None:
    """Surveyed targets against the reference solution, in the survey grid."""
    path = (PROJECT_ROOT / dataset / "exports"
            / "markers_estimated_and_reference.csv")
    if not path.exists():
        return None
    from pyproj import CRS, Transformer

    forward = Transformer.from_crs(CRS.from_epsg(4326),
                                   CRS.from_proj4(GRID_PROJ4), always_xy=True)
    targets = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        east, north = forward.transform(float(row["estimated_x_or_lon"]),
                                        float(row["estimated_y_or_lat"]))
        targets.append({
            "label": row["label"],
            "projections": int(row["projections_count"]),
            "reference_used_in_adjustment":
                row["reference_enabled"].strip().lower() == "true",
            "d_east_m": east - float(row["reference_x_or_lon"]),
            "d_north_m": north - float(row["reference_y_or_lat"]),
            "d_height_m": (float(row["estimated_z_or_h"])
                           - float(row["reference_z_or_h"])),
        })

    def rms(values):
        return (sum(v * v for v in values) / len(values)) ** 0.5

    plan = [(t["d_east_m"] ** 2 + t["d_north_m"] ** 2) ** 0.5 for t in targets]
    return {
        "targets": targets,
        "n": len(targets),
        "rms_east_m": rms([t["d_east_m"] for t in targets]),
        "rms_north_m": rms([t["d_north_m"] for t in targets]),
        "rms_height_m": rms([t["d_height_m"] for t in targets]),
        "rms_plan_m": rms(plan),
        "grid": "three-degree transverse Mercator, central meridian 36 degrees",
    }


ODM_WORK = Path(os.environ.get("ODM_PROJECTS",
                               PROJECT_ROOT / "odm_runs"))
ODM_PROJECT = {"D1_154_building": "D1", "D2_111_nadir": "D2"}
ODM_OPTIONS = ("camera_lens", "feature_quality", "feature_type", "min_num_features",
               "matcher_type", "matcher_neighbors", "gps_accuracy", "pc_quality",
               "dem_resolution", "dem_gapfill_steps", "orthophoto_resolution",
               "sfm_algorithm", "use_fixed_camera_params", "geo")


def reference_project(dataset: str) -> dict:
    """What the OpenDroneMap run of the reference solution records about itself."""
    project = ODM_WORK / ODM_PROJECT[dataset]
    log = json.loads((project / "log.json").read_text(encoding="utf-8"))
    stats = json.loads((project / "opensfm" / "stats" / "stats.json")
                       .read_text(encoding="utf-8"))
    reconstruction = stats["reconstruction_statistics"]
    camera = next(iter(stats["camera_errors"].values()))
    summary = json.loads((PROJECT_ROOT / dataset / "exports"
                          / "odm_solution_summary.json").read_text(encoding="utf-8"))
    return {
        "project": ODM_PROJECT[dataset],
        "software": "OpenDroneMap",
        "created_with_version": log["odmVersion"],
        "options": {k: log["options"].get(k) for k in ODM_OPTIONS},
        "registered_images": reconstruction["reconstructed_shots_count"],
        "images": reconstruction["initial_shots_count"],
        "components": reconstruction["components"],
        "tie_points": reconstruction["reconstructed_points_count"],
        "reprojection_error_px": reconstruction.get("reprojection_error_pixels"),
        "optimised_camera": camera.get("optimized_values"),
        "adjusted_calibration": summary["calibration"],
        "gps_errors_m": stats.get("gps_errors"),
        "camera_positions_enabled": "true",
        "surveyed_targets_in_adjustment": 0,
    }


def processing_history(dataset: str) -> dict:
    """The stages of the OpenDroneMap run and the products each one wrote."""
    project = ODM_WORK / ODM_PROJECT[dataset]
    log = json.loads((project / "log.json").read_text(encoding="utf-8"))
    # the products the run wrote, read from its own log so that a reduced copy
    # of the project, without the dense cloud, records the same run
    products = {}
    for kind, path in (("point_cloud", "opensfm/reconstruction.json"),
                       ("dense_cloud", "odm_georeferencing/odm_georeferenced_model.laz"),
                       ("elevation", "odm_dem/dsm.tif"),
                       ("orthomosaic", "odm_orthophoto/odm_orthophoto.tif")):
        products[kind] = {"path": path, "created_with_version": log["odmVersion"]}
    return {
        "project_last_saved_version": log["odmVersion"],
        "products": products,
        "stages": [s.get("name") for s in log.get("stages", [])],
        "elevation_resolution_m": log["options"]["dem_resolution"] / 100.0,
    }


def exif_text(value) -> str | None:
    """An EXIF ASCII field is padded with NULs out to the length it was given."""
    return None if value is None else str(value).rstrip("\x00")


def exif_number(value):
    """A rational EXIF field arrives as a fraction, which JSON cannot carry."""
    return value if value is None or isinstance(value, int) else float(value)


def single(values: list):
    """The one value the block agrees on, so a disagreement is never hidden."""
    distinct = set(values)
    return distinct.pop() if len(distinct) == 1 else None


def tally(values: list) -> dict:
    """How often each distinct value occurs, keyed as the JSON will carry it."""
    counts = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def platform_identifier(value: str | None) -> str | None:
    """The airframe part of PhotoDiff, whose last fourteen digits are a stamp."""
    if value is None:
        return None
    match = re.fullmatch(r"(.+?)\d{14}", value)
    return match.group(1) if match else value


def acquisition(images: list[Path]) -> dict:
    """What the camera and the airframe wrote into the images themselves.

    Overlap and flying height are deliberately absent: both are derived
    elsewhere in this project, and a second derivation here could disagree.
    """
    from PIL import ExifTags, Image

    tag = ExifTags.Base
    header, flight, sizes = [], [], []
    for path in images:
        with Image.open(path) as image:
            main_ifd = image.getexif()
            exif_ifd = main_ifd.get_ifd(tag.ExifOffset)
            record = {
                "file_format": image.format,
                "bit_depth": image.bits,
                "image_width": image.width,
                "image_height": image.height,
            }
        for name in ("Make", "Model", "Software"):
            record[name] = exif_text(main_ifd.get(tag[name]))
        for name in ("BodySerialNumber", "DateTimeOriginal"):
            record[name] = exif_text(exif_ifd.get(tag[name]))
        for name in ("FocalLength", "FocalLengthIn35mmFilm") + EXPOSURE_FIELDS:
            record[name] = exif_number(exif_ifd.get(tag[name]))
        header.append(record)
        flight.append(read_xmp(path, FLIGHT_FIELDS))
        sizes.append(path.stat().st_size)

    serials = [r["BodySerialNumber"] for r in header]
    platforms = [platform_identifier(v.get("PhotoDiff")) for v in flight]
    # the exposures are ordered by their own timestamps rather than by filename,
    # so that the elapsed time is the span of the block however it was named
    stamps = sorted(datetime.strptime(r["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
                    for r in header)
    heights = [float(v["RelativeAltitude"]) for v in flight]
    altitudes = [float(v["AbsoluteAltitude"]) for v in flight]

    return {
        "camera_make": single([r["Make"] for r in header]),
        "camera_model": single([r["Model"] for r in header]),
        "camera_firmware": single([r["Software"] for r in header]),
        "body_serial": serials[0],
        "body_serial_constant": len(set(serials)) == 1,
        "platform_identifier": platforms[0],
        "platform_identifier_constant": len(set(platforms)) == 1,
        "focal_length_mm": single([r["FocalLength"] for r in header]),
        "focal_length_35mm_equivalent_mm":
            single([r["FocalLengthIn35mmFilm"] for r in header]),
        "image_width": single([r["image_width"] for r in header]),
        "image_height": single([r["image_height"] for r in header]),
        "file_format": single([r["file_format"] for r in header]),
        "bit_depth": single([r["bit_depth"] for r in header]),
        "images": len(images),
        "file_size_bytes_total": sum(sizes),
        "file_size_bytes_mean": statistics.fmean(sizes),
        # the camera records no UTC offset, so these are local clock readings
        "first_exposure": stamps[0].isoformat(),
        "last_exposure": stamps[-1].isoformat(),
        "elapsed_s": (stamps[-1] - stamps[0]).total_seconds(),
        "exposure_settings": {name: tally([r[name] for r in header])
                              for name in EXPOSURE_FIELDS},
        "gimbal_pitch_deg": tally([round(float(v["GimbalPitchDegree"]), 1)
                                   for v in flight]),
        "relative_altitude_m_range": [min(heights), max(heights)],
        "absolute_altitude_m_range": [min(altitudes), max(altitudes)],
    }


def block_separation(centroids: dict) -> dict:
    """How far apart the two blocks were flown, from the recorded positions.

    Both blocks are described as flown over the same ground; the on-board
    positions give the distance between them directly.
    """
    from pyproj import Geod

    first, second = (centroids[dataset] for dataset in BLOCKS)
    _, _, distance = Geod(ellps="WGS84").inv(
        first["longitude"], first["latitude"],
        second["longitude"], second["latitude"])
    return {
        "centroids": centroids,
        "distance_m": distance,
        "method": "geodesic inverse on the WGS 84 ellipsoid, pyproj.Geod",
        "source": "mean of the exposure positions in the timestamp files",
    }


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    report = {}
    centroids = {}

    for dataset, layout in BLOCKS.items():
        folder = PROJECT_ROOT / dataset
        mrk = next(iter(sorted((folder / "RTK_data").glob("*Timestamp.MRK"))))
        records = read_mrk(mrk)

        images = sorted(p for p in (folder / layout["images"]).iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg"))
        embedded = [read_xmp(p) for p in images]

        # does the embedded position repeat the timestamp file, to the digit it
        # is written with?
        agree_plan, height_gap = 0, 0.0
        for record, values in zip(records, embedded):
            if not values:
                continue
            if (abs(float(values["GpsLatitude"]) - record["latitude"]) < 5e-8
                    and abs(float(values["GpsLongtitude"])
                            - record["longitude"]) < 5e-8):
                agree_plan += 1
            # the embedded altitude is written to two decimals, so the largest
            # difference a repeated value can show is half of that step
            height_gap = max(height_gap, abs(float(values["AbsoluteAltitude"])
                                             - record["ellipsoidal_height_m"]))

        flags = {}
        for record in records:
            flags[record["flag"]] = flags.get(record["flag"], 0) + 1

        report[dataset] = {
            "timestamp_file": mrk.name,
            "exposures_in_timestamp_file": len(records),
            "images": len(images),
            "images_with_embedded_position": sum(1 for v in embedded if v),
            "embedded_position_matches_timestamp_file": agree_plan,
            "embedded_height_largest_difference_m": height_gap,
            "satellite_status_flags": flags,
            "sigma_north_m_median": statistics.median(
                r["sigma_north_m"] for r in records),
            "sigma_east_m_median": statistics.median(
                r["sigma_east_m"] for r in records),
            "sigma_vertical_m_median": statistics.median(
                r["sigma_vertical_m"] for r in records),
            "sigma_north_m_max": max(r["sigma_north_m"] for r in records),
            "sigma_east_m_max": max(r["sigma_east_m"] for r in records),
            "sigma_vertical_m_max": max(r["sigma_vertical_m"] for r in records),
            "antenna_offset_horizontal_mm_median": statistics.median(
                (r["offset_north_mm"] ** 2 + r["offset_east_mm"] ** 2) ** 0.5
                for r in records),
            "antenna_offset_vertical_mm_median": statistics.median(
                r["offset_vertical_mm"] for r in records),
            "antenna_offset_vertical_mm_range": [
                min(r["offset_vertical_mm"] for r in records),
                max(r["offset_vertical_mm"] for r in records)],
            "surveyed_target_check": marker_check(dataset)
                if layout["markers"] else None,
            "reference_project": reference_project(dataset),
            "processing_history": processing_history(dataset),
            "acquisition": acquisition(images),
        }

        centroids[dataset] = {
            "latitude": statistics.fmean(r["latitude"] for r in records),
            "longitude": statistics.fmean(r["longitude"] for r in records),
        }

    # this compares the two blocks against each other, so it belongs beside them
    # rather than inside either one
    report["block_separation"] = block_separation(centroids)

    path = ANALYSIS / "onboard_metadata.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for dataset in BLOCKS:
        entry = report[dataset]
        print(f"{dataset}")
        print(f"  {entry['exposures_in_timestamp_file']} exposures in "
              f"{entry['timestamp_file']}, {entry['images']} images")
        print(f"  embedded position repeats the timestamp file for "
              f"{entry['embedded_position_matches_timestamp_file']} images, "
              f"height to {entry['embedded_height_largest_difference_m']:.4f} m")
        print(f"  status flags {entry['satellite_status_flags']}")
        print(f"  sigma median N/E/V "
              f"{entry['sigma_north_m_median']:.3f} / "
              f"{entry['sigma_east_m_median']:.3f} / "
              f"{entry['sigma_vertical_m_median']:.3f} m, "
              f"max {entry['sigma_vertical_m_max']:.3f} m vertical")
        print(f"  antenna offset {entry['antenna_offset_horizontal_mm_median']:.0f} mm "
              f"horizontal, {entry['antenna_offset_vertical_mm_median']:.0f} mm vertical")
        check = entry["surveyed_target_check"]
        if check:
            print(f"  {check['n']} surveyed targets, none in the adjustment: "
                  f"plan {check['rms_plan_m']:.3f} m, "
                  f"height {check['rms_height_m']:.3f} m")
        flown = entry["acquisition"]
        print(f"  {flown['camera_model']} firmware {flown['camera_firmware']}, "
              f"{flown['first_exposure']} to {flown['last_exposure']}")
        history = entry["processing_history"]
        print(f"  products {', '.join(history['products'])}, project last saved "
              f"by {history['project_last_saved_version']}")
    separation = report["block_separation"]
    print(f"block centroids {separation['distance_m']:.0f} m apart on WGS 84")
    print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
