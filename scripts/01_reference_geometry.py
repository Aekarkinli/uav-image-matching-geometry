#!/usr/bin/env python3
"""Recover reference exterior orientation and characterise the image networks.

Reads the reference bundle-adjustment export for each block, converts the
camera poses into a local east-north-up frame, and derives the geometric
descriptors that the matching experiment is stratified by: camera tilt from
nadir, viewing-direction convergence angle for every scheduled image pair,
base-to-height ratio, and image density.
"""

from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
OUT_DIR = RESULTS / "reference_geometry"

DATASETS = {
    "D1": {
        "label": "D1_154_building",
        "xml": PROJECT_ROOT / "D1_154_building" / "exports" / "reference_cameras.xml",
        "pairs": PROJECT_ROOT / "results" / "pairs" / "D1_154_building_pairs.csv",
        "images": PROJECT_ROOT / "D1_154_building" / "D1_images",
        "cloud": PROJECT_ROOT / "D1_154_building" / "exports" / "D1_sparse_cloud.ply",
        "rtk": PROJECT_ROOT / "results" / "parsed_inputs" / "D1_154_building_camera_records.csv",
    },
    "D2": {
        "label": "D2_111_nadir",
        "xml": PROJECT_ROOT / "D2_111_nadir" / "exports" / "reference_cameras.xml",
        "pairs": PROJECT_ROOT / "results" / "pairs" / "D2_111_nadir_pairs.csv",
        "images": PROJECT_ROOT / "D2_111_nadir" / "D2_images",
        "cloud": PROJECT_ROOT / "D2_111_nadir" / "exports" / "D2_sparse_cloud.ply",
        "rtk": PROJECT_ROOT / "results" / "parsed_inputs" / "D2_111_nadir_camera_records.csv",
    },
}

# The reference products are exported in the national projected system.
REFERENCE_CRS = "EPSG:5256"  # TUREF / TM36
ECEF_CRS = "EPSG:4978"
SENSOR_WIDTH_MM = 13.2  # FC6310R

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass
class Camera:
    label: str
    centre_ecef: np.ndarray
    rotation_ecef: np.ndarray  # camera-to-world rotation


def parse_matrix(text: str, shape) -> np.ndarray:
    values = [float(v) for v in text.split()]
    return np.array(values, dtype=float).reshape(shape)


def ecef_to_geodetic(x: float, y: float, z: float):
    """Closed-form Bowring conversion from ECEF to latitude, longitude, height."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    b = WGS84_A * (1.0 - WGS84_F)
    ep2 = (WGS84_A**2 - b**2) / b**2
    theta = math.atan2(z * WGS84_A, p * b)
    lat = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * math.cos(theta) ** 3,
    )
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    height = p / math.cos(lat) - n
    return lat, lon, height


def enu_basis(lat: float, lon: float) -> np.ndarray:
    """Rows are the east, north and up unit vectors expressed in ECEF."""
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return np.vstack([east, north, up])


def read_chunk(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    chunk = root.find("chunk")
    transform = chunk.find("transform")
    rotation = parse_matrix(transform.find("rotation").text, (3, 3))
    translation = parse_matrix(transform.find("translation").text, (3,))
    scale = float(transform.find("scale").text)

    cameras = []
    for element in chunk.find("cameras").iter("camera"):
        node = element.find("transform")
        if node is None:
            continue
        local = parse_matrix(node.text, (4, 4))
        centre_local = local[:3, 3]
        rotation_local = local[:3, :3]
        centre_ecef = scale * rotation @ centre_local + translation
        rotation_ecef = rotation @ rotation_local
        cameras.append(Camera(element.get("label"), centre_ecef, rotation_ecef))

    sensors = []
    for sensor in chunk.find("sensors").iter("sensor"):
        entry = {"label": sensor.get("label"), "type": sensor.get("type")}
        resolution = sensor.find("resolution")
        if resolution is not None:
            entry["width"] = int(resolution.get("width"))
            entry["height"] = int(resolution.get("height"))
        calibration = sensor.find("calibration")
        if calibration is not None:
            for child in calibration:
                if child.tag != "resolution" and child.text:
                    entry[child.tag] = float(child.text)
        sensors.append(entry)
    return cameras, sensors, scale


def build_local_frame(cameras):
    centroid = np.mean([c.centre_ecef for c in cameras], axis=0)
    lat, lon, _ = ecef_to_geodetic(*centroid)
    basis = enu_basis(lat, lon)
    records = {}
    for camera in cameras:
        centre = basis @ (camera.centre_ecef - centroid)
        rotation = basis @ camera.rotation_ecef
        optical_axis = rotation @ np.array([0.0, 0.0, 1.0])
        optical_axis /= np.linalg.norm(optical_axis)
        # tilt measured from the nadir direction, which is -up in the local frame
        tilt = math.degrees(math.acos(np.clip(-optical_axis[2], -1.0, 1.0)))
        azimuth = math.degrees(math.atan2(optical_axis[0], optical_axis[1])) % 360.0
        records[camera.label] = {
            "centre": centre,
            "rotation": rotation,
            "axis": optical_axis,
            "tilt_deg": tilt,
            "azimuth_deg": azimuth,
        }
    return records, centroid, (lat, lon)


def read_ply_points(path: Path, max_points: int = 400000) -> np.ndarray:
    """Read the vertex coordinates of a binary little-endian PLY file."""
    type_map = {
        "float": "f4",
        "double": "f8",
        "uchar": "u1",
        "char": "i1",
        "int": "i4",
        "uint": "u4",
        "short": "i2",
        "ushort": "u2",
    }
    with path.open("rb") as handle:
        header = b""
        while b"end_header" not in header:
            line = handle.readline()
            if not line:
                raise ValueError(f"malformed PLY header in {path}")
            header += line
        text = header.decode("ascii", errors="replace")
        count = int(
            [l for l in text.splitlines() if l.startswith("element vertex")][0].split()[-1]
        )
        dtype = [
            (parts[2], type_map[parts[1]])
            for parts in (l.split() for l in text.splitlines() if l.startswith("property"))
        ]
        data = np.fromfile(handle, dtype=np.dtype(dtype), count=min(count, max_points))
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(float)


def terrain_in_local_frame(cloud_path: Path, centroid_ecef, basis) -> np.ndarray:
    """Project the reference point cloud into the local east-north-up frame."""
    from pyproj import Transformer

    points = read_ply_points(cloud_path)
    transformer = Transformer.from_crs(REFERENCE_CRS, ECEF_CRS, always_xy=True)
    x, y, z = transformer.transform(points[:, 0], points[:, 1], points[:, 2])
    ecef = np.stack([x, y, z], axis=1)
    return (basis @ (ecef - centroid_ecef).T).T


def rtk_in_local_frame(path: Path, centroid_ecef, basis) -> dict:
    """On-board kinematic positions expressed in the same local frame.

    These are the antenna positions recorded at exposure. They differ from the
    perspective centres by the lever arm and by any timing offset, which is a
    systematic effect common to every configuration.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4979", ECEF_CRS, always_xy=True)
    positions = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                lat = float(row["mrk_lat"])
                lon = float(row["mrk_lon"])
                alt = float(row["mrk_alt"])
            except (KeyError, TypeError, ValueError):
                continue
            x, y, z = transformer.transform(lon, lat, alt)
            local = basis @ (np.array([x, y, z]) - centroid_ecef)
            positions[Path(row["image_name"]).stem] = local
    return positions


def percentiles(values, keys=(5, 25, 50, 75, 95)):
    array = np.asarray(values, dtype=float)
    return {f"p{k}": float(np.percentile(array, k)) for k in keys}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for key, spec in DATASETS.items():
        cameras, sensors, chunk_scale = read_chunk(spec["xml"])
        poses, centroid, (lat, lon) = build_local_frame(cameras)

        stems = {label.split(".")[0]: label for label in poses}
        rtk = rtk_in_local_frame(spec["rtk"], centroid, enu_basis(lat, lon))

        camera_rows = []
        for label, pose in poses.items():
            camera_rows.append(
                {
                    "dataset": spec["label"],
                    "camera_label": label,
                    "east_m": pose["centre"][0],
                    "north_m": pose["centre"][1],
                    "up_m": pose["centre"][2],
                    "tilt_from_nadir_deg": pose["tilt_deg"],
                    "viewing_azimuth_deg": pose["azimuth_deg"],
                    "axis_e": pose["axis"][0],
                    "axis_n": pose["axis"][1],
                    "axis_u": pose["axis"][2],
                    # rows of the rotation that maps camera axes into the local
                    # frame, needed for alignment free relative orientation
                    **{
                        f"r{i}{j}": pose["rotation"][i, j]
                        for i in range(3)
                        for j in range(3)
                    },
                    "rtk_east_m": rtk.get(label, [float("nan")] * 3)[0],
                    "rtk_north_m": rtk.get(label, [float("nan")] * 3)[1],
                    "rtk_up_m": rtk.get(label, [float("nan")] * 3)[2],
                }
            )
        camera_rows.sort(key=lambda r: r["camera_label"])
        path = OUT_DIR / f"{spec['label']}_reference_poses.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(camera_rows[0].keys()))
            writer.writeheader()
            writer.writerows(camera_rows)

        # Pairwise geometry for every scheduled pair.
        pair_rows = []
        with spec["pairs"].open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                a = row.get("image1") or row.get("image_1")
                b = row.get("image2") or row.get("image_2")
                ka, kb = stems.get(Path(a).stem), stems.get(Path(b).stem)
                if ka is None or kb is None:
                    continue
                pa, pb = poses[ka], poses[kb]
                baseline = float(np.linalg.norm(pa["centre"] - pb["centre"]))
                convergence = math.degrees(
                    math.acos(np.clip(float(np.dot(pa["axis"], pb["axis"])), -1.0, 1.0))
                )
                mean_height = float(
                    np.mean([pa["centre"][2], pb["centre"][2]])
                )
                pair_rows.append(
                    {
                        "dataset": spec["label"],
                        "pair_id": row.get("pair_id", ""),
                        "image1": a,
                        "image2": b,
                        "pair_source": row.get("pair_source", ""),
                        "order_gap": row.get("order_gap", ""),
                        "baseline_m": baseline,
                        "convergence_angle_deg": convergence,
                        "tilt1_deg": pa["tilt_deg"],
                        "tilt2_deg": pb["tilt_deg"],
                        "mean_camera_height_m": mean_height,
                    }
                )
        path = OUT_DIR / f"{spec['label']}_pair_geometry.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)

        centres = np.array([p["centre"] for p in poses.values()])
        tilts = [p["tilt_deg"] for p in poses.values()]
        baselines = [r["baseline_m"] for r in pair_rows]
        convergences = [r["convergence_angle_deg"] for r in pair_rows]
        extent_e = float(centres[:, 0].max() - centres[:, 0].min())
        extent_n = float(centres[:, 1].max() - centres[:, 1].min())

        terrain = terrain_in_local_frame(spec["cloud"], centroid, enu_basis(lat, lon))
        terrain_median = float(np.median(terrain[:, 2]))
        terrain_relief = float(
            np.percentile(terrain[:, 2], 98) - np.percentile(terrain[:, 2], 2)
        )
        flying_height = float(np.median(centres[:, 2]) - terrain_median)

        focal_px = next(
            (s["f"] for s in sensors if "f" in s), float("nan")
        )
        image_width = next((s["width"] for s in sensors if "width" in s), 5472)
        gsd_m = flying_height / focal_px if focal_px == focal_px else float("nan")
        footprint_m = image_width * gsd_m

        summary[key] = {
            "dataset": spec["label"],
            "image_count": len(poses),
            "pair_count": len(pair_rows),
            "sensors": sensors,
            "frame_scale": chunk_scale,
            "block_centroid_lat_deg": math.degrees(lat),
            "block_centroid_lon_deg": math.degrees(lon),
            "camera_extent_east_m": extent_e,
            "camera_extent_north_m": extent_n,
            "camera_footprint_area_ha": extent_e * extent_n / 10000.0,
            "images_per_hectare": len(poses) / max(extent_e * extent_n / 10000.0, 1e-9),
            "tilt_from_nadir_deg": {
                "mean": float(np.mean(tilts)),
                "std": float(np.std(tilts)),
                "min": float(np.min(tilts)),
                "max": float(np.max(tilts)),
                **percentiles(tilts),
            },
            "tilt_fraction_above_20deg": float(np.mean(np.asarray(tilts) > 20.0)),
            "baseline_m": {"mean": float(np.mean(baselines)), **percentiles(baselines)},
            "convergence_angle_deg": {
                "mean": float(np.mean(convergences)),
                **percentiles(convergences),
            },
            "camera_height_spread_m": float(centres[:, 2].max() - centres[:, 2].min()),
            "terrain_median_height_m": terrain_median,
            "terrain_relief_p2_p98_m": terrain_relief,
            "flying_height_above_terrain_m": flying_height,
            "calibrated_focal_length_px": focal_px,
            "ground_sample_distance_m": gsd_m,
            "ground_footprint_width_m": footprint_m,
            "median_base_to_height_ratio": float(
                np.median(baselines) / max(flying_height, 1e-9)
            ),
        }
        print(
            f"{spec['label']}: n={len(poses)} pairs={len(pair_rows)} "
            f"tilt median={np.median(tilts):.1f} deg "
            f"(p5={np.percentile(tilts,5):.1f}, p95={np.percentile(tilts,95):.1f}) "
            f"convergence median={np.median(convergences):.1f} deg"
        )

    (OUT_DIR / "network_geometry_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
