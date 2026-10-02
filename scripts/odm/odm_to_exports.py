#!/usr/bin/env python3
"""Write an OpenDroneMap solution in the form the pipeline reads a reference in.

The analysis reads a reference solution through five files in each block's
exports/ folder. They are written here from the OpenDroneMap project:

    reference_cameras.xml   camera poses, as camera-to-ECEF transforms in a
                            frame whose own transform is the identity
    sensor_calibration.csv  the self-calibrated frame model, with the
                            principal point as an offset from the image centre
    <block>_DSM.tif         the surface model, reprojected to TUREF / TM36
    <block>_sparse_cloud.ply  the tie points, in TUREF / TM36
    <block>_Orthophoto.tif  the orthophoto, reprojected to TUREF / TM36

OpenSfM stores each shot as a world-to-camera rotation and translation in a
topocentric frame at reference_lla.json (reconstruction.topocentric.json),
with the camera looking along +z, x to
the right and y down, the convention the scripts use. Its Brown model
normalises focal length and principal point by the larger image side and orders
the decentring terms as OpenCV does, which is the order written here.

For the convergent block the surveyed targets, which stayed out of the
adjustment, are triangulated from their measured image positions with the new
poses and calibration and compared with their surveyed coordinates.

    python odm_to_exports.py D1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import Resampling, calculate_default_transform, reproject

ROOT = Path(__file__).resolve().parents[2]           # the repository
PROJECT = ROOT
# the OpenDroneMap projects, full where they were computed or the reduced
# copies kept in odm_runs/
WORK = Path(os.environ.get("ODM_PROJECTS", ROOT / "odm_runs"))
BLOCKS = {
    "D1": {"dataset": "D1_154_building", "prefix": "D1"},
    "D2": {"dataset": "D2_111_nadir", "prefix": "D2"},
}
REFERENCE_CRS = "EPSG:5256"
WGS84 = "EPSG:4326"
ECEF = "EPSG:4978"
SENSOR_LABEL = "FC6310R (8.8mm)"
PIXEL_MM = 0.002412280701754386
# a target whose rays meet at a smaller angle than this cannot be located along
# them, so it is reported but kept out of the check
MIN_INTERSECTION_DEG = 10.0


def rotation_from_axis_angle(vector) -> np.ndarray:
    vector = np.asarray(vector, float)
    angle = np.linalg.norm(vector)
    if angle < 1e-15:
        return np.eye(3)
    k = vector / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def enu_basis(lat: float, lon: float) -> np.ndarray:
    phi, lam = np.radians(lat), np.radians(lon)
    return np.array([
        [-np.sin(lam), np.cos(lam), 0.0],
        [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
    ])


def load_solution(project: Path):
    # reconstruction.json is rewritten by OpenDroneMap in an offset UTM grid;
    # the topocentric copy keeps OpenSfM's east-north-up frame at reference_lla
    path = project / "opensfm" / "reconstruction.topocentric.json"
    if path.exists():
        reconstructions = json.loads(path.read_text(encoding="utf-8"))
    else:
        with gzip.open(path.with_suffix(".json.gz"), "rt", encoding="utf-8") as handle:
            reconstructions = json.load(handle)
    reconstruction = max(reconstructions, key=lambda r: len(r["shots"]))
    lla = json.loads((project / "opensfm" / "reference_lla.json")
                     .read_text(encoding="utf-8"))
    to_ecef = Transformer.from_crs(WGS84, ECEF, always_xy=True)
    origin = np.array(to_ecef.transform(lla["longitude"], lla["latitude"],
                                        lla["altitude"]))
    basis = enu_basis(lla["latitude"], lla["longitude"])   # rows e, n, u
    return reconstruction, origin, basis, len(reconstructions)


def write_cameras_xml(reconstruction, origin, basis, path: Path) -> dict:
    shots = {}
    document = ET.Element("document", version="2.0.0")
    chunk = ET.SubElement(document, "chunk", label="OpenDroneMap", enabled="true")
    sensors = ET.SubElement(chunk, "sensors")
    sensor = ET.SubElement(sensors, "sensor", id="0", label=SENSOR_LABEL, type="frame")
    cameras = ET.SubElement(chunk, "cameras")
    for index, (name, shot) in enumerate(sorted(reconstruction["shots"].items())):
        rotation = rotation_from_axis_angle(shot["rotation"])     # world to camera
        translation = np.asarray(shot["translation"], float)
        centre_topo = -rotation.T @ translation
        centre_ecef = origin + basis.T @ centre_topo
        camera_to_ecef = basis.T @ rotation.T
        matrix = np.eye(4)
        matrix[:3, :3] = camera_to_ecef
        matrix[:3, 3] = centre_ecef
        label = Path(name).stem
        element = ET.SubElement(cameras, "camera", id=str(index), sensor_id="0",
                                label=label)
        ET.SubElement(element, "transform").text = " ".join(f"{v:.12g}" for v in matrix.ravel())
        shots[label] = {"centre_ecef": centre_ecef, "camera_to_ecef": camera_to_ecef}
    transform = ET.SubElement(chunk, "transform")
    ET.SubElement(transform, "rotation").text = "1 0 0 0 1 0 0 0 1"
    ET.SubElement(transform, "translation").text = "0 0 0"
    ET.SubElement(transform, "scale").text = "1"
    return document, sensor, shots


def calibration(reconstruction) -> dict:
    camera = max(reconstruction["cameras"].values(),
                 key=lambda c: c.get("width", 0))
    if camera.get("projection_type") != "brown":
        raise SystemExit(f"expected a Brown camera, found {camera.get('projection_type')}")
    width, height = camera["width"], camera["height"]
    side = max(width, height)
    return {
        "width": width, "height": height,
        "f": camera["focal_x"] * side,
        "f_y": camera["focal_y"] * side,
        "cx": camera["c_x"] * side,
        "cy": camera["c_y"] * side,
        "k1": camera["k1"], "k2": camera["k2"], "k3": camera["k3"],
        "p1_opencv": camera["p1"], "p2_opencv": camera["p2"],
    }


def write_calibration(cal: dict, sensor, path: Path) -> None:
    resolution = ET.SubElement(sensor, "resolution", width=str(cal["width"]),
                               height=str(cal["height"]))
    del resolution
    element = ET.SubElement(sensor, "calibration", type="frame", **{"class": "adjusted"})
    ET.SubElement(element, "resolution", width=str(cal["width"]), height=str(cal["height"]))
    for key, tag in (("f", "f"), ("cx", "cx"), ("cy", "cy"), ("k1", "k1"),
                     ("k2", "k2"), ("k3", "k3"), ("p1_opencv", "p1"),
                     ("p2_opencv", "p2")):
        ET.SubElement(element, tag).text = f"{cal[key]:.12g}"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "type", "width", "height", "pixel_width",
                         "pixel_height", "f", "cx", "cy", "k1", "k2", "k3", "p1", "p2"])
        writer.writerow([SENSOR_LABEL, "Frame", cal["width"], cal["height"], PIXEL_MM,
                         PIXEL_MM, cal["f"], cal["cx"], cal["cy"], cal["k1"], cal["k2"],
                         cal["k3"], cal["p1_opencv"], cal["p2_opencv"]])


def write_sparse_cloud(reconstruction, origin, basis, path: Path) -> int:
    topo = np.array([p["coordinates"] for p in reconstruction["points"].values()])
    ecef = origin + topo @ basis
    to_ref = Transformer.from_crs(ECEF, REFERENCE_CRS, always_xy=True)
    x, y, z = to_ref.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    vertices = np.zeros(len(x), dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    vertices["x"], vertices["y"], vertices["z"] = x, y, z
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(x)}\nproperty double x\nproperty double y\n"
              "property double z\nend_header\n").encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)
    return len(x)


def reproject_raster(source: Path, target: Path, resampling) -> dict:
    with rasterio.open(source) as src:
        transform, width, height = calculate_default_transform(
            src.crs, REFERENCE_CRS, src.width, src.height, *src.bounds,
            resolution=src.res)
        profile = src.profile.copy()
        profile.update(crs=REFERENCE_CRS, transform=transform, width=width,
                       height=height, compress="deflate", tiled=True,
                       BIGTIFF="IF_SAFER")
        nodata = src.nodata
        with rasterio.open(target, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(rasterio.band(src, band), rasterio.band(dst, band),
                          src_nodata=nodata, dst_nodata=nodata,
                          resampling=resampling)
        return {"source_crs": str(src.crs), "resolution_m": list(src.res),
                "width": width, "height": height}


def check_targets(dataset: str, shots: dict, cal: dict) -> dict | None:
    """Triangulate the surveyed targets from their measured image positions.

    The image positions are those measure_targets.py found and kept, and the
    surveyed coordinates are read from surveyed_targets.csv.
    """
    exports = ROOT / dataset / "exports"
    measured, surveyed_file = exports / "target_measurements.csv", exports / "surveyed_targets.csv"
    if not (measured.exists() and surveyed_file.exists()):
        return None
    surveyed = {r["label"]: np.array([float(r["east_m"]), float(r["north_m"]),
                                      float(r["height_m"])])
                for r in csv.DictReader(surveyed_file.open(encoding="utf-8"))}
    observations = {}
    for r in csv.DictReader(measured.open(encoding="utf-8")):
        if r["accepted"] == "True" and r["image"] in shots:
            observations.setdefault(r["target"], []).append(
                (r["image"], float(r["x"]), float(r["y"])))
    to_ref = Transformer.from_crs(ECEF, REFERENCE_CRS, always_xy=True)
    to_wgs84 = Transformer.from_crs(ECEF, WGS84, always_xy=True)
    x0 = cal["width"] / 2.0 + cal["cx"]
    y0 = cal["height"] / 2.0 + cal["cy"]

    def undistort(u, v):
        x, y = (u - x0) / cal["f"], (v - y0) / cal["f"]
        a, b = x, y
        p1, p2 = cal["p1_opencv"], cal["p2_opencv"]
        for _ in range(20):
            r2 = a * a + b * b
            radial = 1 + cal["k1"] * r2 + cal["k2"] * r2 ** 2 + cal["k3"] * r2 ** 3
            dx = 2 * p1 * a * b + p2 * (r2 + 2 * a * a)
            dy = p1 * (r2 + 2 * b * b) + 2 * p2 * a * b
            a, b = (x - dx) / radial, (y - dy) / radial
        return np.array([a, b, 1.0])

    results = {}
    for label, seen in sorted(observations.items()):
        if len(seen) < 2 or label not in surveyed:
            continue
        rows, rhs, rays = [], [], []
        for image, u, v in seen:
            ray = shots[image]["camera_to_ecef"] @ undistort(u, v)
            ray /= np.linalg.norm(ray)
            rays.append(ray)
            projector = np.eye(3) - np.outer(ray, ray)
            rows.append(projector)
            rhs.append(projector @ shots[image]["centre_ecef"])
        widest = max(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0)))
                     for i, a in enumerate(rays) for b in rays[i + 1:])
        point = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        difference = np.array(to_ref.transform(*point)) - surveyed[label]
        lon, lat, height = to_wgs84.transform(*point)
        results[label] = {"images": len(seen),
                          "widest_intersection_deg": float(widest),
                          "used": bool(widest >= MIN_INTERSECTION_DEG),
                          "d_east_m": difference[0], "d_north_m": difference[1],
                          "d_height_m": difference[2],
                          "estimated_lon_lat_h": [lon, lat, height],
                          "surveyed_tm36": surveyed[label].tolist()}
    if not results:
        return None
    d = np.array([[r["d_east_m"], r["d_north_m"], r["d_height_m"]]
                  for r in results.values() if r["used"]])
    return {"targets": results,
            "rms_plan_m": float(np.sqrt((d[:, :2] ** 2).sum(1).mean())),
            "rms_height_m": float(np.sqrt((d[:, 2] ** 2).mean()))}

def write_position_tables(dataset: str, shots: dict, targets: dict | None, out: Path) -> None:
    """Camera and target positions, one row per camera and per target."""
    to_wgs84 = Transformer.from_crs(ECEF, WGS84, always_xy=True)
    records = {row["image_name"][:-4]: row for row in csv.DictReader(
        (PROJECT / "results" / "parsed_inputs" / f"{dataset}_camera_records.csv")
        .open(encoding="utf-8"))}
    fields = ["label", "enabled", "aligned", "sensor_label", "photo_path",
              "estimated_x_or_lon", "estimated_y_or_lat", "estimated_z_or_h",
              "reference_x_or_lon", "reference_y_or_lat", "reference_z_or_h"]
    with (out / "cameras_estimated_and_reference.csv").open("w", newline="",
                                                            encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, row in sorted(records.items()):
            entry = {"label": label, "enabled": True, "aligned": label in shots,
                     "sensor_label": SENSOR_LABEL, "photo_path": row["image_name"],
                     "reference_x_or_lon": row["mrk_lon"],
                     "reference_y_or_lat": row["mrk_lat"],
                     "reference_z_or_h": row["mrk_alt"]}
            if label in shots:
                lon, lat, height = to_wgs84.transform(*shots[label]["centre_ecef"])
                entry.update(estimated_x_or_lon=lon, estimated_y_or_lat=lat,
                             estimated_z_or_h=height)
            writer.writerow(entry)
    if not targets:
        return
    fields = ["label", "enabled", "reference_enabled", "projections_count",
              "estimated_x_or_lon", "estimated_y_or_lat", "estimated_z_or_h",
              "reference_x_or_lon", "reference_y_or_lat", "reference_z_or_h"]
    with (out / "markers_estimated_and_reference.csv").open("w", newline="",
                                                            encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, target in sorted(targets["targets"].items()):
            if not target["used"]:
                continue
            lon, lat, height = target["estimated_lon_lat_h"]
            east, north, up = target["surveyed_tm36"]
            writer.writerow({"label": label, "enabled": True, "reference_enabled": False,
                             "projections_count": target["images"],
                             "estimated_x_or_lon": lon, "estimated_y_or_lat": lat,
                             "estimated_z_or_h": height, "reference_x_or_lon": east,
                             "reference_y_or_lat": north, "reference_z_or_h": up})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("block", choices=sorted(BLOCKS))
    parser.add_argument("--project", help="ODM project folder name, default the block")
    args = parser.parse_args()
    spec = BLOCKS[args.block]
    project = WORK / (args.project or args.block)
    out = ROOT / spec["dataset"] / "exports"
    out.mkdir(parents=True, exist_ok=True)

    reconstruction, origin, basis, components = load_solution(project)
    document, sensor, shots = write_cameras_xml(reconstruction, origin, basis,
                                                out / "reference_cameras.xml")
    cal = calibration(reconstruction)
    write_calibration(cal, sensor, out / "sensor_calibration.csv")
    ET.indent(document)
    ET.ElementTree(document).write(out / "reference_cameras.xml", encoding="UTF-8",
                                   xml_declaration=True)
    points = write_sparse_cloud(reconstruction, origin, basis,
                                out / f"{spec['prefix']}_sparse_cloud.ply")
    dsm = reproject_raster(project / "odm_dem" / "dsm.tif",
                           out / f"{spec['prefix']}_DSM.tif", Resampling.bilinear)
    ortho = None
    orthophoto = project / "odm_orthophoto" / "odm_orthophoto.tif"
    if orthophoto.exists():
        ortho = reproject_raster(orthophoto, out / f"{spec['prefix']}_Orthophoto.tif",
                                 Resampling.bilinear)
    targets = check_targets(spec["dataset"], shots, cal)
    write_position_tables(spec["dataset"], shots, targets, out)

    images = len(list((PROJECT / spec["dataset"]).glob(f"{spec['prefix']}_images/*.JPG")))
    summary = {
        "software": "OpenDroneMap",
        "registered_images": len(shots), "images": images,
        "components": components, "tie_points": points,
        "calibration": {k: v for k, v in cal.items()},
        "dsm": dsm, "orthophoto": ortho, "target_check": targets,
    }
    (out / "odm_solution_summary.json").write_text(json.dumps(summary, indent=2,
                                                              default=float),
                                                   encoding="utf-8")
    print(f"{spec['dataset']}: {len(shots)}/{images} images, {components} component(s), "
          f"{points} tie points, f {cal['f']:.1f} px")
    if targets:
        used = [k for k, v in targets["targets"].items() if v["used"]]
        print(f"  targets: plan RMS {targets['rms_plan_m']:.3f} m, "
              f"height RMS {targets['rms_height_m']:.3f} m over {len(used)} of "
              f"{len(targets['targets'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
