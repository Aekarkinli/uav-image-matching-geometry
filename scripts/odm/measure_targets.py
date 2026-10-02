#!/usr/bin/env python3
"""Measure the surveyed targets of the convergent block in the imagery.

Each target is a two-by-two chequer whose centre is the point where the four
squares meet. Its surveyed position is projected into every image with the
poses and calibration of the reference solution to decide in which images it
appears and where to look; the measurement itself comes from the image alone.
In a window around the projected position the centre is located as the saddle
point of the blurred intensity, where the determinant of the Hessian is most
negative, and refined to sub-pixel precision with OpenCV's corner refinement.

A crop of every measurement, with the measured centre marked, is written to a
contact sheet for visual confirmation; measurements rejected there are listed
in REJECTED and left out. The accepted measurements are written to
exports/target_measurements.csv, from which odm_to_exports.py triangulates the
targets.

    python measure_targets.py            measure and write the contact sheet
"""

from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
DATASET = "D1_154_building"
EXPORTS = ROOT / DATASET / "exports"
IMAGES = ROOT / DATASET / "D1_images"
SHEET = ROOT / "results" / "analysis" / "target_contact_sheet.jpg"
OUT = EXPORTS / "target_measurements.csv"

SEARCH_PX = 45          # half-width of the search window around the projection
EDGE_PX = 60            # projections closer than this to the image edge are skipped
CROP_PX = 28            # half-width of the crop shown on the contact sheet
OFFSET_PX = 8.0        # a centre found farther than this from the projection is not used
RESIDUAL_PX = 1.5      # largest reprojection residual a kept measurement may carry
REJECTED: set[tuple[str, str]] = set()   # (target, image) pairs rejected on the sheet


def cameras():
    root = ET.parse(EXPORTS / "reference_cameras.xml").getroot()
    out = {}
    for element in root.iter("camera"):
        node = element.find("transform")
        if node is None:
            continue
        matrix = np.array([float(v) for v in node.text.split()]).reshape(4, 4)
        out[element.get("label")] = (matrix[:3, 3], matrix[:3, :3])   # centre, camera to ECEF
    return out


def calibration():
    rows = list(csv.DictReader((EXPORTS / "sensor_calibration.csv").open(encoding="utf-8")))
    row = max(rows, key=lambda r: float(r["f"]))
    width, height = float(row["width"]), float(row["height"])
    return {"f": float(row["f"]), "x0": width / 2 + float(row["cx"]),
            "y0": height / 2 + float(row["cy"]), "width": width, "height": height,
            "k1": float(row["k1"]), "k2": float(row["k2"]), "k3": float(row["k3"]),
            "p1": float(row["p1"]), "p2": float(row["p2"])}


def targets():
    to_ecef = Transformer.from_crs("EPSG:5256", "EPSG:4978", always_xy=True)
    out = {}
    with (EXPORTS / "surveyed_targets.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            east, north, up = (float(row[k]) for k in ("east_m", "north_m", "height_m"))
            out[row["label"]] = np.array(to_ecef.transform(east, north, up))
    return out


def project(point, centre, camera_to_ecef, cal):
    local = camera_to_ecef.T @ (point - centre)
    if local[2] <= 0:
        return None
    x, y = local[0] / local[2], local[1] / local[2]
    r2 = x * x + y * y
    radial = 1 + cal["k1"] * r2 + cal["k2"] * r2 ** 2 + cal["k3"] * r2 ** 3
    xd = x * radial + 2 * cal["p1"] * x * y + cal["p2"] * (r2 + 2 * x * x)
    yd = y * radial + cal["p1"] * (r2 + 2 * y * y) + 2 * cal["p2"] * x * y
    return np.array([cal["x0"] + cal["f"] * xd, cal["y0"] + cal["f"] * yd])


def undistort(u, v, cal):
    x, y = (u - cal["x0"]) / cal["f"], (v - cal["y0"]) / cal["f"]
    a, b = x, y
    for _ in range(20):
        r2 = a * a + b * b
        radial = 1 + cal["k1"] * r2 + cal["k2"] * r2 ** 2 + cal["k3"] * r2 ** 3
        dx = 2 * cal["p1"] * a * b + cal["p2"] * (r2 + 2 * a * a)
        dy = cal["p1"] * (r2 + 2 * b * b) + 2 * cal["p2"] * a * b
        a, b = (x - dx) / radial, (y - dy) / radial
    return np.array([a, b, 1.0])


def triangulate(observations, cams, cal):
    """Least-squares intersection of the rays, dropping the worst until all fit."""
    kept = list(observations)
    while len(kept) >= 2:
        rows, rhs = [], []
        for row in kept:
            centre, rotation = cams[row["image"]]
            ray = rotation @ undistort(row["x"], row["y"], cal)
            ray /= np.linalg.norm(ray)
            projector = np.eye(3) - np.outer(ray, ray)
            rows.append(projector)
            rhs.append(projector @ centre)
        point = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        residuals = [np.linalg.norm(project(point, *cams[row["image"]], cal)
                                    - np.array([row["x"], row["y"]])) for row in kept]
        worst = int(np.argmax(residuals))
        if residuals[worst] <= RESIDUAL_PX:
            return point, kept, residuals
        kept.pop(worst)
    return None, [], []


def saddle(gray: np.ndarray, guess: np.ndarray):
    """The chequer centre near guess, or None when no clear saddle is found."""
    x0, y0 = int(round(guess[0])) - SEARCH_PX, int(round(guess[1])) - SEARCH_PX
    window = gray[y0:y0 + 2 * SEARCH_PX + 1, x0:x0 + 2 * SEARCH_PX + 1].astype(np.float32)
    if window.shape != (2 * SEARCH_PX + 1, 2 * SEARCH_PX + 1):
        return None, 0.0
    blurred = cv2.GaussianBlur(window, (0, 0), 2.0)
    dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=5)
    dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=5)
    dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=5)
    determinant = dxx * dyy - dxy * dxy
    row, column = np.unravel_index(np.argmin(determinant), determinant.shape)
    strength = float(-determinant[row, column] / (np.median(np.abs(determinant)) + 1e-6))
    corner = np.array([[[x0 + column, y0 + row]]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), criteria)
    return corner[0, 0].astype(float), strength


def main() -> int:
    cams, cal, marks = cameras(), calibration(), targets()
    rows, crops = [], []
    for label, point in sorted(marks.items()):
        for image_label, (centre, rotation) in sorted(cams.items()):
            guess = project(point, centre, rotation, cal)
            if guess is None:
                continue
            if not (EDGE_PX < guess[0] < cal["width"] - EDGE_PX
                    and EDGE_PX < guess[1] < cal["height"] - EDGE_PX):
                continue
            gray = cv2.imread(str(IMAGES / f"{image_label}.JPG"), cv2.IMREAD_GRAYSCALE)
            found, strength = saddle(gray, guess)
            if found is None:
                continue
            offset = float(np.linalg.norm(found - guess))
            index = len(crops) + 1
            accepted = (label, image_label) not in REJECTED
            rows.append({"index": index, "target": label, "image": image_label,
                         "x": round(found[0], 3), "y": round(found[1], 3),
                         "projected_x": round(guess[0], 2), "projected_y": round(guess[1], 2),
                         "offset_px": round(offset, 2), "saddle_strength": round(strength, 1),
                         "accepted": accepted})
            cx, cy = int(round(found[0])), int(round(found[1]))
            crop = Image.open(IMAGES / f"{image_label}.JPG").crop(
                (cx - CROP_PX, cy - CROP_PX, cx + CROP_PX, cy + CROP_PX)).resize((280, 280))
            draw = ImageDraw.Draw(crop)
            scale = 280 / (2 * CROP_PX)
            mx, my = (found[0] - (cx - CROP_PX)) * scale, (found[1] - (cy - CROP_PX)) * scale
            draw.line((mx - 14, my, mx - 5, my), fill=(255, 40, 40), width=2)
            draw.line((mx + 5, my, mx + 14, my), fill=(255, 40, 40), width=2)
            draw.line((mx, my - 14, mx, my - 5), fill=(255, 40, 40), width=2)
            draw.line((mx, my + 5, mx, my + 14), fill=(255, 40, 40), width=2)
            draw.rectangle((0, 0, 280, 22), fill=(0, 0, 0))
            draw.text((5, 4), f"#{index}  target {label}  {image_label[-4:]}  "
                      f"d={offset:.1f}px", fill=(255, 255, 255))
            crops.append(crop)

    # geometric screening: only centres close to the projection enter the
    # intersection, and a measurement the others contradict is dropped
    for label in sorted(marks):
        candidates = [r for r in rows if r["target"] == label and r["accepted"]
                      and r["offset_px"] <= OFFSET_PX]
        _, kept, residuals = triangulate(candidates, cams, cal)
        keep = {r["index"]: e for r, e in zip(kept, residuals)}
        for r in rows:
            if r["target"] == label:
                r["residual_px"] = round(keep[r["index"]], 2) if r["index"] in keep else ""
                r["accepted"] = r["index"] in keep
    crops = [c for c, r in zip(crops, rows) if r["accepted"]]
    shown = [r for r in rows if r["accepted"]]

    columns = 8
    sheet = Image.new("RGB", (columns * 284, math.ceil(len(crops) / columns) * 284), "white")
    for i, crop in enumerate(crops):
        sheet.paste(crop, ((i % columns) * 284, (i // columns) * 284))
    sheet.save(SHEET, quality=90)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for label in sorted(marks):
        mine = [r for r in rows if r["target"] == label]
        used = [r for r in mine if r["accepted"]]
        print(f"target {label}: in {len(mine)} images, {len(used)} kept, residual "
              f"max {max([r['residual_px'] for r in used] or [0]):.2f} px")
    print(f"{len(shown)} of {len(rows)} measurements kept, contact sheet {SHEET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
