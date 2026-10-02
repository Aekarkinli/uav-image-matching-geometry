#!/usr/bin/env python3
"""Graphical abstract.

The study read from left to right in four steps, each drawn from the records.

1. The convergent block as flown: the orthophoto of the reference solution
   draped on its surface model, with every exposure station at its adjusted
   position and a short stroke along its optical axis. The two stations of the
   example pair are drawn in the colour the example uses.
2. The protocol: six methods of establishing correspondence enter one candidate
   pair list, one geometric verification and one bundle adjustment, and every
   run is compared with the reference solution.
3. One real pair, 41 degrees apart in viewing direction: the correspondences
   RootSIFT keeps and the ten SuperPoint keeps, below the fifteen the pipeline
   needs, and beneath them the verified and reference-consistent
   correspondences of all six methods on the same pair.
4. What the whole block shows: the viewing-direction difference at which each
   method loses half of its pairs, and how widely the camera network and the
   local component of the delivered surface spread across the detector-based
   configurations.

The two images of step 3 and their correspondences are kept at display size
beside the analysis records, so the figure can be redrawn without the imagery or
the match stores. The orthophoto and the surface model of step 1 are read from
the reference solution when it is present and cached the same way.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as fs  # noqa: E402
from paths import FEATURES_DIR, MATCHES_DIR  # noqa: E402

DATASET = "D1_154_building"
PAIR = ("107_0005_0055.JPG", "107_0005_0148.JPG")
BLOCK_DIR = fs.PROJECT_ROOT / DATASET
IMAGES = BLOCK_DIR / "D1_images"
EXPORTS = BLOCK_DIR / "exports"
CACHE = fs.PROJECT_ROOT / "results" / "analysis" / "graphical_abstract_pair"
NATIVE_WIDTH = 5472
DISPLAY_WIDTH = 900
TERRAIN_CELLS = 170

WIDTH_MM = 300.0
HEIGHT_MM = 130.0
TYPE = 8.5
HEAD = 10.5

HIGHLIGHT = "#FFC53D"        # the example pair, wherever it appears
INK = "#1F2328"
SOFT = "#5C636B"
CARD = "#F4F5F7"
STATION = "#2F3A45"

SHOWN_LINES = 60
AXIS_MAX = 60.0
CONSISTENT_LIMIT = 35.0
EXAMPLES = [
    ("rootsift", "rootsift-r2048-k8192", "rootsift-lightglue-r2048-k8192"),
    ("superpoint", "superpoint-r2048-k8192", "superpoint-lightglue-r2048-k8192"),
]
ROWS = [
    ("rootsift", "rootsift-lightglue-r2048-k8192"),
    ("doghardnet", "doghardnet-lightglue-r2048-k8192"),
    ("loftr", "loftr-dense-r1024-kall"),
    ("superpoint", "superpoint-lightglue-r2048-k8192"),
    ("aliked", "aliked-lightglue-r2048-k8192"),
    ("disk", "disk-lightglue-r2048-k8192"),
]
SHORT_LABEL = {"doghardnet": "DoG+HardNet"}


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
def analysis(name: str) -> Path:
    return fs.PROJECT_ROOT / "results" / "analysis" / name


def pair_record() -> dict:
    wanted = tuple(name[:-4] for name in PAIR)
    path = analysis(f"pair_pose_support_operating_{DATASET}.csv")
    return {row["config"]: row
            for row in csv.DictReader(path.open(encoding="utf-8"))
            if (row["image1"], row["image2"]) == wanted}


def half_loss_angles() -> dict:
    record = json.loads(analysis("pair_support_summary_operating.json")
                        .read_text(encoding="utf-8"))[DATASET]
    return {front_end: (record[config]["half_angle_accepted_deg"]
                        if record[config]["half_angle_reached"] else None)
            for front_end, config in ROWS}


def correspondences(features: str, matches: str):
    """Verified correspondences of the example pair, in native pixels."""
    cache = CACHE / "correspondences.json"
    stored = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    feature_store = FEATURES_DIR / DATASET / f"{features}.h5"
    match_store = MATCHES_DIR / DATASET / f"{matches}.h5"
    if feature_store.exists() and match_store.exists():
        first, second = PAIR
        with h5py.File(feature_store, "r") as store:
            points = {name: store[name]["keypoints"][()] for name in PAIR}
        with h5py.File(match_store, "r") as store:
            if first in store and second in store[first]:
                index = store[first][second]["matches0"][()]
                valid = index > -1
                left, right = points[first][valid], points[second][index[valid]]
            else:
                index = store[second][first]["matches0"][()]
                valid = index > -1
                left, right = points[first][index[valid]], points[second][valid]
        stored[matches] = {"first": left.round(2).tolist(),
                           "second": right.round(2).tolist()}
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(stored) + "\n", encoding="utf-8")
    entry = stored[matches]
    return np.array(entry["first"]), np.array(entry["second"])


def load_image(name: str):
    cached = CACHE / name.replace(".JPG", ".jpg")
    source = IMAGES / name
    if source.exists():
        image = Image.open(source).convert("RGB")
        image = image.resize((DISPLAY_WIDTH, round(image.height * DISPLAY_WIDTH
                                                   / image.width)), Image.LANCZOS)
        CACHE.mkdir(parents=True, exist_ok=True)
        image.save(cached, quality=92)
    else:
        image = Image.open(cached).convert("RGB")
    return np.asarray(image), DISPLAY_WIDTH / NATIVE_WIDTH


def terrain():
    """Orthophoto colours on the surface model, on one coarse grid, in metres."""
    cache = CACHE / "terrain.npz"
    ortho, dsm = EXPORTS / "D1_Orthophoto.tif", EXPORTS / "D1_DSM.tif"
    if ortho.exists() and dsm.exists():
        import rasterio
        from rasterio.enums import Resampling

        shape = (TERRAIN_CELLS, TERRAIN_CELLS)
        with rasterio.open(dsm) as source:
            height = source.read(1, out_shape=shape,
                                 resampling=Resampling.average).astype(float)
            height[height < -1000] = np.nan
            bounds = source.bounds
        with rasterio.open(ortho) as source:
            colour = source.read((1, 2, 3), out_shape=(3, *shape),
                                 resampling=Resampling.average)
        east = np.linspace(bounds.left, bounds.right, shape[1])
        north = np.linspace(bounds.top, bounds.bottom, shape[0])
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, height=height, colour=colour, east=east,
                            north=north)
    data = np.load(cache)
    return data["east"], data["north"], data["height"], data["colour"]


def stations():
    """Adjusted exposure stations in the grid of the orthophoto, with axes."""
    cache = CACHE / "stations.json"
    source = EXPORTS / "cameras_estimated_and_reference.csv"
    if source.exists():
        from pyproj import Transformer

        to_grid = Transformer.from_crs("EPSG:4326", "EPSG:5256", always_xy=True)
        axes = {row["camera_label"]: (float(row["axis_e"]), float(row["axis_n"]),
                                      float(row["axis_u"]),
                                      float(row["tilt_from_nadir_deg"]))
                for row in fs.reference_poses(DATASET)}
        records = []
        for row in csv.DictReader(source.open(encoding="utf-8")):
            if not row["estimated_x_or_lon"] or row["label"] not in axes:
                continue
            east, north = to_grid.transform(float(row["estimated_x_or_lon"]),
                                            float(row["estimated_y_or_lat"]))
            records.append([row["label"], east, north,
                            float(row["estimated_z_or_h"]), *axes[row["label"]]])
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(records) + "\n", encoding="utf-8")
    return json.loads(cache.read_text(encoding="utf-8"))


def spreads():
    """Camera-network and local-surface range of the detector-based runs."""
    from fig07_repeatability_propagation import propagation

    out = {}
    for dataset in fs.DATASETS:
        entries = [e for e in propagation(dataset) if e["front_end"] != "loftr"]
        network = [e["network"] for e in entries]
        local = [e["local"] for e in entries]
        out[dataset] = ((min(network), max(network)), (min(local), max(local)))
    return out


# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------
def mm(x, y, w, h):
    return (x / WIDTH_MM, y / HEIGHT_MM, w / WIDTH_MM, h / HEIGHT_MM)


def fx(x):
    return x / WIDTH_MM


def fy(y):
    return y / HEIGHT_MM


def rounded(fig, x, y, w, h, face, edge="none", radius=0.010, width=0.6):
    fig.patches.append(FancyBboxPatch(
        (fx(x), fy(y)), fx(w), fy(h),
        boxstyle=f"round,pad=0,rounding_size={radius}", transform=fig.transFigure,
        facecolor=face, edgecolor=edge, linewidth=width, zorder=-5))


def line(fig, start, end, colour="#B7BDC4", width=0.7, arrow=False):
    style = "-|>,head_length=3,head_width=2.2" if arrow else "-"
    fig.patches.append(FancyArrowPatch(
        (fx(start[0]), fy(start[1])), (fx(end[0]), fy(end[1])),
        transform=fig.transFigure, arrowstyle=style, color=colour,
        linewidth=width, mutation_scale=1, zorder=-4))


def step(fig, x, y, number, title, subtitle):
    fig.text(fx(x), fy(y), str(number), fontsize=HEAD, fontweight="bold",
             color="white", ha="center", va="center",
             bbox=dict(boxstyle="circle,pad=0.28", facecolor=INK,
                       edgecolor="none"))
    fig.text(fx(x + 4.4), fy(y + 0.9), title, fontsize=HEAD, fontweight="bold",
             color=INK, ha="left", va="center")
    fig.text(fx(x + 4.4), fy(y - 3.8), subtitle, fontsize=TYPE, color=SOFT,
             ha="left", va="center")


def chevron(fig, x, y):
    fig.patches.append(FancyArrowPatch(
        (fx(x), fy(y)), (fx(x + 4.0), fy(y)), transform=fig.transFigure,
        arrowstyle="-|>,head_length=6,head_width=4.5", color="#AEB4BB",
        linewidth=1.6, mutation_scale=1))


def plain(ax):
    ax.set_facecolor("none")
    ax.grid(False)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9A9A9A")
    ax.set_yticks([])
    ax.minorticks_off()
    ax.tick_params(axis="x", length=2, pad=1.5, colors=SOFT, labelsize=TYPE)


def name(front_end):
    return SHORT_LABEL.get(front_end, fs.FRONT_END_LABEL[front_end])


# --------------------------------------------------------------------------
# step 1: the block
# --------------------------------------------------------------------------
def block_view(ax) -> None:
    east, north, height, colour = terrain()
    grid_e, grid_n = np.meshgrid(east, north)
    ground = float(np.nanmedian(height))
    # the edge cells of the surface model average in its empty border and drop
    # below the ground, so the outermost three rings are left out
    valid = np.isfinite(height)
    for _ in range(3):
        valid[1:-1, 1:-1] &= (valid[:-2, 1:-1] & valid[2:, 1:-1]
                              & valid[1:-1, :-2] & valid[1:-1, 2:])
    height = np.where(valid, np.maximum(height, np.nanpercentile(height, 2)),
                      np.nan)
    # the surface model reaches beyond the stations, with noisy margins; only
    # the part under the block is drawn, without spikes above the buildings
    records_ = np.array([r[1:3] for r in stations()])
    low_, high_ = records_.min(0), records_.max(0)
    inside = ((grid_e >= low_[0]) & (grid_e <= high_[0])
              & (grid_n >= low_[1]) & (grid_n <= high_[1]))
    spike = np.abs(height - ground) > 14.0
    height = np.where(inside & ~spike, height, np.nan)
    faces = np.moveaxis(colour, 0, -1).astype(float) / 255.0
    ax.plot_surface(grid_e, grid_n, height, facecolors=faces, rstride=1,
                    cstride=1, linewidth=0, antialiased=False, shade=False,
                    zorder=1)

    records = stations()
    chosen = {n[:-4] for n in PAIR}
    for label, e, n, h, ae, an, au, tilt in records:
        if label in chosen:
            continue
        stroke = 11.0
        colour_ = STATION if tilt > 5 else "#98A1AB"
        ax.plot([e, e + ae * stroke], [n, n + an * stroke], [h, h + au * stroke],
                color=colour_, linewidth=0.55, alpha=0.85, zorder=5)
        ax.scatter([e], [n], [h], s=2.4, color=colour_, depthshade=False, zorder=6)
    for label, e, n, h, ae, an, au, _ in records:
        if label not in chosen:
            continue
        reach = (h - ground) / -au
        ax.plot([e, e + ae * reach], [n, n + an * reach], [h, h + au * reach],
                color=HIGHLIGHT, linewidth=1.6, zorder=8)
        ax.scatter([e], [n], [h], s=18, color=HIGHLIGHT, edgecolor=INK,
                   linewidth=0.45, depthshade=False, zorder=9)

    points = np.array([r[1:4] for r in records])
    low, high = points[:, :2].min(0), points[:, :2].max(0)
    centre, half = (low + high) / 2, (high - low).max() / 2 + 4
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(ground - 6, points[:, 2].max() + 6)
    ax.set_box_aspect((1, 1, 0.5), zoom=1.12)
    ax.view_init(elev=24, azim=-58)
    ax.set_axis_off()


# --------------------------------------------------------------------------
# step 2: the protocol
# --------------------------------------------------------------------------
def protocol(fig, x, w, top) -> None:
    chip_w, chip_h, col_gap, row_step = (w - 2.0) / 2, 6.0, 2.0, 8.0
    hub = (x + w / 2, top - 3 * row_step - 9.0)
    for index, (front_end, _) in enumerate(ROWS):
        column, row = index % 2, index // 2
        cx = x + column * (chip_w + col_gap)
        cy = top - row * row_step - chip_h
        line(fig, (cx + chip_w / 2, cy), hub)
        fig.patches.append(FancyBboxPatch(
            (fx(cx), fy(cy)), fx(chip_w), fy(chip_h),
            boxstyle="round,pad=0,rounding_size=0.006", transform=fig.transFigure,
            facecolor=fs.FRONT_END_COLOR[front_end], edgecolor="none", zorder=-3))
        fig.text(fx(cx + chip_w / 2), fy(cy + chip_h / 2), name(front_end),
                 fontsize=TYPE, fontweight="bold", ha="center", va="center",
                 color=INK if front_end == "disk" else "white")
    stages = [("same candidate pairs", "1937 in D1, 1217 in D2"),
              ("same geometric verification", "MAGSAC++, 2 px threshold"),
              ("same bundle adjustment", "COLMAP, one camera model"),
              ("compared with the reference", "camera centres, epipolar\n"
                                              "geometry, delivered surface")]
    box_h, gap = 11.0, 4.6
    y = hub[1] - 3.0
    line(fig, hub, (hub[0], y + 0.2), colour="#9AA1A9", width=0.8, arrow=True)
    for index, (title, detail) in enumerate(stages):
        last = index == len(stages) - 1
        height = box_h + (3.4 if last else 0)
        rounded(fig, x, y - height, w, height, "white" if not last else "#FFF4D6",
                edge="#D4D8DD" if not last else "#F0C45A", radius=0.008)
        fig.text(fx(x + 2.4), fy(y - 3.4), title, fontsize=TYPE,
                 fontweight="bold", color=INK, ha="left", va="center")
        fig.text(fx(x + 2.4), fy(y - 6.2), detail, fontsize=TYPE, color=SOFT,
                 ha="left", va="top", linespacing=1.1)
        if not last:
            line(fig, (x + w / 2, y - height - 0.2),
                 (x + w / 2, y - height - gap + 0.2), colour="#9AA1A9",
                 width=0.8, arrow=True)
        y -= height + gap


# --------------------------------------------------------------------------
# step 3: one pair
# --------------------------------------------------------------------------
def even_subset(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return np.arange(len(points))
    grid = 12
    span = points.max(0) - points.min(0) + 1e-9
    keys = ((points - points.min(0)) / span * (grid - 1)).astype(int)
    cells = {}
    for index in np.random.default_rng(7).permutation(len(points)):
        cells.setdefault(tuple(keys[index]), []).append(index)
    chosen = []
    while len(chosen) < count:
        for bucket in cells.values():
            if bucket and len(chosen) < count:
                chosen.append(bucket.pop())
    return np.array(chosen)


def pair_panel(ax, images, front_end, left, right, count):
    (first, scale), (second, _) = images
    gap = 24
    height, width = first.shape[:2]
    canvas = np.full((height, 2 * width + gap, 3), 255, dtype=np.uint8)
    canvas[:, :width] = first
    canvas[:, width + gap:] = second
    ax.imshow(canvas, interpolation="lanczos")
    ax.set_xlim(0, canvas.shape[1])
    ax.set_ylim(height, 0)
    ax.axis("off")
    shown = even_subset(left, SHOWN_LINES)
    for index in shown:
        x0, y0 = left[index] * scale
        x1, y1 = right[index] * scale
        ax.plot([x0, x1 + width + gap], [y0, y1], color=HIGHLIGHT,
                linewidth=0.55, alpha=0.9, solid_capstyle="round", zorder=3)
    ax.scatter(np.r_[left[shown, 0] * scale, right[shown, 0] * scale + width + gap],
               np.r_[left[shown, 1], right[shown, 1]] * scale, s=2.4,
               color=HIGHLIGHT, edgecolor="none", zorder=4)
    box = dict(edgecolor="none", pad=1.8, alpha=0.95)
    ax.text(18, 20, name(front_end), ha="left", va="top", fontsize=TYPE,
            color="white", fontweight="bold", zorder=6,
            bbox={**box, "facecolor": fs.FRONT_END_COLOR[front_end]})
    ax.text(canvas.shape[1] - 18, 20, count, ha="right", va="top",
            fontsize=TYPE, color=INK, zorder=6, bbox={**box, "facecolor": "white"})


def tally(ax, record) -> None:
    """Verified and reference-consistent correspondences of the pair."""
    base = 0.7
    ax.set_xscale("log")
    ax.set_xlim(base, 6000)
    ax.set_ylim(len(ROWS) - 0.4, -0.6)
    plain(ax)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.axvline(15, color=INK, linewidth=0.7, linestyle=(0, (2.5, 1.8)), zorder=1)
    ax.text(15, -0.95, "15 needed", ha="center", va="bottom", fontsize=TYPE,
            color=INK)
    bar = 0.62
    for row, (front_end, config) in enumerate(ROWS):
        colour = fs.FRONT_END_COLOR[front_end]
        verified = int(record[config]["verified_inliers"])
        consistent = int(record[config]["reference_consistent"])
        ax.add_patch(Rectangle((base, row - bar / 2), verified - base, bar,
                               facecolor=colour, alpha=0.30, edgecolor="none",
                               zorder=2))
        if consistent:
            ax.add_patch(Rectangle((base, row - bar / 2), consistent - base, bar,
                                   facecolor=colour, edgecolor="none", zorder=3))
        ax.text(verified * 1.18, row, f"{verified}", ha="left", va="center",
                fontsize=TYPE, color=INK if verified >= 15 else SOFT, zorder=4,
                bbox=dict(facecolor=CARD, edgecolor="none", pad=0.4))
        ax.text(0.62, row, name(front_end), ha="right", va="center",
                fontsize=TYPE, color=INK, clip_on=False)


# --------------------------------------------------------------------------
# step 4: what the block shows
# --------------------------------------------------------------------------
def holding(ax, angles: dict) -> None:
    ax.set_xlim(0, AXIS_MAX + 5)
    ax.set_ylim(len(ROWS) - 0.4, -0.6)
    plain(ax)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xticklabels(["0°", "20°", "40°", "60°"])
    bar = 0.62
    for row, (front_end, _) in enumerate(ROWS):
        colour = fs.FRONT_END_COLOR[front_end]
        angle = angles[front_end]
        ax.text(-1.8, row, name(front_end), ha="right", va="center",
                fontsize=TYPE, color=INK, clip_on=False)
        if angle is None:
            ax.add_patch(Rectangle((0, row - bar / 2), AXIS_MAX - 1.5, bar,
                                   facecolor=colour, edgecolor="none", zorder=2))
            ax.add_patch(FancyArrowPatch(
                (AXIS_MAX - 2.0, row), (AXIS_MAX + 4.6, row),
                arrowstyle="-|>,head_length=3.2,head_width=3.2", mutation_scale=1,
                color=colour, linewidth=0, zorder=3))
            continue
        ax.add_patch(Rectangle((0, row - bar / 2), angle, bar, facecolor=colour,
                               edgecolor="none", zorder=2))
        ax.text(angle + 1.4, row, f"{angle:.0f}°", ha="left", va="center",
                fontsize=TYPE, color=SOFT, zorder=4)


def spread_panel(ax, values: dict) -> None:
    ax.set_xscale("log")
    ax.set_xlim(10, 300)
    ax.set_ylim(3.55, -0.95)
    plain(ax)
    ax.set_xticks([10, 20, 50, 100, 200])
    ax.set_xticklabels(["10", "20", "50", "100", "200 mm"])
    rows = []
    for dataset, short in (("D1_154_building", "D1"), ("D2_111_nadir", "D2")):
        network, local = values[dataset]
        rows.append((f"{short} camera centres", network, INK))
        rows.append((f"{short} surface, detrended", local, "#8E6BC9"))
    for row, (label, (low, high), colour) in enumerate(rows):
        ax.plot([low, high], [row, row], color=colour, linewidth=3.0,
                solid_capstyle="round", zorder=2)
        ax.scatter([low, high], [row, row], s=12, color=colour, zorder=3)
        ax.text(10.4, row - 0.34, label, ha="left", va="bottom", fontsize=TYPE,
                color=INK)
        ax.text(high * 1.1, row, f"{high / low:.1f}×", ha="left", va="center",
                fontsize=TYPE, color=INK, fontweight="bold")


# --------------------------------------------------------------------------
def main() -> int:
    fs.apply_style()
    fig = fs.figure(WIDTH_MM, HEIGHT_MM)
    fig.patch.set_facecolor("white")

    columns = [(3.0, 84.0), (93.0, 58.0), (157.0, 72.0), (235.0, 62.0)]
    for x, w in columns:
        rounded(fig, x, 3.0, w, HEIGHT_MM - 6.0, CARD, radius=0.012)
    for x, w in columns[:-1]:
        chevron(fig, x + w + 1.0, HEIGHT_MM / 2)

    record = pair_record()
    operating = record["rootsift-lightglue-r2048-k8192"]
    angle = float(operating["convergence_angle_deg"])
    baseline = float(operating["baseline_m"])
    head = HEIGHT_MM - 8.5
    step(fig, 8.0, head, 1, "A real UAV block", "D1, 154 images, convergent")
    step(fig, 98.0, head, 2, "One fixed protocol",
         "only the matching method changes")
    step(fig, 162.0, head, 3, rf"One pair, $\theta$ = {angle:.0f}°",
         f"viewing directions {angle:.0f}° apart, {baseline:.0f} m baseline")
    step(fig, 240.0, head, 4, "Whole blocks", "every candidate pair, every run")

    view = fig.add_axes(mm(1.0, 19.0, 88.0, 98.0), projection="3d",
                        computed_zorder=False)
    view.set_facecolor("none")
    block_view(view)
    fig.text(fx(45.0), fy(7.0), "stations and optical axes, grey where near-nadir,\n"
             "over the orthophoto and surface model;\n"
             "the pair of step 3 in yellow",
             fontsize=TYPE, color=SOFT, ha="center", va="bottom",
             linespacing=1.15)

    protocol(fig, 96.0, 52.0, head - 9.5)

    images = [load_image(n) for n in PAIR]
    panel_w = 66.0
    panel_h = panel_w * images[0][0].shape[0] / (2 * images[0][0].shape[1] + 24)
    for index, (front_end, features, matches) in enumerate(EXAMPLES):
        left, right = correspondences(features, matches)
        kept = int(record[matches]["verified_inliers"])
        if kept != len(left):
            raise SystemExit(f"{matches}: store holds {len(left)}, record {kept}")
        count = f"{kept} matches" if kept >= 15 else f"{kept} matches, pair lost"
        bottom = head - 9.5 - panel_h - index * (panel_h + 3.0)
        pair_panel(fig.add_axes(mm(160.0, bottom, panel_w, panel_h)), images,
                   front_end, left, right, count)
    fig.text(fx(160.0), fy(56.0), "correspondences in this pair",
             fontsize=TYPE, fontweight="bold", color=INK, ha="left", va="center")
    tally(fig.add_axes(mm(181.0, 12.0, 45.0, 32.0)), record)
    for x, alpha, text in ((162.0, 0.30, "verified"),
                           (179.0, 1.0, "consistent with the reference")):
        fig.patches.append(Rectangle((fx(x), fy(5.0)), fx(2.6), fy(2.4),
                                     transform=fig.transFigure,
                                     facecolor=STATION, alpha=alpha,
                                     edgecolor="none"))
        fig.text(fx(x + 3.6), fy(6.2), text, fontsize=TYPE, color=SOFT,
                 ha="left", va="center")

    fig.text(fx(239.0), fy(head - 11.0), r"D1, half of the pairs lost at $\theta$",
             fontsize=TYPE, fontweight="bold", color=INK, ha="left", va="center")
    holding(fig.add_axes(mm(257.0, 71.0, 36.0, 32.0)), half_loss_angles())
    fig.text(fx(239.0), fy(64.0), "arrow: half never lost up to 61°",
             fontsize=TYPE, color=SOFT, ha="left", va="center")
    fig.text(fx(239.0), fy(55.0), "range over detector-based runs",
             fontsize=TYPE, fontweight="bold", color=INK, ha="left", va="center")
    spread_panel(fig.add_axes(mm(239.0, 12.0, 54.0, 38.0)), spreads())

    fs.save(fig, "fig00_graphical_abstract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
