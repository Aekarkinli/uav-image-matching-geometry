#!/usr/bin/env python3
"""Figure 7. Signed difference between the delivered surface and the reference.

Each map is the height difference between the surface produced from one
configuration and the reference surface, on a common colour scale. The blocks
are drawn rotated onto their own principal axis and cropped, which removes the
empty corners a north-up frame would leave and roughly doubles the mapped area
in each panel.

The cells are aggregated until the mapped area is essentially complete, so the
colour a reader sees is the difference itself and not the density of the point
cloud showing through. The strip beneath each map is the distribution of the
same values, coloured through the same ramp, with the out-of-range cells
collected into the end bins rather than silently clipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import gridspec
from matplotlib.colors import Normalize
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as fs  # noqa: E402

# the classical baseline and the two learned front ends bracketing the range
COLUMNS = [
    ("rootsift", "rootsift-lightglue-r2048-k8192"),
    ("superpoint", "superpoint-lightglue-r2048-k8192"),
    ("aliked", "aliked-lightglue-r2048-k8192"),
]
AGGREGATE = 4           # block median, chosen so the mapped area is complete
LIMIT = 0.30            # m, the colour range
BIN_WIDTH = 0.01
SCALE_BAR = {"D1_154_building": 25.0, "D2_111_nadir": 50.0}
NO_DATA = fs.NO_DATA


def coarsen(grid: np.ndarray, factor: int) -> np.ndarray:
    """Median of each block of cells, ignoring the cells with no data."""
    height, width = grid.shape
    trimmed = grid[: height // factor * factor, : width // factor * factor]
    blocks = trimmed.reshape(height // factor, factor, width // factor, factor)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(blocks, axis=(1, 3))


def principal_angle(mask: np.ndarray) -> float:
    rows, columns = np.nonzero(mask)
    points = np.stack([columns - columns.mean(), rows - rows.mean()])
    _, vectors = np.linalg.eigh(np.cov(points))
    return float(np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1])))


def north_direction(angle: float):
    """Where north points after the rotation, found by rotating a marker."""
    probe = np.zeros((41, 41))
    probe[:6, 18:23] = 1.0
    turned = ndimage.rotate(probe, angle, reshape=True, order=0, cval=0.0)
    rows, columns = np.nonzero(turned > 0.5)
    centre = np.array(turned.shape) / 2.0
    vector = np.array([columns.mean() - centre[1], rows.mean() - centre[0]])
    return vector / np.linalg.norm(vector)


def prepare(dataset: str):
    """One rotation and one crop window for every panel of a block."""
    paths = {front_end: fs.SURFACE_DIR / dataset / f"{config}_difference.tif"
             for front_end, config in COLUMNS}
    grids, transform, cell = {}, None, None
    for front_end, path in paths.items():
        if not path.exists():
            continue
        with rasterio.open(path) as handle:
            grids[front_end] = handle.read(1, masked=True).filled(np.nan)
            transform, cell = handle.transform, handle.res[0]
    if not grids:
        return None

    coarse = {f: coarsen(g, AGGREGATE) for f, g in grids.items()}
    union = np.zeros(next(iter(coarse.values())).shape, dtype=bool)
    for grid in coarse.values():
        union |= np.isfinite(grid)
    angle = principal_angle(union)

    turned = {f: ndimage.rotate(g, angle, reshape=True, order=0, cval=np.nan,
                                mode="constant") for f, g in coarse.items()}
    footprint = np.zeros(next(iter(turned.values())).shape, dtype=bool)
    for grid in turned.values():
        footprint |= np.isfinite(grid)
    rows, columns = np.nonzero(footprint)
    window = (slice(rows.min(), rows.max() + 1),
              slice(columns.min(), columns.max() + 1))
    return {
        "panels": {f: g[window] for f, g in turned.items()},
        "native": grids,
        "cell": cell * AGGREGATE,
        "north": north_direction(angle),
    }


def map_panel(ax, grid, block, norm, cmap, first: bool, dataset: str):
    ax.set_facecolor(NO_DATA)
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm,
              interpolation="nearest", zorder=2)
    ax.set_axis_off()
    if not first:
        return
    height, width = grid.shape
    length = SCALE_BAR[dataset] / block["cell"]
    x = width * 0.96 - length
    y = height * 0.945
    ax.plot([x, x + length], [y, y], color="#1A1A1A", linewidth=1.4,
            solid_capstyle="butt", zorder=6)
    for position in (x, x + length):
        ax.plot([position, position], [y, y - height * 0.018], color="#1A1A1A",
                linewidth=1.0, zorder=6)
    ax.text(x + length / 2, y - height * 0.028, f"{SCALE_BAR[dataset]:.0f} m",
            ha="center", va="bottom", fontsize=8.5, color="#1A1A1A", zorder=6)

    # the label follows the arrow, because north is no longer up
    centre = np.array([width * 0.90, height * 0.15])
    tip = centre + block["north"] * height * 0.055
    tail = centre - block["north"] * height * 0.055
    ax.annotate("", xy=tip, xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color="#1A1A1A", linewidth=0.8,
                                shrinkA=0, shrinkB=0), zorder=6)
    label = centre + block["north"] * height * 0.105
    ax.text(label[0], label[1], "N", ha="center", va="center", fontsize=8.5,
            color="#1A1A1A", zorder=6,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.8))


def histogram_panel(ax, values, cmap, norm, median, ceiling):
    edges = np.arange(-LIMIT, LIMIT + BIN_WIDTH / 2, BIN_WIDTH)
    inside = values[np.abs(values) <= LIMIT]
    counts, _ = np.histogram(inside, bins=edges)
    fraction = counts / max(values.size, 1)
    below = float(np.mean(values < -LIMIT))
    above = float(np.mean(values > LIMIT))

    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centres, fraction, width=BIN_WIDTH, color=cmap(norm(centres)),
           edgecolor="none", zorder=3)
    pad = BIN_WIDTH * 2.2
    ax.bar([-LIMIT - pad], [below], width=BIN_WIDTH * 2, color=cmap(0.0),
           edgecolor="#404040", linewidth=0.3, zorder=3)
    ax.bar([LIMIT + pad], [above], width=BIN_WIDTH * 2, color=cmap(1.0),
           edgecolor="#404040", linewidth=0.3, zorder=3)
    for edge in (-LIMIT - pad / 2, LIMIT + pad / 2):
        ax.axvline(edge, color="#9AA0A6", linewidth=0.4, zorder=4)

    ax.axvline(0, color="#606060", linewidth=0.5, linestyle=(0, (2, 2)), zorder=5)
    ax.plot([median], [0], marker="^", markersize=3.0, color="#1A1A1A",
            clip_on=False, zorder=6)

    ax.set_xlim(-LIMIT - pad * 2, LIMIT + pad * 2)
    ax.set_ylim(0, ceiling)
    ax.set_yticks([])
    ax.set_xticks([-LIMIT, 0, LIMIT])
    ax.set_xticklabels([f"\N{MINUS SIGN}{LIMIT:.1f}", "0", f"+{LIMIT:.1f}"],
                       fontsize=8.5)
    ax.tick_params(length=1.5, pad=1)
    ax.grid(False)
    for name in ("top", "right", "left"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)


def main(columns=None, name="fig06_surface_maps") -> int:
    global COLUMNS
    if columns:
        COLUMNS = list(columns)
    fs.apply_style()
    summary = json.loads(
        (fs.PROJECT_ROOT / "results" / "tables" / "surface_summary.json")
        .read_text(encoding="utf-8"))

    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(vmin=-LIMIT, vmax=LIMIT)

    blocks = {dataset: prepare(dataset) for dataset in fs.DATASETS}
    if not any(blocks.values()):
        print("  no surface rasters available")
        return 1

    # one vertical scale for every histogram, so the strips are comparable
    ceiling = 0.0
    for dataset, block in blocks.items():
        if not block:
            continue
        for values in block["native"].values():
            finite = values[np.isfinite(values)]
            counts, _ = np.histogram(finite[np.abs(finite) <= LIMIT],
                                     bins=np.arange(-LIMIT, LIMIT + BIN_WIDTH / 2,
                                                    BIN_WIDTH))
            ceiling = max(ceiling, (counts / max(finite.size, 1)).max())
    ceiling *= 1.12

    fig = fs.figure(fs.FULL_WIDTH_MM, 150.0)
    outer = gridspec.GridSpec(2, len(COLUMNS), figure=fig, left=0.038,
                              right=0.988, bottom=0.118, top=0.905,
                              wspace=0.10, hspace=0.30)

    letters = "abcdef"
    for row, dataset in enumerate(fs.DATASETS):
        block = blocks[dataset]
        for column, (front_end, config) in enumerate(COLUMNS):
            stack = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[row, column], height_ratios=[1.0, 0.19],
                hspace=0.05)
            ax_map = fig.add_subplot(stack[0, 0])
            ax_hist = fig.add_subplot(stack[1, 0])

            grid = (block or {}).get("panels", {}).get(front_end)
            if grid is None:
                ax_map.set_axis_off()
                ax_hist.set_axis_off()
                fs.not_applicable(ax_map, 0.5, 0.5, size=7.0,
                                  transform=ax_map.transAxes)
                continue

            map_panel(ax_map, grid, block, norm, cmap, column == 0, dataset)
            native = block["native"][front_end]
            values = native[np.isfinite(native)]
            record = summary.get(f"{dataset}/{config}", {})
            median = record.get("core_median", float(np.median(values)))
            histogram_panel(ax_hist, values, cmap, norm, median, ceiling)

            letter = letters[row * len(COLUMNS) + column]
            ax_map.text(0.02, 0.975, f"({letter})", transform=ax_map.transAxes,
                        fontsize=8.5, family="serif", fontweight="bold", va="top",
                        ha="left", zorder=8,
                        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none",
                                  pad=1.2))
            fs.note(ax_map,
                    f"{values.size:,}".replace(",", " ") + " cells\n"
                    + f"median {100 * median:+.1f}".replace("-", "\N{MINUS SIGN}")
                    + " cm\n"
                    + f"NMAD {100 * record.get('core_nmad', np.nan):.1f} cm",
                    corner="lower left", fontsize=8.5)
            if row == 0:
                ax_map.set_title(fs.FRONT_END_LABEL[front_end], fontsize=8.5,
                                 fontweight="bold", loc="left", pad=4)
            if column == 0:
                fs.row_label(fig, ax_map, dataset, x_name=0.008,
                             x_descriptor=0.024)
                ax_hist.text(-0.02, 0.5, "cells", transform=ax_hist.transAxes,
                             rotation=90, ha="right", va="center", fontsize=8.5,
                             color="#666666")
        if block:
            last = fig.axes[-2].get_position()
            fig.text(last.x1, last.y1 + 0.006,
                     f"display cell {block['cell']:.2f} m",
                     fontsize=8.5, color="#777777", ha="right", va="baseline")

    bar = fig.add_axes((0.300, 0.048, 0.400, 0.017))
    colours = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar,
                           orientation="horizontal", extend="both")
    colours.set_ticks(np.arange(-0.3, 0.31, 0.1))
    colours.ax.set_xticklabels([f"{v:+.1f}".replace("+0.0", "0")
                                                .replace("-", "\N{MINUS SIGN}")
                                for v in np.arange(-0.3, 0.31, 0.1)], fontsize=8.5)
    colours.ax.tick_params(length=2)
    colours.outline.set_linewidth(0.4)
    bar.set_title("Height difference from the reference surface (m)",
                  fontsize=8.5, pad=3)

    swatch = fig.add_axes((0.716, 0.048, 0.020, 0.017))
    swatch.set_xticks([])
    swatch.set_yticks([])
    swatch.set_facecolor(NO_DATA)
    for spine in swatch.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#9AA0A6")
    fig.text(0.742, 0.0565, "no data", fontsize=8.5, va="center", color="#444444")

    fs.save(fig, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
