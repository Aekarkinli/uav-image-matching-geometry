#!/usr/bin/env python3
"""Figure 6. Where the camera-position error sits in the block.

A single number cannot say whether a block is uniformly slightly wrong or
smoothly deformed across its extent. Each camera is drawn at its own position,
the horizontal part of its residual as an arrow and the vertical part as a
signed colour, after the similarity alignment that fixes datum and scale.

One arrow scale is used for all eight panels, so a quiet panel is quiet because
the residuals are small and not because the arrows were rescaled to fill it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as fs  # noqa: E402

# the classical baseline plus the three learned front ends that span the range
COLUMNS = ["rootsift", "superpoint", "disk", "aliked"]
TILT_THRESHOLD = 5.0
SCALE_BAR = {"D1_154_building": 50.0, "D2_111_nadir": 100.0}
MARGIN = 0.09
TOP_MM = 16.0        # the row of front-end titles
STATS_MM = 20.0      # up to five lines of statistics beneath a row of maps
GAP_MM = 5.0
STRIP_MM = 30.0      # the key shared by every panel

# signed classes for the vertical residual, in centimetres
CLASS_EDGES = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], dtype=float)
KEY_ARROW_M = 0.20        # the reference arrow, a round value
ARROW_FRACTION = 0.18     # the share of the panel width it occupies
NO_ARROW_FRACTION = 0.015
CLIP_FRACTION = 0.30


def residual_fields():
    fields = {}
    for run in fs.load_runs():
        network = run.get("network")
        if not network or run["variant"] or run["front_end"] not in COLUMNS:
            continue
        # one matcher per front end, so the nearest-neighbour variant of the
        # classical front end does not overwrite the panel it is not labelled as
        if run["matcher"] not in ("lightglue", "dense"):
            continue
        reference = (run["resolution"] == fs.REFERENCE_RESOLUTION
                     and run["budget"] == fs.REFERENCE_BUDGET)
        if not (reference or run["matcher"] == "dense"):
            continue
        per_camera = network.get("per_camera") or {}
        if not per_camera:
            continue
        fields[(run["dataset"], run["front_end"])] = {
            "per_camera": per_camera,
            "systematic": 1.0 - (network["residual_detrended"]["rmse"] /
                                 network["similarity_in_sample"]["rmse"]) ** 2,
        }
    return fields


def station_table(dataset: str):
    poses = fs.reference_poses(dataset)
    return {p["camera_label"]: (float(p["east_m"]), float(p["north_m"]),
                                float(p["tilt_from_nadir_deg"])) for p in poses}


def tilted_extent(dataset: str):
    """Outline of the tilted subset, drawn identically in every panel of a row."""
    stations = station_table(dataset)
    tilted = [(e, n) for e, n, t in stations.values() if t >= TILT_THRESHOLD]
    if not tilted or len(tilted) > 20:
        return None
    east = np.array([p[0] for p in tilted])
    north = np.array([p[1] for p in tilted])
    pad = 14.0
    return (east.min() - pad, north.min() - pad,
            np.ptp(east) + 2 * pad, np.ptp(north) + 2 * pad), len(tilted)


def panel(ax, field, stations, window, scale, norm, cmap):
    labels = [label for label in field["per_camera"] if label in stations]
    east = np.array([stations[label][0] for label in labels])
    north = np.array([stations[label][1] for label in labels])
    residual = field["per_camera"]
    delta_e = np.array([residual[label]["residual_east"] for label in labels])
    delta_n = np.array([residual[label]["residual_north"] for label in labels])
    delta_z = np.array([residual[label]["residual_up"] for label in labels])
    horizontal = np.hypot(delta_e, delta_n)

    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.set_aspect("equal")
    span = window[1] - window[0]

    drawn = horizontal >= scale * NO_ARROW_FRACTION
    clipped = horizontal > scale * CLIP_FRACTION
    plain = drawn & ~clipped
    if plain.any():
        ax.quiver(east[plain], north[plain],
                  delta_e[plain] / scale * span, delta_n[plain] / scale * span,
                  angles="xy", scale_units="xy", scale=1.0, width=0.0042,
                  headwidth=3.6, headlength=4.2, headaxislength=3.6,
                  color="#595959", alpha=0.75, zorder=3, clip_on=True)

    # a residual too large to draw is truncated to the key length and given an
    # open head; the count and the largest value are reported under the panel,
    # because labelling each one individually is unreadable where many coincide
    for index in np.flatnonzero(clipped):
        direction = np.array([delta_e[index], delta_n[index]]) / horizontal[index]
        start = np.array([east[index], north[index]])
        tip = start + direction * ARROW_FRACTION * span
        annotation = ax.annotate("", xy=tip, xytext=start,
                                 arrowprops=dict(arrowstyle="-|>",
                                                 facecolor="white",
                                                 edgecolor="#303030",
                                                 linewidth=0.5, shrinkA=0,
                                                 shrinkB=0), zorder=4)
        annotation.arrow_patch.set_clip_path(ax.patch)
        annotation.arrow_patch.set_clip_on(True)

    for sign, marker in ((1, "^"), (-1, "v")):
        member = np.sign(delta_z) == sign
        if not member.any():
            continue
        ax.scatter(east[member], north[member], c=100 * delta_z[member],
                   cmap=cmap, norm=norm, marker=marker, s=7.0,
                   edgecolor="white", linewidth=0.15, zorder=6)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#C3C7CB")
        spine.set_linewidth(0.5)

    lines = [
        f"median {100 * np.median(horizontal):.1f} cm horizontal",
        f"median {100 * np.median(np.abs(delta_z)):.1f} cm vertical",
        f"systematic {100 * field['systematic']:.0f} %",
    ]
    if clipped.any():
        lines.append(f"{int(clipped.sum())} above "
                     f"{100 * scale * CLIP_FRACTION:.0f} cm,")
        lines.append(f"largest {100 * horizontal.max():.0f} cm")
    return lines


def legend_strip(fig, cmap, norm, scale, marker, height_mm):
    def band(mm):
        return mm / height_mm

    bar = fig.add_axes((0.365, band(20.0), 0.270, band(2.4)))
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colours = fig.colorbar(mappable, cax=bar, orientation="horizontal",
                           extend="both", spacing="proportional")
    colours.set_ticks(CLASS_EDGES[::2])
    colours.ax.tick_params(labelsize=8.5, length=2, pad=2)
    colours.outline.set_linewidth(0.4)
    bar.set_title("Vertical residual (cm), upward positive", fontsize=8.5, pad=4)

    key = fig.add_axes((0.240, band(7.6), 0.520, band(7.4)))
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.axis("off")
    key.annotate("", xy=(0.145, 0.76), xytext=(0.0, 0.76),
                 arrowprops=dict(arrowstyle="-|>", color="#595959", linewidth=0.9,
                                 shrinkA=0, shrinkB=0))
    key.text(0.175, 0.76, f"{100 * scale * ARROW_FRACTION:.0f} cm horizontally",
             fontsize=8.5, va="center")
    key.text(0.0, 0.20,
             f"no arrow below {100 * scale * NO_ARROW_FRACTION:.1f} cm, "
             f"open head above {100 * scale * CLIP_FRACTION:.0f} cm",
             fontsize=8.5, color="#666666", va="center")

    handles = [
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="#B2182B",
               markeredgecolor="white", markersize=4, label="camera above reference"),
        Line2D([0], [0], marker="v", linestyle="none", markerfacecolor="#2166AC",
               markeredgecolor="white", markersize=4, label="camera below reference"),
    ]
    if marker is not None:
        handles.append(Line2D([0], [0], marker="s", linestyle="none",
                              markerfacecolor="none", markeredgecolor="#00A0B0",
                              markersize=5,
                              label=f"tilted images (n = {marker[1]}, one azimuth)"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, band(1.4)), fontsize=8.5, columnspacing=1.8,
               handletextpad=0.4, frameon=False)


def main(columns=None, name="fig05_residual_fields") -> int:
    global COLUMNS
    if columns:
        COLUMNS = list(columns)
    fs.apply_style()
    fields = residual_fields()
    if not fields:
        print("  no per-camera residuals available")
        return 1

    scale = KEY_ARROW_M / ARROW_FRACTION
    cmap = plt.get_cmap("RdBu_r", len(CLASS_EDGES) + 1)
    norm = BoundaryNorm(CLASS_EDGES, cmap.N, extend="both")

    panel_width = (fs.FULL_WIDTH_MM * 0.936 - 0.07 * len(COLUMNS) * 40.0) / len(COLUMNS)
    cell = min(panel_width, 62.0)
    rows_mm = 2 * cell + STATS_MM + GAP_MM
    height = TOP_MM + rows_mm + STATS_MM + STRIP_MM
    fig = fs.figure(fs.FULL_WIDTH_MM, height)
    outer = gridspec.GridSpec(2, len(COLUMNS), figure=fig, left=0.052,
                              right=0.988, bottom=(STRIP_MM + STATS_MM) / height,
                              top=1 - TOP_MM / height, wspace=0.07,
                              hspace=(STATS_MM + GAP_MM) / cell)

    row_marker = None
    for row, dataset in enumerate(fs.DATASETS):
        stations = station_table(dataset)
        east = np.array([v[0] for v in stations.values()])
        north = np.array([v[1] for v in stations.values()])
        half = max(np.ptp(east), np.ptp(north)) * (0.5 + MARGIN)
        window = (east.mean() - half, east.mean() + half,
                  north.mean() - half, north.mean() + half)
        marker = tilted_extent(dataset)
        row_marker = marker or row_marker

        for column, front_end in enumerate(COLUMNS):
            ax = fig.add_subplot(outer[row, column])
            entry = fields.get((dataset, front_end))
            if entry is None:
                ax.set_xticks([])
                ax.set_yticks([])
                fs.not_applicable(ax, 0.5, 0.5, size=7.0, transform=ax.transAxes)
                continue
            lines = panel(ax, entry, stations, window, scale, norm, cmap)
            if marker is not None:
                # the extent is in object coordinates and must not be confused
                # with the figure height the layout below is expressed in
                (east, north, span_east, span_north), _ = marker
                ax.add_patch(Rectangle((east, north), span_east, span_north,
                                       facecolor="none", edgecolor="#00A0B0",
                                       linewidth=0.6, zorder=7))
            fs.panel_letter(ax, "abcdefgh"[row * len(COLUMNS) + column], inside=True)

            # the statistics sit under the map, never over it
            box = ax.get_position()
            for position, line in enumerate(lines):
                fig.text(box.x0, box.y0 - (4.0 + 3.6 * position) / height, line,
                         fontsize=8.5, ha="left", va="baseline", color="#444444")

            if column == 0:
                fs.scale_bar(ax, SCALE_BAR[dataset], f"{SCALE_BAR[dataset]:.0f} m")
                fs.row_label(fig, ax, dataset, x_name=0.008, x_descriptor=0.026)
            if row == 0:
                ax.set_title(fs.FRONT_END_LABEL[front_end], fontsize=8.5,
                             fontweight="bold", loc="left", pad=4)

    legend_strip(fig, cmap, norm, scale, row_marker, height)
    fs.save(fig, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
