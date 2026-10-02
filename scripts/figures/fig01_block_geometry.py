#!/usr/bin/env python3
"""Figure 1. Measured geometry of the two image networks.

Where the stations sit and what each image is aimed at, how the optical axes
are distributed in tilt and azimuth, and which combinations of separation and
viewpoint change the scheduled pairs occupy.

The plan views share one 360 m window, so the difference in block extent is
read directly rather than inferred from axis numbers. The occupancy panel keeps
its full domain and leaves unpopulated combinations white, so every later
result can be read against the support it rests on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib import gridspec
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as fs  # noqa: E402

PLAN_WINDOW = 360.0    # m, identical for both rows
TILT_THRESHOLD = 5.0   # deg, above which a station counts as tilted
TILT_LIMIT = 52.0      # deg, outer radius of the viewing-geometry panel
CLASS_COLOURS = ["#004E9E", "#008E73", "#D2691E", "#7A4FA3"]
NADIR_COLOUR = "#8A8F94"

# One grid for both blocks, so the two occupancy panels are directly comparable.
BASELINE_GRID = fs.BASELINE_EDGES + [150.0]
CONVERGENCE_GRID = fs.CONVERGENCE_EDGES + [65.0]
COUNT_MAX = 500.0


def block(dataset: str):
    poses = fs.reference_poses(dataset)
    classes, means = fs.look_direction_classes(dataset, TILT_THRESHOLD)
    aim_east, aim_north = fs.ground_aim_points(dataset)
    labels = [p["camera_label"] for p in poses]
    return {
        "east": np.array([float(p["east_m"]) for p in poses]),
        "north": np.array([float(p["north_m"]) for p in poses]),
        "tilt": np.array([float(p["tilt_from_nadir_deg"]) for p in poses]),
        "azimuth": np.array([float(p["viewing_azimuth_deg"]) for p in poses]),
        "aim_east": aim_east,
        "aim_north": aim_north,
        "labels": labels,
        "classes": classes,
        "means": means,
        "group": np.array([classes.get(label, -1) for label in labels]),
    }


def colour_of(index: int) -> str:
    return NADIR_COLOUR if index < 0 else CLASS_COLOURS[index % len(CLASS_COLOURS)]


# ---------------------------------------------------------------------------
# (a), (d) camera network in plan
# ---------------------------------------------------------------------------
def plan_panel(ax, data, first: bool):
    east, north = data["east"], data["north"]
    centre_e, centre_n = east.mean(), north.mean()
    half = PLAN_WINDOW / 2.0
    ax.set_xlim(centre_e - half, centre_e + half)
    ax.set_ylim(centre_n - half, centre_n + half)
    ax.set_aspect("equal")

    for index in np.unique(data["group"]):
        member = data["group"] == index
        colour = colour_of(int(index))
        segments = np.stack([
            np.stack([east[member], data["aim_east"][member]], axis=1),
            np.stack([north[member], data["aim_north"][member]], axis=1),
        ], axis=2)
        for line in segments:
            ax.plot(line[:, 0], line[:, 1], color=colour, linewidth=0.4,
                    alpha=0.38, solid_capstyle="round", zorder=2)
        ax.scatter(data["aim_east"][member], data["aim_north"][member], s=2.0,
                   color=colour, alpha=0.5, linewidths=0, zorder=3)
        if index < 0:
            ax.scatter(east[member], north[member], s=5.5, facecolor="white",
                       edgecolor=colour, linewidth=0.4, zorder=5)
        else:
            ax.scatter(east[member], north[member], s=5.5, color=colour,
                       linewidths=0, zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#C3C7CB")
        spine.set_linewidth(0.5)

    fs.scale_bar(ax, 100.0, "100 m")
    if first:
        fs.north_arrow(ax)


# ---------------------------------------------------------------------------
# (b), (e) viewing geometry
# ---------------------------------------------------------------------------
def viewing_panel(ax, data):
    """Optical axis of every image, as tilt from nadir against azimuth.

    Nadir images fall at the centre, tilted ones at a radius equal to their
    tilt. The acquisition design is read off directly: four crossing look
    directions at a common tilt, or a single one.
    """
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, TILT_LIMIT)

    theta_ring = np.linspace(0, 2 * np.pi, 181)
    ax.fill_between(theta_ring, 20, TILT_LIMIT, color="#F4F5F6", zorder=0)

    for index in np.unique(data["group"]):
        member = data["group"] == index
        colour = colour_of(int(index))
        ax.scatter(np.radians(data["azimuth"][member]), data["tilt"][member],
                   s=6.0, color=colour, alpha=0.65, linewidths=0, zorder=4)

    for index, bearing in enumerate(data["means"]):
        member = data["group"] == index
        if not member.any():
            continue
        radius = float(np.median(data["tilt"][member]))
        colour = CLASS_COLOURS[index % len(CLASS_COLOURS)]
        ax.annotate(f"{int(member.sum())}",
                    xy=(np.radians(bearing + 21.0), min(radius, TILT_LIMIT - 6)),
                    ha="center", va="center", fontsize=8.5, color=colour,
                    fontweight="bold", zorder=6)
    nadir = int((data["group"] < 0).sum())
    ax.annotate(f"{nadir} nadir", xy=(np.radians(232.0), 11.0), ha="center",
                va="center", fontsize=8.5, color="#555555", zorder=6)

    ax.set_thetagrids([0, 90, 180, 270], labels=["N", "E", "S", "W"], fontsize=8.5)
    ax.set_rgrids([10, 20, 30, 40, 50], labels=["", "20°", "", "40°", ""],
                  fontsize=8.5, color="#555555")
    ax.set_rlabel_position(270)
    ax.grid(True, color="#D5D9DD", linewidth=0.4)
    ax.tick_params(pad=0.5)
    ax.spines["polar"].set_color("#C3C7CB")
    ax.spines["polar"].set_linewidth(0.5)


# ---------------------------------------------------------------------------
# (c), (f) occupancy of the scheduled pairs
# ---------------------------------------------------------------------------
def occupancy_panel(ax, ax_margin, pairs):
    baseline = np.array([float(p["baseline_m"]) for p in pairs])
    convergence = np.array([float(p["convergence_angle_deg"]) for p in pairs])
    counts, _, _ = np.histogram2d(baseline, convergence,
                                  bins=[BASELINE_GRID, CONVERGENCE_GRID])

    columns, rows = len(BASELINE_GRID) - 1, len(CONVERGENCE_GRID) - 1
    norm = LogNorm(vmin=1.0, vmax=COUNT_MAX)
    display = np.ma.masked_where(counts.T == 0, counts.T)
    ax.pcolormesh(np.arange(columns + 1), np.arange(rows + 1), display,
                  cmap="Blues", norm=norm, zorder=2)

    # Every cell is outlined, so an empty cell reads as a count of zero rather
    # than as absent domain.
    for i in range(columns + 1):
        ax.plot([i, i], [0, rows], color="#D5D9DD", linewidth=0.4, zorder=3)
    for j in range(rows + 1):
        ax.plot([0, columns], [j, j], color="#D5D9DD", linewidth=0.4, zorder=3)

    for i in range(columns):
        for j in range(rows):
            value = int(counts[i, j])
            if value == 0:
                continue
            ax.text(i + 0.5, j + 0.5, f"{value}", ha="center", va="center",
                    fontsize=8.5, zorder=4,
                    color="white" if norm(counts[i, j]) > 0.62 else "#2A2A2A")

    ax.set_xlim(0, columns)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(columns + 1))
    ax.set_xticklabels([f"{int(v)}" for v in BASELINE_GRID], fontsize=8.5)
    ax.set_yticks(np.arange(rows + 1))
    ax.set_yticklabels([f"{int(v)}" for v in CONVERGENCE_GRID], fontsize=8.5)
    ax.grid(False)
    ax.tick_params(length=2)

    shares = 100.0 * counts.sum(axis=0) / max(counts.sum(), 1)
    ax_margin.barh(np.arange(rows) + 0.5, shares, height=0.70,
                   color=fs.SUPPORT_BAR, edgecolor="none", zorder=3)
    for index, share in enumerate(shares):
        if share <= 0:
            continue
        text = "<1" if share < 0.5 else f"{share:.0f}"
        ax_margin.text(min(share + 6, 58), index + 0.5, text, ha="left",
                       va="center", fontsize=8.5, color="#444444")
    ax_margin.set_ylim(0, rows)
    ax_margin.set_xlim(0, 100)
    ax_margin.set_yticks([])
    ax_margin.set_xticks([])
    ax_margin.grid(False)
    for spine in ax_margin.spines.values():
        spine.set_visible(False)


def main() -> int:
    fs.apply_style()

    fig = fs.figure(fs.FULL_WIDTH_MM, 104.0)
    outer = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[0.90, 0.74, 1.30],
                              left=0.050, right=0.988, bottom=0.120, top=0.900,
                              wspace=0.40, hspace=0.34)

    # the third header shares a baseline with the key of the marginal bars, so
    # it is kept short enough to leave a gap between them
    headers = ["Camera network in plan", "Viewing geometry",
               "Support of scheduled pairs"]
    column_axes = []

    for row, dataset in enumerate(fs.DATASETS):
        data = block(dataset)
        pairs = fs.pair_geometry(dataset)

        ax_plan = fig.add_subplot(outer[row, 0])
        plan_panel(ax_plan, data, first=(row == 0))
        fs.panel_letter(ax_plan, "ad"[row], inside=True)
        fs.row_label(fig, ax_plan, dataset, descriptor=False)

        ax_view = fig.add_subplot(outer[row, 1], projection="polar")
        viewing_panel(ax_view, data)
        ax_view.text(-0.16, 1.00, f"({'be'[row]})", transform=ax_view.transAxes,
                     fontsize=8.5, family="serif", fontweight="bold", va="top",
                     ha="left")

        cell = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[row, 2], width_ratios=[1.0, 0.16], wspace=0.05)
        ax_matrix = fig.add_subplot(cell[0, 0])
        ax_margin = fig.add_subplot(cell[0, 1])
        occupancy_panel(ax_matrix, ax_margin, pairs)
        fs.panel_letter(ax_matrix, "cf"[row], inside=True, corner="upper right",
                        suffix=f"{len(pairs)} pairs")
        ax_matrix.set_ylabel("Convergence angle (°)", labelpad=2)
        if row == 0:
            ax_margin.text(0.5, 1.015, "% of\npairs", transform=ax_margin.transAxes,
                           ha="center", va="bottom", fontsize=8.5, color="#777777",
                           linespacing=1.15)
        if row == 1:
            ax_matrix.set_xlabel("Baseline (m)")

        if row == 0:
            column_axes.extend([ax_plan, ax_view, ax_matrix])

    # Column headers are placed on the figure, so the three sit on one baseline
    # regardless of how tall each axis is.
    for axis, header in zip(column_axes, headers):
        box = axis.get_position()
        fig.text(box.x0, 0.955, header, fontsize=8.5, fontweight="bold",
                 ha="left", va="baseline")

    fs.save(fig, "fig01_block_geometry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
