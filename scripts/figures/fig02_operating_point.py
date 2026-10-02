#!/usr/bin/env python3
"""Figure 2. Choosing the operating point.

Two settings govern everything downstream: how large the images are when they
are matched, and how many keypoints each image is allowed to carry. The first
column shows what a requested keypoint budget actually delivers, which is not
the same for every front end. The remaining columns sweep the working
resolution and report the number of correspondences that survive verification
and the orientation error that follows from them, on scales shared between the
two blocks so that the shapes of the curves can be compared directly.

The extraction resolution used for every later result is marked on the sweeps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as fs  # noqa: E402

RESOLUTIONS = [1024, 1600, 2048, 3200, "native"]
NATIVE_LONG_SIDE = 5472
BUDGETS = [2048, 4096, 8192, 16384]
REFERENCE_INDEX = RESOLUTIONS.index(2048)

# A configuration that could not be run, with the reason, as distinct from one
# that was simply not part of the design.
CANNOT_RUN = {
    ("disk", "native"), ("loftr", 1600), ("loftr", 2048), ("loftr", 3200),
    ("loftr", "native"),
}

INLIER_LIMITS = (120.0, 4000.0)
ROTATION_LIMITS = (5.0, 90.0)

# horizontal slots for the cannot-run marks, so two front ends never coincide
ABSENT_SLOT = {front_end: index for index, front_end in
               enumerate(sorted({f for f, _ in CANNOT_RUN}))}


def census() -> dict:
    path = fs.PROJECT_ROOT / "results" / "tables" / "keypoint_census.json"
    return json.loads(path.read_text(encoding="utf-8"))


def keypoints(store: dict, dataset: str, front_end: str, budget: int):
    key = f"{dataset}/{front_end}-r{fs.REFERENCE_RESOLUTION}-k{budget}"
    entry = store.get(key)
    return entry["median"] if entry else None


def sweeps(dataset: str):
    """Median verified correspondences and relative rotation error by resolution."""
    table = {}
    for run in fs.load_runs():
        if run["dataset"] != dataset or run["variant"]:
            continue
        # the panels are labelled with one matcher, so the nearest-neighbour
        # variant of the classical front end must not share its key
        if run["matcher"] not in ("lightglue", "dense"):
            continue
        if run["budget"] not in (fs.REFERENCE_BUDGET, None):
            continue
        if run["resolution"] not in RESOLUTIONS:
            continue
        matching, network = run.get("matching"), run.get("network")
        table[(run["front_end"], run["resolution"])] = (
            matching.get("median_verified_inliers") if matching else None,
            1000.0 * network["relative_rotation_error_deg"]["median"] if network else None,
        )
    return table


def budget_panel(ax, store, dataset):
    rows = len(fs.FRONT_ENDS)
    grid = np.full((rows, len(BUDGETS)), np.nan)
    text = {}
    for row, front_end in enumerate(fs.FRONT_ENDS):
        for column, budget in enumerate(BUDGETS):
            value = keypoints(store, dataset, front_end, budget)
            if value is None:
                continue
            grid[row, column] = value / budget
            text[(row, column)] = value

    display = np.ma.masked_invalid(grid)
    norm = Normalize(vmin=0.4, vmax=1.0)
    ax.pcolormesh(np.arange(len(BUDGETS) + 1), np.arange(rows + 1), display,
                  cmap="Blues", norm=norm, zorder=2)
    for i in range(len(BUDGETS) + 1):
        ax.plot([i, i], [0, rows], color="#D5D9DD", linewidth=0.4, zorder=3)
    for j in range(rows + 1):
        ax.plot([0, len(BUDGETS)], [j, j], color="#D5D9DD", linewidth=0.4, zorder=3)

    for (row, column), value in text.items():
        shade = norm(grid[row, column])
        ax.text(column + 0.5, row + 0.5, f"{value:,.0f}".replace(",", " "),
                ha="center", va="center", fontsize=8.5, zorder=4,
                color="white" if shade > 0.72 else "#2A2A2A")

    loftr = fs.FRONT_ENDS.index("loftr")
    for column in range(len(BUDGETS)):
        fs.not_applicable(ax, column + 0.5, loftr + 0.5, size=6.0)

    ax.set_xlim(0, len(BUDGETS))
    ax.set_ylim(rows, 0)
    ax.set_xticks(np.arange(len(BUDGETS)) + 0.5)
    ax.set_xticklabels([f"{b // 1024}k" for b in BUDGETS], fontsize=8.5)
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels([fs.FRONT_END_LABEL[f] for f in fs.FRONT_ENDS], fontsize=8.5)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.set_xlabel("Requested keypoint budget", fontsize=8.5, labelpad=2)


def sweep_panel(ax, table, index, limits, formatter, ticks):
    ax.set_yscale("log")
    ax.set_ylim(*limits)
    ax.set_xlim(-0.18, len(RESOLUTIONS) - 0.82)
    ax.grid(True, axis="y")
    ax.yaxis.set_major_formatter(FuncFormatter(formatter))

    ax.axvline(REFERENCE_INDEX, color="#333333", linewidth=0.7,
               linestyle=(0, (1.5, 1.8)), zorder=1)

    for front_end in fs.FRONT_ENDS:
        colour = fs.FRONT_END_COLOR[front_end]
        x, y = [], []
        for position, resolution in enumerate(RESOLUTIONS):
            entry = table.get((front_end, resolution))
            value = entry[index] if entry else None
            if value is None or not np.isfinite(value):
                if (front_end, resolution) in CANNOT_RUN:
                    # drawn beneath the axis, where it cannot be read as a value
                    offset = 0.17 * (ABSENT_SLOT[front_end] - 0.5)
                    fs.not_applicable(ax, position + offset, -0.215,
                                      colour=colour, size=5.4,
                                      transform=ax.get_xaxis_transform())
                continue
            x.append(position)
            y.append(value)
        if len(x) > 1:
            ax.plot(x, y, color=colour, linewidth=1.1, zorder=4)
        ax.plot(x, y, linestyle="none", marker=fs.FRONT_END_MARKER[front_end],
                markersize=3.4, color=colour, markeredgecolor="white",
                markeredgewidth=0.4, zorder=5)

    ax.set_xticks(range(len(RESOLUTIONS)))
    labels = [str(r) if r != "native" else str(NATIVE_LONG_SIDE)
              for r in RESOLUTIONS]
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks(ticks)
    ax.set_yticks([], minor=True)
    ax.tick_params(axis="y", labelsize=8.5)


def main() -> int:
    fs.apply_style()
    store = census()

    fig = fs.figure(fs.FULL_WIDTH_MM, 150.0)
    outer = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.10, 1.0, 1.0],
                              left=0.158, right=0.966, bottom=0.250, top=0.888,
                              wspace=0.44, hspace=0.40)

    headings = [
        ("(a, d)  Delivered keypoints", "2048 px, LightGlue"),
        ("(b, e)  Correspondences", "budget 8192"),
        ("(c, f)  Rotation error", "budget 8192"),
    ]

    for row, dataset in enumerate(fs.DATASETS):
        table = sweeps(dataset)

        ax_budget = fig.add_subplot(outer[row, 0])
        budget_panel(ax_budget, store, dataset)
        fs.row_label(fig, ax_budget, dataset, descriptor=False, x_name=0.013)

        ax_inliers = fig.add_subplot(outer[row, 1])
        sweep_panel(ax_inliers, table, 0, INLIER_LIMITS,
                    lambda v, _: f"{v:g}", [200, 500, 1000, 2000])

        ax_rotation = fig.add_subplot(outer[row, 2])
        sweep_panel(ax_rotation, table, 1, ROTATION_LIMITS,
                    lambda v, _: f"{v:g}", [5, 10, 20, 50])

        if row == 0:
            for axis, (heading, fixed) in zip([ax_budget, ax_inliers, ax_rotation],
                                              headings):
                axis.set_title(heading, fontsize=8.5, fontweight="bold",
                               loc="left", pad=16)
                axis.text(0.0, 1.015, fixed, transform=axis.transAxes, ha="left",
                          va="bottom", fontsize=8.5, color="#666666")
            ax_inliers.text(REFERENCE_INDEX + 0.12, INLIER_LIMITS[1] * 0.90,
                            "operating point\nused hereafter", fontsize=8.5,
                            color="#333333", ha="left", va="top", linespacing=1.2)

        if row == 1:
            for axis in (ax_inliers, ax_rotation):
                axis.set_xlabel("Extraction resolution (px)", fontsize=8.5,
                                labelpad=17)
        ax_inliers.set_ylabel("Median correspondences", fontsize=8.5, labelpad=2)
        ax_rotation.set_ylabel("Median error (mdeg)", fontsize=8.5, labelpad=2)

    handles = [Line2D([0], [0], marker=fs.FRONT_END_MARKER[f],
                      color=fs.FRONT_END_COLOR[f], markersize=3.6, linewidth=1.1,
                      markeredgecolor="white", markeredgewidth=0.4,
                      label=fs.FRONT_END_LABEL[f]) for f in fs.FRONT_ENDS]
    handles.append(Line2D([0], [0], marker="o", markerfacecolor="none",
                          markeredgecolor="#666666", color="none", markersize=4.0,
                          label="could not be run"))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.062), fontsize=8.5, columnspacing=1.6,
               labelspacing=0.5, handletextpad=0.4, frameon=False)
    fig.text(0.5, 0.030,
             "Extraction resolutions are equally spaced and 5472 px is the native "
             "long side.",
             ha="center", va="baseline", fontsize=8.5, color="#555555")
    fig.text(0.5, 0.006,
             "LoFTR takes no keypoint budget; DoG + HardNet was run at two "
             "resolutions.",
             ha="center", va="baseline", fontsize=8.5, color="#555555")

    fs.save(fig, "fig02_operating_point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
