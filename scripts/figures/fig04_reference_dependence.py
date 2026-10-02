#!/usr/bin/env python3
"""Figures 4 and 5. How the ordering of the configurations depends on the reference.

Every configuration is ordered three ways: against the reference solution,
against the reference orientation pair by pair, which needs neither a datum nor
a scale, and against the raw on-board exposure positions. Placing the
alignment-free ordering in the middle lets all three comparisons be read from
one diagram.

Where the network geometry is strong the three orderings agree. In the weak,
almost purely nadir block the ordering taken from the on-board positions leaves
the other two, because those positions carry the antenna offset and the
exposure timing error, so a reconstruction can rank well against them while
being geometrically worse.

Ranks separated by less than the spread the same configuration shows when it is
simply run again are shown as ties, so that a crossing inside a tie band is not
mistaken for a disagreement.

Each block is drawn on its own page. Thirty-six named rows cannot be set at a
readable type size in half a page, and the naming is what lets a reader follow
an individual configuration across the three orderings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as fs  # noqa: E402

# the fourth entry names the repeated-run range that sets the tie band, which
# for the on-board axis is the range measured against the on-board positions
AXES = [
    ("adjusted", "Reference solution", "", "similarity_held_out_rmse_m"),
    ("rotation", "Relative orientation", "(alignment-free)",
     "relative_rotation_error_deg"),
    ("kinematic", "On-board positions", "", "against_kinematic_rmse_m"),
]
HELD_OUT = (fs.PROJECT_ROOT / "results" / "analysis" / "held_out_axes.csv")
POSITIONS = [0.0, 0.5, 1.0]
SHORT_FEATURE = {"rootsift": "RootSIFT", "sift": "SIFT", "doghardnet": "DoG+HardNet",
                 "superpoint": "SuperPoint", "aliked": "ALIKED", "disk": "DISK",
                 "loftr": "LoFTR"}
SHORT_MATCHER = {"lightglue": "LG", "nn_ratio": "NN", "dense": ""}
EMPHASIS_COUNT = 3      # largest movers highlighted in addition to the leader


def describe(run) -> str:
    feature = SHORT_FEATURE[run["front_end"]]
    matcher = SHORT_MATCHER.get(run["matcher"], "")
    resolution = "native" if run["resolution"] == "native" else str(run["resolution"])
    budget = "all" if run["budget"] is None else (
        f"{run['budget'] // 1024}k" if run["budget"] >= 1024 else str(run["budget"]))
    stem = f"{feature} {matcher}".strip()
    return f"{stem} {resolution}/{budget}", f"{feature} {resolution}/{budget}"


def held_out_axes(dataset: str) -> dict:
    """Held-out discrepancy against both references, from 25_onboard_decomposition."""
    import csv

    values = {}
    with HELD_OUT.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != dataset:
                continue
            values[row["config"]] = (
                float(row["held_out_against_reference_m"]),
                float(row["held_out_against_onboard_m"]))
    return values


def collect(dataset: str):
    axes = held_out_axes(dataset)
    entries = []
    for run in fs.load_runs():
        network = run.get("network")
        if not network or run["dataset"] != dataset:
            continue
        # the merging-tolerance run is a sensitivity check, not part of the design
        if run["variant"]:
            continue
        kinematic = network.get("against_kinematic_positions")
        if not kinematic or run["config"] not in axes:
            continue
        reference_held_out, onboard_held_out = axes[run["config"]]
        label, short = describe(run)
        entries.append({
            "label": label,
            "short": short,
            "front_end": run["front_end"],
            "matcher": run["matcher"],
            "adjusted": reference_held_out,
            "rotation": network["relative_rotation_error_deg"]["median"],
            "kinematic": onboard_held_out,
        })
    entries.sort(key=lambda e: e["rotation"])
    for field, _, _, _ in AXES:
        values = np.array([e[field] for e in entries])
        ranks = np.argsort(np.argsort(values)) + 1
        for entry, rank in zip(entries, ranks):
            entry[f"rank_{field}"] = int(rank)
    return entries


def tie_runs(values, tolerance: float):
    """Contiguous runs of ranks whose values are not separable at a tolerance."""
    order = np.argsort(values)
    sorted_values = np.asarray(values)[order]
    runs, start = [], 0
    for index in range(1, len(sorted_values)):
        if sorted_values[index] - sorted_values[index - 1] >= tolerance:
            if index - start >= 2:
                runs.append((start + 1, index))
            start = index
    if len(sorted_values) - start >= 2:
        runs.append((start + 1, len(sorted_values)))
    return runs


def minus(text: str) -> str:
    """Typographic minus, so a negative correlation is not set with a hyphen."""
    return text.replace("-", chr(0x2212))


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float((a * b).mean())


def panel(ax, entries, count):
    ax.set_xlim(-0.62, 1.58)
    ax.set_ylim(count + 0.7, 0.3)
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30, count])
    ax.set_xticks([])
    ax.grid(False)
    for name in ("top", "right", "bottom"):
        ax.spines[name].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")

    for field, _, _, quantity in AXES:
        x = POSITIONS[[a[0] for a in AXES].index(field)]
        ax.plot([x, x], [0.6, count + 0.4], color=fs.RULE, linewidth=0.5, zorder=2)
        tolerance = fs.seed_spread(quantity)
        if np.isfinite(tolerance):
            for first, last in tie_runs([e[field] for e in entries], tolerance):
                ax.add_patch(Rectangle((x - 0.030, first - 0.45), 0.060,
                                       last - first + 0.9, facecolor="#EDEDED",
                                       edgecolor="none", zorder=1))

    shifts = np.array([max(abs(e["rank_rotation"] - e["rank_adjusted"]),
                           abs(e["rank_kinematic"] - e["rank_rotation"]))
                       for e in entries])
    emphasised = set(np.argsort(-shifts)[:EMPHASIS_COUNT].tolist())
    emphasised.add(int(np.argmin([e["rank_rotation"] for e in entries])))

    closing = []
    for index, entry in enumerate(entries):
        colour = fs.FRONT_END_COLOR[entry["front_end"]]
        ranks = [entry[f"rank_{field}"] for field, _, _, _ in AXES]
        strong = index in emphasised
        ax.plot(POSITIONS, ranks, color=colour,
                linewidth=1.4 if strong else 0.5,
                alpha=1.0 if strong else 0.32,
                linestyle=fs.MATCHER_STYLE.get(entry["matcher"], "-"),
                zorder=6 if strong else 3)
        ax.plot(POSITIONS, ranks, linestyle="none",
                marker=fs.FRONT_END_MARKER[entry["front_end"]],
                markersize=3.2 if strong else 2.4, color=colour,
                markeredgecolor="white", markeredgewidth=0.4,
                alpha=1.0 if strong else 0.45, zorder=7 if strong else 4)

        ax.text(-0.050, ranks[0], entry["label"], ha="right", va="center",
                fontsize=8.5, color=colour if strong else "#4A4A4A",
                fontweight="bold" if strong else "normal",
                alpha=1.0 if strong else 0.85, zorder=8)
        if strong:
            closing.append((ranks[2], entry["short"], colour))

    for rank, text, colour in separated(closing, minimum=1.1):
        ax.text(1.050, rank, text, ha="left", va="center", fontsize=8.5,
                color=colour, fontweight="bold", zorder=8)
    return emphasised


def separated(items, minimum: float):
    """Push labels apart where two emphasised lines finish at adjacent ranks."""
    placed, previous = [], None
    for rank, text, colour in sorted(items, key=lambda item: item[0]):
        position = rank if previous is None else max(rank, previous + minimum)
        placed.append((position, text, colour))
        previous = position
    return placed


HEADING = {"D1_154_building": "D1, convergent block",
           "D2_111_nadir": "D2, near-nadir block"}


def draw(dataset: str, name: str) -> None:
    fig = fs.figure(fs.FULL_WIDTH_MM, 176.0)
    outer = gridspec.GridSpec(1, 1, figure=fig, left=0.072, right=0.988,
                              bottom=0.090, top=0.845)

    entries = collect(dataset)
    count = len(entries)
    ax = fig.add_subplot(outer[0, 0])
    panel(ax, entries, count)
    ax.set_ylabel("Rank (1 = lowest error)", fontsize=8.5, labelpad=3)

    for (field, title, subtitle, _), x in zip(AXES, POSITIONS):
        ax.text(x, 1.112, title, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        if subtitle:
            ax.text(x, 1.088, subtitle, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, style="italic",
                    color="#555555")

    threshold = max(1, count // 4)
    for left, right, centre in [("adjusted", "rotation", 0.25),
                                ("rotation", "kinematic", 0.75)]:
        a = [e[f"rank_{left}"] for e in entries]
        b = [e[f"rank_{right}"] for e in entries]
        movers = int(np.sum(np.abs(np.array(a) - np.array(b)) >= threshold))
        ax.text(centre, 1.006, minus(
                f"ρ = {spearman(a, b):+.2f}\n{movers} of {count} shift\n"
                f"{threshold} places or more"),
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8.5, color="#333333", linespacing=1.25)

    ax.text(0.0, 1.168, f"{HEADING[dataset]}, {count} configurations",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            fontweight="bold")

    handles = [Line2D([0], [0], marker=fs.FRONT_END_MARKER[f],
                      color=fs.FRONT_END_COLOR[f], markersize=3.6, linewidth=1.1,
                      markeredgecolor="white", markeredgewidth=0.4,
                      label=fs.FRONT_END_LABEL[f]) for f in fs.FRONT_ENDS]
    handles += [
        Line2D([0], [0], color="#666666", linewidth=1.1, linestyle="-",
               label="LightGlue or detector-free"),
        Line2D([0], [0], color="#666666", linewidth=1.1,
               linestyle=fs.MATCHER_STYLE["nn_ratio"],
               label="Nearest neighbour with ratio test"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#EDEDED",
               markeredgecolor="#CCCCCC", markersize=6,
               label="not separable from its neighbours"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.004), fontsize=8.5, columnspacing=1.8,
               handletextpad=0.5, labelspacing=0.45, frameon=False)

    fs.save(fig, name)


def main() -> int:
    fs.apply_style()
    draw("D1_154_building", "fig04_reference_dependence_d1")
    draw("D2_111_nadir", "fig04_reference_dependence_d2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
