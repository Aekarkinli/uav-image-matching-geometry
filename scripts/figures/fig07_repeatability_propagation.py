#!/usr/bin/env python3
"""Figure 8. What the study can resolve, and what reaches the surface.

Incremental mapping is not deterministic, so a difference between two
configurations only means something if it exceeds the spread the same
configuration shows when it is simply run again. The upper row measures that
spread over repeated runs that differ only in the mapper seed, and the envelope
it establishes is carried into the lower row as a rule.

The lower row follows each configuration from the camera network into the
delivered surface, separating the part of the surface difference that is a
smooth deformation of the block from the part that is local. A horizontal line
means the object-space error matches the camera-network error in magnitude; a
crossing means a change of order.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as fs  # noqa: E402

STAGES = [
    ("network", "Camera\nnetwork"),
    ("systematic", "Surface\nsystematic"),
    ("local", "Surface\nlocal"),
]
POSITIONS = [0.0, 0.5, 1.0]
REPEATED = ["rootsift", "superpoint", "aliked"]
HEIGHT_MM = 178.0


def envelope_mm() -> float:
    return 1000.0 * fs.seed_spread("similarity_held_out_rmse_m")


def repeatability():
    """Deviation of each repeated run from the mean of its own repeats."""
    grouped = {}
    for (dataset, config), entries in fs.load_seed_runs().items():
        parsed = fs.parse_config(config)
        if not parsed:
            continue
        values = np.array([e["similarity_held_out"]["rmse"] for e in entries]) * 1000.0
        grouped[(dataset, parsed["front_end"])] = values - values.mean()
    return grouped


def propagation(dataset: str):
    summary = json.loads(
        (fs.PROJECT_ROOT / "results" / "tables" / "surface_summary.json")
        .read_text(encoding="utf-8"))
    runs = {(r["dataset"], r["config"]): r for r in fs.load_runs()}
    entries = []
    for record in summary.values():
        if record["dataset"] != dataset:
            continue
        run = runs.get((dataset, record["config"]))
        network = run.get("network") if run else None
        if not network or not np.isfinite(record.get("trend_amplitude", np.nan)):
            continue
        entries.append({
            "front_end": record["front_end"],
            "resolution": record["resolution"],
            "network": 1000.0 * network["similarity_held_out"]["rmse"],
            "systematic": 1000.0 * record["trend_amplitude"],
            "local": 1000.0 * record["detrended_nmad"],
        })
    entries.sort(key=lambda e: e["network"])
    return entries


def rank_agreement(first, second):
    first, second = np.asarray(first), np.asarray(second)
    ranks = [np.argsort(np.argsort(v)).astype(float) for v in (first, second)]
    ranks = [(r - r.mean()) / r.std() for r in ranks]
    inverted = sum(1 for i, j in combinations(range(len(first)), 2)
                   if (first[i] - first[j]) * (second[i] - second[j]) < 0)
    total = len(first) * (len(first) - 1) // 2
    return float((ranks[0] * ranks[1]).mean()), inverted, total


def separated(items, minimum: float):
    """Enforce a minimum vertical separation between labels, in log units."""
    placed, previous = [], None
    for position, *rest in sorted(items, key=lambda item: item[0]):
        value = position if previous is None else max(position, previous + minimum)
        placed.append((value, *rest))
        previous = value
    return placed


def repeatability_panel(ax, deviations, dataset, limit, show_names: bool):
    """One block of repeated runs, three configurations deep."""
    rows, labels = [], []
    for position, front_end in enumerate(REPEATED):
        key = (dataset, front_end)
        if key not in deviations:
            continue
        rows.append((float(position), front_end, deviations[key]))
        labels.append(fs.FRONT_END_LABEL[front_end])

    ax.axvspan(-limit / 2, limit / 2, color="#EDEDED", zorder=1)
    for edge in (-limit / 2, limit / 2):
        ax.axvline(edge, color="#9AA0A6", linewidth=0.7, linestyle=(0, (3, 2)),
                   zorder=2)
    ax.axvline(0, color="#8A8F94", linewidth=0.7, zorder=2)

    for y, front_end, values in rows:
        colour = fs.FRONT_END_COLOR[front_end]
        ax.plot([values.min(), values.max()], [y, y], color="#9AA0A6",
                linewidth=0.8, zorder=3)
        ax.plot(values, np.full(values.size, y), linestyle="none", marker="o",
                markersize=3.2, color=colour, markeredgecolor="#2B2B2B",
                markeredgewidth=0.4, zorder=5)
        ax.text(3.02, y, f"{np.ptp(values):.2f}", ha="right", va="center",
                fontsize=8.5, color=colour)

    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels(labels if show_names else [""] * len(rows), fontsize=8.5)
    ax.set_ylim(len(rows) - 0.4, -1.25)
    ax.set_xlim(-1.9, 3.05)
    ax.set_xticks([-1.5, -0.5, 0, 0.5, 1.5])
    ax.set_xticklabels(["\N{MINUS SIGN}1.5", "\N{MINUS SIGN}0.5", "0", "0.5",
                        "1.5"], fontsize=8.5)
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    ax.text(3.02, -0.85, "spread (mm)", fontsize=8.5, color="#777777",
            ha="right", va="center")


def propagation_panel(ax, entries, limit, show_axis: bool):
    ax.set_yscale("log")
    ax.set_ylim(2.0, 6000.0)
    ax.set_xlim(-0.82, 1.30)
    ax.set_xticks([])
    ax.grid(True, axis="y")

    ticks = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks([], minor=True)
    ax.tick_params(axis="y", labelsize=8.5, labelleft=show_axis)

    for x in POSITIONS:
        ax.plot([x, x], [2.0, 6000.0], color="#E5E5E5", linewidth=0.6, zorder=1)
    ax.axhline(limit, color="#333333", linewidth=0.7, linestyle=(0, (3, 2)),
               zorder=2)
    ax.text(-0.80, limit * 1.16, f"run-to-run envelope, {limit:.1f} mm",
            fontsize=8.5, color="#333333", ha="left", va="bottom")

    names, values = [], []
    for entry in entries:
        colour = fs.FRONT_END_COLOR[entry["front_end"]]
        track = [entry[field] for field, _ in STAGES]
        ax.plot(POSITIONS, track, color=colour, linewidth=1.1, zorder=4)
        ax.plot(POSITIONS, track, linestyle="none",
                marker=fs.FRONT_END_MARKER[entry["front_end"]], markersize=3.2,
                color=colour, markeredgecolor="white", markeredgewidth=0.4,
                zorder=5)
        resolution = entry["resolution"]
        suffix = "" if resolution == fs.REFERENCE_RESOLUTION else f" {resolution}"
        names.append((np.log10(track[0]),
                      fs.FRONT_END_LABEL[entry["front_end"]] + suffix, colour))
        for x, value in zip(POSITIONS, track):
            values.append((x, np.log10(value), value, colour))

    for position, text, colour in separated(names, minimum=0.157):
        ax.text(-0.075, 10 ** position, text, ha="right", va="center",
                fontsize=8.5, color=colour, zorder=8)

    for x in POSITIONS:
        column = [(p, v, c) for cx, p, v, c in values if cx == x]
        for position, value, colour in separated(column, minimum=0.157):
            # a label displaced from its own marker can sit on a neighbouring
            # track, so it carries the background with it
            ax.text(x + 0.045, 10 ** position, f"{value:.0f}", ha="left",
                    va="center", fontsize=8.5, color=colour, zorder=7,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                              pad=0.8))

    for x, (field, title) in zip(POSITIONS, STAGES):
        ax.text(x, 1.020, title, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=8.5, linespacing=1.2)

    network = [e["network"] for e in entries]
    lines = [f"n = {len(entries)} configurations"]
    for field, name in [("systematic", "systematic"), ("local", "local")]:
        rho, inverted, total = rank_agreement(network, [e[field] for e in entries])
        lines.append(f"surface {name}: ρ = {rho:+.2f}, "
                     f"{inverted}/{total} inverted")
    return lines


def band(mm: float) -> float:
    return mm / HEIGHT_MM


def main() -> int:
    fs.apply_style()
    limit = envelope_mm()
    deviations = repeatability()

    fig = fs.figure(fs.FULL_WIDTH_MM, HEIGHT_MM)

    upper = gridspec.GridSpec(1, 2, figure=fig, left=0.098, right=0.988,
                              bottom=band(145.0), top=band(172.0), wspace=0.32)
    lower = gridspec.GridSpec(1, 2, figure=fig, left=0.092, right=0.988,
                              bottom=band(26.0), top=band(110.0), wspace=0.22)

    letters = "ab"
    for index, dataset in enumerate(fs.DATASETS):
        ax = fig.add_subplot(upper[0, index])
        repeatability_panel(ax, deviations, dataset, limit, show_names=True)
        heading = {"D1_154_building": "D1, convergent block",
                   "D2_111_nadir": "D2, near-nadir block"}[dataset]
        ax.set_title(f"({letters[index]})  Repeated runs, {heading}", fontsize=8.5,
                     fontweight="bold", loc="left", pad=6)

    fig.text(0.53, band(137.0), "Deviation from the run mean (mm)", fontsize=9,
             ha="center", va="baseline")
    fig.text(0.098, band(128.0),
             "Shaded band: the run-to-run envelope of "
             f"{limit:.1f} mm peak to peak, carried into the lower row.\n"
             "DoG + HardNet, DISK and LoFTR were not repeated.",
             fontsize=8.5, color="#777777", ha="left", va="baseline",
             linespacing=1.35)

    letters = "cd"
    for index, dataset in enumerate(fs.DATASETS):
        ax = fig.add_subplot(lower[0, index])
        entries = propagation(dataset)
        lines = propagation_panel(ax, entries, limit, show_axis=(index == 0))
        box = ax.get_position()
        for position, line in enumerate(lines):
            fig.text(box.x0, band(22.0 - 3.6 * position), line, fontsize=8.5,
                     ha="left", va="baseline", color="#444444")
        heading = {"D1_154_building": "D1, convergent block",
                   "D2_111_nadir": "D2, near-nadir block"}[dataset]
        # the stage headings stand two lines deep above the axis, so the title
        # clears them rather than sharing their band
        ax.set_title(f"({letters[index]})  Into the surface, {heading}",
                     fontsize=8.5, fontweight="bold", loc="left", pad=30)
        if index == 0:
            ax.set_ylabel("Object-space magnitude (mm), log scale", fontsize=8.5,
                          labelpad=4)

    handles = [Line2D([0], [0], marker=fs.FRONT_END_MARKER[f],
                      color=fs.FRONT_END_COLOR[f], markersize=3.6, linewidth=1.1,
                      markeredgecolor="white", markeredgewidth=0.4,
                      label=fs.FRONT_END_LABEL[f]) for f in fs.FRONT_ENDS]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, band(1.8)), fontsize=8.5, columnspacing=1.6,
               handletextpad=0.4, frameon=False)

    fs.save(fig, "fig07_repeatability_propagation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
