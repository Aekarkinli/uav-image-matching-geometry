#!/usr/bin/env python3
"""Figure 3. Pair recovery and pair support resolved by the geometry of the pair.

An aggregate failure rate hides its own cause. Every scheduled pair is placed
in a stratum by the angle between the two viewing directions, and separately by
the distance between the two exposures, and the proportion of pairs retaining
fewer than fifteen verified correspondences is reported for each stratum with an
interval built by resampling images rather than pairs, because a pair is not an
independent trial.

Beside each of those proportions the figure marks the proportion the reference
geometry cannot support, so that the gap between the two is the share of pairs a
configuration accepted on a consistent but unsupported set of matches.

The count of pairs supporting each estimate is carried on its own axis beneath
the estimate, so a rate read from six pairs is never mistaken for a rate read
from six hundred. Strata that the block does not contain are shown as such
rather than interpolated across.

The strata are unequal in extent. A physically proportional axis cannot host
six separated series inside a two-degree stratum at journal column width, so
the strata are drawn at equal width with their true limits on the axis and the
empty ones marked explicitly.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import csv
import json

import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as fs  # noqa: E402

CONFIGS = [
    ("rootsift", "rootsift-lightglue-r2048-k8192"),
    ("doghardnet", "doghardnet-lightglue-r2048-k8192"),
    ("superpoint", "superpoint-lightglue-r2048-k8192"),
    ("aliked", "aliked-lightglue-r2048-k8192"),
    ("disk", "disk-lightglue-r2048-k8192"),
    ("loftr", "loftr-dense-r1024-kall"),
]
SPARSE = 10          # below this many pairs the marker is drawn open
ANALYSIS = fs.PROJECT_ROOT / "results" / "analysis"
MARKER_SIZE = 3.4
STRATA = [
    {"field": "convergence_angle_deg", "edges": fs.CONVERGENCE_EDGES,
     "label": "Angle between the two viewing directions (°)",
     "heading": "By viewpoint change", "unit": "°"},
    {"field": "baseline_m", "edges": fs.BASELINE_EDGES,
     "label": "Distance between the two exposures (m)",
     "heading": "By separation", "unit": " m"},
]


def tick_labels(edges):
    labels = [f"{int(a)}–{int(b)}" for a, b in zip(edges[:-1], edges[1:])]
    return labels + [f">{int(edges[-1])}"]


def support_record(dataset: str) -> dict:
    """Per-pair reference support, written by scripts/22_pair_pose_support.py."""
    path = ANALYSIS / f"pair_pose_support_operating_{dataset}.csv"
    record = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["image1"] + ".JPG", row["image2"] + ".JPG")
            record[row["config"]][key] = row["supportable"] == "True"
    return record


def clustered_intervals(dataset: str) -> dict:
    """Image-resampled intervals, written by scripts/23_pair_support_summary.py."""
    path = ANALYSIS / "pair_support_summary_operating.json"
    return json.loads(path.read_text(encoding="utf-8"))[dataset]


def stratify(dataset: str, field: str, edges):
    """Per stratum, the share not recovered and the share not supported."""
    limits = np.array(edges + [np.inf])
    geometry = {(row["image1"], row["image2"]): float(row[field])
                for row in fs.pair_geometry(dataset)}
    counts = np.zeros(len(limits) - 1, dtype=int)
    for value in geometry.values():
        counts[int(np.digitize(value, limits) - 1)] += 1

    supported = support_record(dataset)
    intervals = clustered_intervals(dataset)
    which = "convergence" if field == "convergence_angle_deg" else "baseline"

    series = {}
    for front_end, config in CONFIGS:
        metrics = fs.pairwise_metrics(dataset, config)
        if not metrics:
            continue
        tally = defaultdict(lambda: [0, 0, 0])
        for row in metrics:
            key = (row["image1"], row["image2"])
            if key not in geometry:
                continue
            index = int(np.digitize(geometry[key], limits) - 1)
            tally[index][1] += 1
            if str(row["failed_pair"]).lower() in ("true", "1"):
                tally[index][0] += 1
            if not supported.get(config, {}).get(key, False):
                tally[index][2] += 1
        rate = np.full(len(counts), np.nan)
        low = np.full(len(counts), np.nan)
        high = np.full(len(counts), np.nan)
        unsupported = np.full(len(counts), np.nan)
        support = np.zeros(len(counts), dtype=int)
        classes = intervals.get(config, {}).get("classes", {}).get(which, [])
        for index in range(len(counts)):
            failed, total, missing = tally.get(index, [0, 0, 0])
            support[index] = total
            if total == 0:
                continue
            rate[index] = 100.0 * failed / total
            unsupported[index] = 100.0 * missing / total
            entry = classes[index] if index < len(classes) else {}
            if entry.get("count"):
                low[index] = entry["failed_low"]
                high[index] = entry["failed_high"]
            else:
                point, bottom, top = fs.wilson(failed, total)
                low[index], high[index] = 100 * bottom, 100 * top
        series[front_end] = {"rate": rate, "low": low, "high": high,
                             "unsupported": unsupported, "support": support}
    return series, counts


def metric_panel(ax, series, counts):
    bins = len(counts)
    ax.set_xlim(0, bins)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, axis="y")
    ax.set_xticks([])

    empty = np.flatnonzero(counts == 0)
    for index in empty:
        ax.axvspan(index, index + 1, color=fs.EMPTY_BIN, zorder=1)
    # neighbouring empty strata carry one label between them, so two labels
    # never meet at a shared edge
    run_start = None
    for index in range(bins + 1):
        inside = index < bins and counts[index] == 0
        if inside and run_start is None:
            run_start = index
        elif not inside and run_start is not None:
            ax.text((run_start + index) / 2.0, 3.0, "no pairs", ha="center",
                    va="bottom", fontsize=8.5, color="#7A7F85", zorder=5)
            run_start = None
    for index in range(1, bins):
        ax.axvline(index, color="#E4E7EA", linewidth=0.4, zorder=1)
    # the final stratum is open above its lower limit
    ax.axvline(bins - 1, color="#B9BEC3", linewidth=0.5,
               linestyle=(0, (2, 2)), zorder=2)

    order = [f for f, _ in CONFIGS if f in series]
    width = 1.0 / max(len(order), 1)
    for position, front_end in enumerate(order):
        entry = series[front_end]
        colour = fs.FRONT_END_COLOR[front_end]
        x = np.arange(bins) + (position + 0.5) * width
        populated = entry["support"] > 0

        # a connector is drawn only where two neighbouring strata both hold
        # pairs, so no line crosses a range the block does not contain
        for start in range(bins - 1):
            if populated[start] and populated[start + 1]:
                ax.plot(x[start:start + 2], entry["rate"][start:start + 2],
                        color=colour, linewidth=0.8, alpha=0.55, zorder=3)

        for index in np.flatnonzero(populated):
            gap = entry["unsupported"][index] - entry["rate"][index]
            if np.isfinite(gap) and gap > 1.5:
                ax.plot([x[index], x[index]],
                        [entry["rate"][index], entry["unsupported"][index]],
                        color=colour, linewidth=2.6, alpha=0.55, zorder=5,
                        solid_capstyle="butt")
                ax.plot([x[index]], [entry["unsupported"][index]], marker="_",
                        markersize=MARKER_SIZE + 3.0, color=colour,
                        markeredgewidth=1.4, linestyle="none", zorder=7)
            ax.plot([x[index], x[index]],
                    [entry["low"][index], entry["high"][index]],
                    color=colour, linewidth=0.7, alpha=0.28, zorder=3,
                    solid_capstyle="butt")
            for end in (entry["low"][index], entry["high"][index]):
                ax.plot([x[index] - width * 0.22, x[index] + width * 0.22],
                        [end, end], color=colour, linewidth=0.7, alpha=0.28,
                        zorder=3)
            dense = entry["support"][index] >= SPARSE
            ax.plot([x[index]], [entry["rate"][index]],
                    marker=fs.FRONT_END_MARKER[front_end],
                    markersize=MARKER_SIZE, color=colour,
                    markerfacecolor=colour if dense else "white",
                    markeredgecolor=colour, markeredgewidth=0.8,
                    linestyle="none", zorder=6)


def support_panel(ax, counts, ceiling):
    bins = len(counts)
    ax.set_xlim(0, bins)
    ax.set_ylim(0, ceiling * 1.32)
    ax.set_xticks(np.arange(bins) + 0.5)
    ax.set_yticks([0, ceiling])
    ax.set_yticklabels(["0", f"{int(ceiling)}"], fontsize=8.5)
    ax.grid(False)
    ax.bar(np.arange(bins) + 0.5, counts, width=0.92, color=fs.SUPPORT_BAR,
           edgecolor="none", zorder=3)
    for index, value in enumerate(counts):
        if value == ceiling:
            continue          # the axis already carries this number
        ax.text(index + 0.5, value + ceiling * 0.07, f"{int(value)}",
                ha="center", va="bottom", fontsize=8.5, color="#444444")
    ax.tick_params(axis="x", length=2)
    ax.tick_params(axis="y", length=2)


def main() -> int:
    fs.apply_style()

    tables = {(dataset, stratum["field"]): stratify(dataset, stratum["field"],
                                                    stratum["edges"])
              for dataset in fs.DATASETS for stratum in STRATA}
    ceilings = {stratum["field"]: max(tables[(d, stratum["field"])][1].max()
                                      for d in fs.DATASETS)
                for stratum in STRATA}

    fig = fs.figure(fs.FULL_WIDTH_MM, 152.0)
    outer = gridspec.GridSpec(2, 2, figure=fig, left=0.140, right=0.988,
                              bottom=0.198, top=0.900, wspace=0.15, hspace=0.62)

    letters = "abcd"
    for row, dataset in enumerate(fs.DATASETS):
        for column, stratum in enumerate(STRATA):
            series, counts = tables[(dataset, stratum["field"])]
            stack = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[row, column], height_ratios=[1.0, 0.24],
                hspace=0.06)
            ax_metric = fig.add_subplot(stack[0, 0])
            ax_support = fig.add_subplot(stack[1, 0])

            metric_panel(ax_metric, series, counts)
            support_panel(ax_support, counts, ceilings[stratum["field"]])
            ax_support.set_xticklabels(tick_labels(stratum["edges"]), fontsize=8.5)
            ax_support.set_xlabel(stratum["label"], labelpad=2)
            if column == 0:
                ax_support.set_ylabel("pairs", fontsize=8.5, labelpad=24)

            fs.panel_letter(ax_metric, letters[row * 2 + column], inside=True)
            ax_metric.set_title(stratum["heading"], fontsize=8.5,
                                fontweight="bold", loc="left", pad=3)
            if column == 0:
                ax_metric.set_ylabel("Pairs with fewer than 15 verified\n"
                                     "correspondences (%)",
                                     fontsize=8.5, labelpad=3)
                fs.row_label(fig, ax_metric, dataset, descriptor=False,
                             x_name=0.012)
            else:
                ax_metric.set_yticklabels([])

            if column == 0:
                header = ax_metric.get_position()

        summary = {
            "D1_154_building": "D1, convergent block: 154 images in four look "
                               "directions, 1937 scheduled pairs",
            "D2_111_nadir": "D2, near-nadir block: 111 images, nine of them "
                            "tilted in a single azimuth, 1217 scheduled pairs",
        }[dataset]
        fig.text(header.x0 - 0.098, header.y1 + 0.056, summary, fontsize=8.5,
                 fontweight="bold", ha="left", va="baseline")

    handles = [Line2D([0], [0], marker=fs.FRONT_END_MARKER[f], color=fs.FRONT_END_COLOR[f],
                      markersize=MARKER_SIZE + 0.6, linewidth=1.0,
                      label=fs.FRONT_END_LABEL[f]) for f, _ in CONFIGS]
    handles.append(Line2D([0], [0], marker="o", markerfacecolor="white",
                          markeredgecolor="#555555", color="#555555",
                          markersize=MARKER_SIZE + 0.6, linewidth=0,
                          label=f"fewer than {SPARSE} pairs"))
    handles.append(Line2D([0], [0], marker="_", color="#555555",
                          markersize=MARKER_SIZE + 2.0, markeredgewidth=1.0,
                          linewidth=0, label="not supported by the reference"))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.054), fontsize=8.5, columnspacing=1.6,
               labelspacing=0.5, handletextpad=0.4, frameon=False)
    fig.text(0.5, 0.032,
             "LightGlue at an extraction resolution of 2048 px and 8192 keypoints per "
             "image; LoFTR is detector-free at 1024 px.",
             ha="center", va="baseline", fontsize=8.5, color="#555555")
    fig.text(0.5, 0.012,
             "The bar above a marker reaches the share the reference geometry does "
             "not support; intervals resample images.",
             ha="center", va="baseline", fontsize=8.5, color="#555555")

    fs.save(fig, "fig03_pair_stratification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
