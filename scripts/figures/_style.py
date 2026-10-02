"""Shared visual language and data access for the figures of the article.

The specification is fixed for the whole set so that a front end carries the
same colour, the same marker and the same position in every legend, and so that
a figure is authored at the size it will be printed at.

Two rules matter more than the rest. Figures are authored in millimetres at the
final printed width, and are never saved with a tight bounding box, because
that silently changes the delivered width and therefore the effective type
size. And every visual channel that carries meaning has a key entry.

Colours are ordered by relative luminance with a wide gap between neighbours,
so the ordering survives conversion to greyscale, and the hues are separable
under the common forms of colour vision deficiency. Marker shape repeats what
colour says, so neither channel is load bearing on its own.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (  # noqa: E402
    EVALUATION_DIR,
    MANIFESTS_DIR,
    PAIRWISE_DIR,
    PROJECT_ROOT,
    REFERENCE_DIR,
    SFM_REPORTS_DIR,
)

OUT = PROJECT_ROOT / "results" / "figures"
SURFACE_DIR = PROJECT_ROOT / "results" / "surface"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# physical size
# ----------------------------------------------------------------------------
MM = 1.0 / 25.4
FULL_WIDTH_MM = 178.0   # the text width of the journal page
COLUMN_WIDTH_MM = 84.0
MAX_HEIGHT_MM = 230.0


def figure(width_mm: float, height_mm: float):
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


# ----------------------------------------------------------------------------
# identities
# ----------------------------------------------------------------------------
DATASETS = ["D1_154_building", "D2_111_nadir"]
SHORT = {"D1_154_building": "D1", "D2_111_nadir": "D2"}
ROW_LABEL = {
    "D1_154_building": ("D1", "154 images, convergent"),
    "D2_111_nadir": ("D2", "111 images, near-nadir"),
}

FRONT_ENDS = ["rootsift", "doghardnet", "superpoint", "aliked", "disk", "loftr"]
FRONT_END_LABEL = {
    "rootsift": "RootSIFT",
    "sift": "SIFT",
    "doghardnet": "DoG + HardNet",
    "superpoint": "SuperPoint",
    "aliked": "ALIKED",
    "disk": "DISK",
    "loftr": "LoFTR",
}
# ordered by relative luminance, minimum gap 25 levels, separable under the
# common colour vision deficiencies
FRONT_END_COLOR = {
    "rootsift": "#1A1A1A",
    "sift": "#5A5A5A",
    "doghardnet": "#004E9E",
    "superpoint": "#008E73",
    "aliked": "#D2691E",
    "disk": "#B48CE0",
    "loftr": "#C9930A",
}
FRONT_END_MARKER = {
    "rootsift": "o",
    "sift": "o",
    "doghardnet": "s",
    "superpoint": "^",
    "aliked": "D",
    "disk": "v",
    "loftr": "P",
}
MATCHER_LABEL = {
    "lightglue": "LightGlue",
    "nn_ratio": "Nearest neighbour with ratio test",
    "dense": "Detector-free",
}
MATCHER_STYLE = {"lightglue": "-", "nn_ratio": (0, (5, 2)), "dense": "-"}

# shared strata, used identically by the block-geometry and stratification figures
CONVERGENCE_EDGES = [0, 2, 10, 25, 35, 50]
BASELINE_EDGES = [0, 20, 40, 60, 80, 120]

RESOLUTIONS = [1024, 1600, 2048, 3200, "native"]
BUDGETS = [2048, 4096, 8192, 16384]
REFERENCE_RESOLUTION = 2048
REFERENCE_BUDGET = 8192

NO_DATA = "#DCDCDC"
EMPTY_BIN = "#F2F2F2"
SUPPORT_BAR = "#C9CDD4"
RULE = "#999999"

CONFIG_PATTERN = re.compile(
    r"(.+?)-(lightglue|nn_ratio|dense)-(r\d+|native)-k(\d+|all)(-c\d+)?$")


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 4,
        "axes.labelsize": 9.5,
        "axes.labelcolor": "#222222",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.1,
        "lines.markersize": 4.0,
        "figure.dpi": 150,
        "savefig.dpi": 400,          # filled meshes reach the page at this density
        "pdf.fonttype": 42,          # embedded TrueType, never Type 3
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    })


def save(fig, name: str, dpi: int = 600) -> Path:
    """Save at the authored size. A tight bounding box would change the width."""
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=dpi)
    fig.savefig(OUT / f"{name}.pdf")
    size = fig.get_size_inches() * 25.4
    plt.close(fig)
    print(f"  wrote {path.name}  {size[0]:.0f} x {size[1]:.0f} mm")
    return path


# ----------------------------------------------------------------------------
# annotation and marks, one vocabulary for the whole set
# ----------------------------------------------------------------------------
def note(ax, text: str, corner: str = "upper left", fontsize: float = 8.5):
    """One statistics box per panel, in the same corner throughout a figure."""
    from matplotlib.offsetbox import AnchoredText

    box = AnchoredText(text, loc=corner, prop=dict(size=fontsize, color="#3c3c3c"),
                       frameon=True, borderpad=0.3, pad=0.28)
    box.patch.set_boxstyle("square,pad=0.3")
    box.patch.set_facecolor("white")
    box.patch.set_alpha(0.9)
    box.patch.set_edgecolor(RULE)
    box.patch.set_linewidth(0.3)
    box.set_zorder(12)
    ax.add_artist(box)
    return box


def panel_letter(ax, letter: str, inside: bool = False, corner: str = "upper left",
                 suffix: str = "") -> None:
    """Panel letter in a corner the data leaves free, optionally with a count."""
    text = f"({letter})" + (f"  {suffix}" if suffix else "")
    if not inside:
        ax.text(0.5, -0.32, text, transform=ax.transAxes, fontsize=8.5,
                family="serif", va="top", ha="center")
        return
    x, ha = (0.025, "left") if corner.endswith("left") else (0.975, "right")
    y, va = (0.97, "top") if corner.startswith("upper") else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.5, family="serif",
            fontweight="bold", va=va, ha=ha, zorder=12,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.2))


def row_label(fig, ax, dataset: str, x_name: float = 0.012,
              descriptor: bool = True, x_descriptor: float = 0.032) -> None:
    """Block identity in the left margin, never inside an axis label.

    The label is rotated and centred on the position given, so it needs half its
    own height of clearance from the edge of the page. That is about five points
    at this size, and a position closer than that cuts the stems, so the request
    is held off the edge rather than taken literally.
    """
    name, text = ROW_LABEL[dataset]
    box = ax.get_position()
    y = (box.y0 + box.y1) / 2
    margin = 6.3 / (fig.get_size_inches()[0] * 72.0)
    fig.text(max(x_name, margin), y, name, rotation=90, va="center",
             ha="center", fontsize=9, family="serif", fontweight="bold")
    if descriptor:
        fig.text(x_descriptor, y, text, rotation=90, va="center", ha="center",
                 fontsize=8.5, family="serif", style="italic", color="#555555")


def not_applicable(ax, x, y, colour: str = "#666666", size: float = 5.0,
                   transform=None) -> None:
    """The mark for a factor that does not apply, drawn where it would have sat."""
    shared = dict(linestyle="none", zorder=6, clip_on=transform is None)
    if transform is not None:
        shared["transform"] = transform
    ax.plot([x], [y], marker="o", markersize=size, markerfacecolor="none",
            markeredgecolor=colour, markeredgewidth=0.7, **shared)
    ax.plot([x], [y], marker=(2, 0, 45), markersize=size * 0.8, color=colour,
            markeredgewidth=0.7, **shared)


def wilson(successes: int, total: int, z: float = 1.96):
    """Wilson score interval, which behaves at rates near zero and one."""
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    return p, max(centre - spread, 0.0), min(centre + spread, 1.0)


def inverted_pairs(a, b) -> int:
    """Pairs whose order disagrees between two rankings, for small samples."""
    a, b = np.asarray(a), np.asarray(b)
    count = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if (a[i] - a[j]) * (b[i] - b[j]) < 0:
                count += 1
    return count


def scale_bar(ax, length: float, label: str, colour: str = "#222222") -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x1 - (x1 - x0) * 0.05 - length
    y = y0 + (y1 - y0) * 0.06
    ax.plot([x, x + length], [y, y], color=colour, linewidth=1.4,
            solid_capstyle="butt", zorder=11)
    for position in (x, x + length):
        ax.plot([position, position], [y, y + (y1 - y0) * 0.016], color=colour,
                linewidth=1.0, zorder=11)
    ax.text(x + length / 2, y + (y1 - y0) * 0.028, label, ha="center", va="bottom",
            fontsize=8.5, color=colour, zorder=11)


def north_arrow(ax, colour: str = "#222222") -> None:
    ax.annotate("N", xy=(0.95, 0.93), xytext=(0.95, 0.80),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="bottom", fontsize=8.5, color=colour,
                arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=0.8,
                                shrinkA=0, shrinkB=0), zorder=11)


# ----------------------------------------------------------------------------
# data access
# ----------------------------------------------------------------------------
def parse_config(name: str):
    match = CONFIG_PATTERN.match(name)
    if not match:
        return None
    resolution = match.group(3)
    return {
        "front_end": match.group(1),
        "matcher": match.group(2),
        "resolution": "native" if resolution == "native" else int(resolution[1:]),
        "budget": None if match.group(4) == "all" else int(match.group(4)),
        "variant": match.group(5) or "",
        "config": name,
    }


def load_runs() -> list[dict]:
    joined = defaultdict(dict)
    for path in MANIFESTS_DIR.glob("*/*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        joined[(path.parent.name, manifest["config"])]["matching"] = manifest
    for path in SFM_REPORTS_DIR.glob("*/*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["seed"] != 0:
            continue
        joined[(report["dataset"], report["config"])]["sfm"] = report
    for path in EVALUATION_DIR.glob("*/*.json"):
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        if "__seed" in evaluation["run_id"] or evaluation.get("status") != "ok":
            continue
        joined[(evaluation["dataset"], evaluation["run_id"])]["network"] = evaluation
    for path in SURFACE_DIR.glob("*/*.json"):
        surface = json.loads(path.read_text(encoding="utf-8"))
        joined[(surface["dataset"], surface["config"])]["surface"] = surface

    runs = []
    for (dataset, config), entry in joined.items():
        parsed = parse_config(config)
        if not parsed:
            continue
        runs.append({"dataset": dataset, **parsed, **entry})
    return runs


def load_seed_runs() -> dict:
    grouped = defaultdict(list)
    for path in EVALUATION_DIR.glob("*/*.json"):
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        if evaluation.get("status") != "ok":
            continue
        base = evaluation["run_id"].split("__seed")[0]
        grouped[(evaluation["dataset"], base)].append(evaluation)
    return {k: v for k, v in grouped.items() if len(v) >= 3}


def reference_geometry() -> dict:
    return json.loads(
        (REFERENCE_DIR / "network_geometry_summary.json").read_text(encoding="utf-8"))


def reference_poses(dataset: str):
    import csv

    with (REFERENCE_DIR / f"{dataset}_reference_poses.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pair_geometry(dataset: str):
    import csv

    with (REFERENCE_DIR / f"{dataset}_pair_geometry.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pairwise_metrics(dataset: str, config: str):
    import csv

    path = PAIRWISE_DIR / dataset / f"{config}.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _repeatability() -> dict:
    path = PROJECT_ROOT / "results" / "tables" / "resolution_limit.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolution_limit_m() -> float:
    return _repeatability().get("largest_seed_range_m", float("nan"))


def seed_spread(quantity: str) -> float:
    """Largest run-to-run spread measured for a reported quantity.

    A difference smaller than this is not resolvable by the study, so it is
    shown as a tie rather than as an ordering.
    """
    return _repeatability().get("largest_seed_range", {}).get(quantity, float("nan"))


def ground_aim_points(dataset: str):
    """Where each optical axis meets the median terrain height.

    The aim point makes the acquisition design visible: an inward looking
    oblique orbit concentrates its aim points over the object, whereas nadir
    strips place them under the stations.
    """
    poses = reference_poses(dataset)
    summary = reference_geometry()[SHORT[dataset]]
    terrain = summary["terrain_median_height_m"]
    east = np.array([float(p["east_m"]) for p in poses])
    north = np.array([float(p["north_m"]) for p in poses])
    up = np.array([float(p["up_m"]) for p in poses])
    axis = np.array([[float(p["axis_e"]), float(p["axis_n"]), float(p["axis_u"])]
                     for p in poses])
    slant = (up - terrain) / np.maximum(-axis[:, 2], 1e-6)
    return east + axis[:, 0] * slant, north + axis[:, 1] * slant


def look_direction_classes(dataset: str, tilt_threshold: float = 5.0):
    """Group the tilted stations by viewing azimuth.

    The classes are found from the data rather than assumed, because the two
    blocks differ in exactly this: one has four crossing look directions and
    the other has a single one.
    """
    poses = reference_poses(dataset)
    tilted = [(float(p["viewing_azimuth_deg"]), p["camera_label"]) for p in poses
              if float(p["tilt_from_nadir_deg"]) >= tilt_threshold]
    if not tilted:
        return {}, []
    azimuths = np.array([a for a, _ in tilted])
    order = np.argsort(azimuths)
    sorted_azimuths = azimuths[order]

    # Split at any gap wider than 15 degrees, then fold the sparse groups into
    # the nearest populated one. A handful of stations sit between the main
    # bearings and would otherwise bridge two clusters into one.
    breaks = np.flatnonzero(np.diff(sorted_azimuths) > 15.0)
    groups = [g for g in np.split(np.arange(len(sorted_azimuths)), breaks + 1) if len(g)]
    populated = [g for g in groups if len(g) >= 5]
    if not populated:
        populated = groups

    def circular_distance(a: float, b: float) -> float:
        difference = abs(a - b) % 360.0
        return min(difference, 360.0 - difference)

    def circular_mean(values: np.ndarray) -> float:
        """Mean bearing, which an arithmetic mean gets wrong across north.

        One of the four groups of the convergent block straddles zero, holding
        both 359 and 19 degrees, and an arithmetic mean of those places the
        group eleven degrees away from almost every one of its members.
        """
        radians = np.radians(values)
        return float(np.degrees(np.arctan2(np.sin(radians).mean(),
                                           np.cos(radians).mean())) % 360.0)

    centres = [circular_mean(sorted_azimuths[g]) for g in populated]
    classes, members = {}, [list(g) for g in populated]
    for group in groups:
        if any(np.array_equal(group, g) for g in populated):
            continue
        target = int(np.argmin([circular_distance(circular_mean(sorted_azimuths[group]), c)
                                for c in centres]))
        members[target].extend(group.tolist())

    means = []
    for index, group in enumerate(members):
        means.append(circular_mean(sorted_azimuths[np.array(sorted(group))]))
        for position in group:
            classes[tilted[order[position]][1]] = index
    return classes, means
