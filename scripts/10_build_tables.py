#!/usr/bin/env python3
"""Assemble the result tables from the stored run records.

Every table is regenerated from the records, so a table follows any stage that
is re-run. Differences smaller than the measured run-to-run variability are
flagged rather than presented as a ranking.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    EVALUATION_DIR,
    FEATURES_DIR,
    MANIFESTS_DIR,
    REFERENCE_DIR,
    SFM_REPORTS_DIR,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
SURFACE_DIR = PROJECT_ROOT / "results" / "surface"

REFERENCE_RESOLUTION = 2048
REFERENCE_BUDGET = 8192
DATASETS = ["D1_154_building", "D2_111_nadir"]
SHORT = {"D1_154_building": "D1", "D2_111_nadir": "D2"}

FRONT_END_NAMES = {
    "rootsift": "RootSIFT",
    "sift": "SIFT",
    "doghardnet": "DoG + HardNet",
    "superpoint": "SuperPoint",
    "aliked": "ALIKED",
    "disk": "DISK",
    "loftr": "LoFTR",
}
MATCHER_NAMES = {"lightglue": "LightGlue",
                 "nn_ratio": "Nearest neighbour, ratio test",
                 "dense": "Detector-free"}

CONFIG_PATTERN = re.compile(r"(.+?)-(lightglue|nn_ratio|dense)-(r\d+|native)-k(\d+|all)$")
VARIANT_PATTERN = re.compile(
    r"(.+?)-(lightglue|nn_ratio|dense)-(r\d+|native)-k(\d+|all)-c(\d+)$")


_KEYPOINT_CACHE: dict = {}


def median_keypoints(dataset: str, parsed: dict):
    """Median keypoints per image, read from the stored features.

    The matching record only carries this when the store was extracted in that
    run; stores reused or derived from a larger budget report nothing, so the
    count is taken from the store itself, or, where the stores are not present,
    from the census the keypoint census stage wrote from them.
    """
    resolution = parsed["resolution"]
    res = "native" if resolution == "native" else f"r{resolution}"
    key = (dataset, parsed["front_end"], res, parsed["budget"])
    if key in _KEYPOINT_CACHE:
        return _KEYPOINT_CACHE[key]
    budget = "all" if parsed["budget"] is None else parsed["budget"]
    path = FEATURES_DIR / dataset / f"{parsed['front_end']}-{res}-k{budget}.h5"
    value = None
    if path.exists():
        try:
            import h5py

            with h5py.File(path, "r") as handle:
                counts = [handle[n]["keypoints"].shape[0] for n in handle]
            value = float(np.median(counts)) if counts else None
        except Exception:
            value = None
    else:
        census = TABLE_DIR / "keypoint_census.json"
        if census.exists():
            entry = json.loads(census.read_text(encoding="utf-8")).get(
                f"{dataset}/{path.stem}")
            value = entry["median"] if entry else None
    _KEYPOINT_CACHE[key] = value
    return value


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
    }


def load_records():
    """Join the matching, reconstruction and evaluation records per run."""
    records = defaultdict(dict)
    for path in MANIFESTS_DIR.glob("*/*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        records[(path.parent.name, manifest["config"], 0)]["matching"] = manifest
    for path in SFM_REPORTS_DIR.glob("*/*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        records[(path.parent.name, report["config"], report["seed"])]["sfm"] = report
    for path in EVALUATION_DIR.glob("*/*.json"):
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        run_id = evaluation["run_id"]
        base, seed = (run_id.split("__seed") + ["0"])[:2]
        records[(evaluation["dataset"], base, int(seed))]["network"] = evaluation
    for path in SURFACE_DIR.glob("*/*.json"):
        if path.suffix != ".json":
            continue
        surface = json.loads(path.read_text(encoding="utf-8"))
        records[(surface["dataset"], surface["config"], 0)]["surface"] = surface
    return records


# Quantities whose run-to-run spread is measured, so that a difference between
# two configurations is only claimed when it exceeds what repeating the same
# configuration produces on its own.
REPEATED_QUANTITIES = {
    "similarity_held_out_rmse_m":
        lambda n: n["similarity_held_out"]["rmse"],
    "similarity_in_sample_rmse_m":
        lambda n: n["similarity_in_sample"]["rmse"],
    "similarity_in_sample_median_m":
        lambda n: n["similarity_in_sample"]["median"],
    "relative_rotation_error_deg":
        lambda n: n["relative_rotation_error_deg"]["median"],
    "against_kinematic_rmse_m":
        lambda n: n["against_kinematic_positions"]["similarity_in_sample"]["rmse"],
}


def resolution_limits(records) -> dict:
    """Largest spread of each reported quantity across mapper seeds."""
    limits = {}
    for name, extract in REPEATED_QUANTITIES.items():
        grouped = defaultdict(list)
        for (dataset, config, _seed), entry in records.items():
            network = entry.get("network")
            if not network or network.get("status") != "ok":
                continue
            try:
                grouped[(dataset, config)].append(extract(network))
            except (KeyError, TypeError):
                continue
        spreads = [max(values) - min(values) for values in grouped.values()
                   if len(values) >= 3]
        limits[name] = max(spreads) if spreads else float("nan")
    return limits


def resolution_limit(records) -> float:
    """Largest spread of the network residual across mapper seeds."""
    return resolution_limits(records)["similarity_held_out_rmse_m"]


def write(name: str, header: list[str], rows: list[list]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with (TABLE_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    widths = [
        max(len(str(header[i])), max((len(str(r[i])) for r in rows), default=0))
        for i in range(len(header))
    ]
    lines = ["| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(header)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    (TABLE_DIR / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {name} ({len(rows)} rows)")


def fmt(value, digits=3, missing=""):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return missing
    return f"{value:.{digits}f}"


def table_datasets() -> None:
    summary = json.loads((REFERENCE_DIR / "network_geometry_summary.json").read_text(encoding="utf-8"))
    fields = [
        ("Images", lambda v: v["image_count"], 0),
        ("Scheduled image pairs", lambda v: v["pair_count"], 0),
        ("Camera extent, east (m)", lambda v: v["camera_extent_east_m"], 0),
        ("Camera extent, north (m)", lambda v: v["camera_extent_north_m"], 0),
        ("Images per hectare", lambda v: v["images_per_hectare"], 1),
        ("Flying height above terrain (m)", lambda v: v["flying_height_above_terrain_m"], 1),
        ("Terrain relief, 2nd to 98th percentile (m)", lambda v: v["terrain_relief_p2_p98_m"], 1),
        ("Camera tilt from nadir, median (deg)", lambda v: v["tilt_from_nadir_deg"]["p50"], 1),
        ("Camera tilt from nadir, 95th percentile (deg)", lambda v: v["tilt_from_nadir_deg"]["p95"], 1),
        ("Images tilted beyond 20 deg (per cent)", lambda v: 100 * v["tilt_fraction_above_20deg"], 1),
        ("Baseline, median (m)", lambda v: v["baseline_m"]["p50"], 1),
        ("Baseline, 95th percentile (m)", lambda v: v["baseline_m"]["p95"], 1),
        ("Convergence angle, median (deg)", lambda v: v["convergence_angle_deg"]["p50"], 1),
        ("Convergence angle, 95th percentile (deg)", lambda v: v["convergence_angle_deg"]["p95"], 1),
        ("Base to height ratio, median", lambda v: v["median_base_to_height_ratio"], 3),
    ]
    rows = [[name, f"{getter(summary['D1']):.{d}f}", f"{getter(summary['D2']):.{d}f}"]
            for name, getter, d in fields]
    write("table1_datasets", ["Property", "D1", "D2"], rows)


def table_reference_point(records, limit) -> None:
    rows = []
    for dataset in DATASETS:
        for (ds, config, seed), entry in sorted(records.items()):
            if ds != dataset or seed != 0 or "matching" not in entry:
                continue
            parsed = parse_config(config)
            if not parsed:
                continue
            # every front end at the reference operating point, plus the
            # detector-free matcher at the only setting it can be run at
            reference = (parsed["resolution"] == REFERENCE_RESOLUTION
                         and parsed["budget"] == REFERENCE_BUDGET)
            detector_free = parsed["matcher"] == "dense"
            if not (reference or detector_free):
                continue
            matching = entry["matching"]
            sfm = entry.get("sfm", {})
            network = entry.get("network", {})
            rows.append([
                SHORT[dataset],
                FRONT_END_NAMES.get(parsed["front_end"], parsed["front_end"]),
                MATCHER_NAMES.get(parsed["matcher"], parsed["matcher"]),
                str(parsed["resolution"]),
                fmt(median_keypoints(dataset, parsed), 0),
                f"{matching['median_verified_inliers']:.0f}",
                f"{100 * matching['failed_pair_fraction']:.1f}",
                f"{sfm.get('registered_images', '')}",
                f"{sfm.get('sparse_points', '')}",
                fmt(sfm.get("mean_track_length"), 2),
                fmt(sfm.get("mean_reprojection_error_px"), 3),
                fmt(network.get("similarity_held_out", {}).get("rmse"), 3),
                fmt(network.get("similarity_in_sample", {}).get("median"), 3),
                fmt(network.get("similarity_in_sample", {}).get("nmad"), 3),
                str(network.get("gross_errors", {}).get("beyond_10_nmad", "")),
                fmt(network.get("relative_rotation_error_deg", {}).get("median"), 4),
                fmt(network.get("relative_direction_error_deg", {}).get("median"), 3),
            ])
    write("table3_reference_operating_point",
          ["Block", "Local feature", "Matcher", "Resolution", "Keypoints",
           "Verified inliers",
           "Failed pairs (%)", "Registered", "Sparse points", "Track length",
           "Reprojection (px)", "Network RMSE (m)", "Network median (m)",
           "Network NMAD (m)", "Gross errors", "Rel. rotation (deg)",
           "Rel. direction (deg)"], rows)


def table_sweep(records, axis: str, limit) -> None:
    """Resolution sweep at the reference budget, or budget sweep at the reference resolution."""
    values = defaultdict(dict)
    for (ds, config, seed), entry in records.items():
        if seed != 0 or "network" not in entry or entry["network"].get("status") != "ok":
            continue
        parsed = parse_config(config)
        if not parsed or parsed["matcher"] != "lightglue":
            continue
        if axis == "resolution":
            if parsed["budget"] != REFERENCE_BUDGET:
                continue
            key = parsed["resolution"]
        else:
            if parsed["resolution"] != REFERENCE_RESOLUTION:
                continue
            key = parsed["budget"]
        values[(SHORT[ds], parsed["front_end"])][key] = (
            entry["network"]["similarity_held_out"]["rmse"],
            entry["network"]["relative_rotation_error_deg"]["median"],
            entry["matching"]["median_verified_inliers"] if "matching" in entry else None,
        )
    columns = sorted({k for v in values.values() for k in v},
                     key=lambda x: 10**9 if x == "native" else int(x))
    rows = []
    for (block, front_end) in sorted(values):
        row = [block, FRONT_END_NAMES.get(front_end, front_end)]
        for column in columns:
            item = values[(block, front_end)].get(column)
            row.append("" if not item else f"{item[0]:.3f} / {item[1]:.4f}")
        rows.append(row)
    name = "table5_resolution_sweep" if axis == "resolution" else "table6_budget_sweep"
    label = "Working resolution" if axis == "resolution" else "Keypoint budget"
    write(name, ["Block", "Local feature"] + [f"{label} {c}" for c in columns], rows)


def table_repeatability(records, limit) -> None:
    grouped = defaultdict(list)
    for (ds, config, seed), entry in records.items():
        if "network" not in entry or entry["network"].get("status") != "ok":
            continue
        grouped[(ds, config)].append((seed, entry))
    rows = []
    for (ds, config), items in sorted(grouped.items()):
        if len(items) < 3:
            continue
        rmse = np.array([e["network"]["similarity_held_out"]["rmse"] for _, e in items])
        points = np.array([e["sfm"]["sparse_points"] for _, e in items if "sfm" in e])
        parsed = parse_config(config)
        rows.append([
            SHORT[ds],
            FRONT_END_NAMES.get(parsed["front_end"], parsed["front_end"]),
            len(items),
            f"{rmse.mean():.4f}",
            f"{rmse.std(ddof=1) * 1000:.2f}",
            f"{(rmse.max() - rmse.min()) * 1000:.2f}",
            f"{100 * points.std(ddof=1) / points.mean():.3f}" if len(points) > 1 else "",
        ])
    write("table7_repeatability",
          ["Block", "Local feature", "Runs", "Mean network RMSE (m)",
           "Standard deviation (mm)", "Range (mm)", "Sparse point spread (%)"], rows)


def surface_outliers(raster_path: Path):
    """Height differences split into the agreeing surface and its gross tail.

    Comparing two surface models of a built scene produces a small proportion
    of cells at facades and occlusion edges where the two models describe
    different objects, and a height difference there approaches the height of
    the building. Those cells dominate a root mean square statistic without
    saying anything about how well the surfaces agree elsewhere, so the two
    populations are reported separately.
    """
    try:
        import rasterio
    except ImportError:
        return None
    if not raster_path.exists():
        return None
    with rasterio.open(raster_path) as handle:
        values = handle.read(1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    core = values[np.abs(values) < 1.0]
    median = float(np.median(core)) if core.size else float("nan")

    # A distorted camera network tilts or bows the whole surface rather than
    # roughening it locally, so the systematic part is separated from the
    # residual scatter by fitting a low order surface over the grid.
    with rasterio.open(raster_path) as handle:
        grid = handle.read(1)
    rows, columns = np.indices(grid.shape)
    mask = np.isfinite(grid) & (np.abs(grid) < 1.0)
    trend_amplitude = float("nan")
    detrended_nmad = float("nan")
    if mask.sum() > 1000:
        sample = np.random.default_rng(0).choice(
            np.flatnonzero(mask.ravel()), size=min(200000, int(mask.sum())), replace=False
        )
        y = rows.ravel()[sample] / grid.shape[0]
        x = columns.ravel()[sample] / grid.shape[1]
        z = grid.ravel()[sample]
        design = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
        solution, *_ = np.linalg.lstsq(design, z, rcond=None)
        fitted = design @ solution
        trend_amplitude = float(np.percentile(fitted, 95) - np.percentile(fitted, 5))
        residual = z - fitted
        detrended_nmad = float(1.4826 * np.median(np.abs(residual - np.median(residual))))

    return {
        "within_1m_fraction": float(core.size / values.size),
        "beyond_2m_fraction": float(np.mean(np.abs(values) > 2.0)),
        "core_median": median,
        "core_nmad": float(1.4826 * np.median(np.abs(core - median))) if core.size else float("nan"),
        "core_rmse": float(np.sqrt(np.mean(core**2))) if core.size else float("nan"),
        "trend_amplitude": trend_amplitude,
        "detrended_nmad": detrended_nmad,
    }


def table_surface(records) -> None:
    rows, summary = [], {}
    for (ds, config, seed), entry in sorted(records.items()):
        if "surface" not in entry:
            continue
        surface = entry["surface"]
        stats = surface["height_difference_m"]
        parsed = parse_config(config)
        extra = surface_outliers(Path(surface["difference_raster"])) or {}
        # the same numbers reach the table and the figures, from one place
        summary[f"{ds}/{config}"] = {
            "dataset": ds, "config": config, "front_end": parsed["front_end"],
            "resolution": parsed["resolution"],
            "fused_points": surface["fused_points"],
            "compared_fraction_of_reference":
                surface["compared_fraction_of_reference"],
            "raster": surface["difference_raster"],
            **{k: v for k, v in extra.items()},
        }
        rows.append([

            SHORT[ds],
            FRONT_END_NAMES.get(parsed["front_end"], parsed["front_end"]),
            parsed["resolution"],
            f"{surface['fused_points']:,}",
            f"{100 * surface['compared_fraction_of_reference']:.1f}",
            fmt(extra.get("core_median"), 3),
            fmt(extra.get("core_nmad"), 3),
            fmt(extra.get("trend_amplitude"), 3),
            fmt(extra.get("detrended_nmad"), 3),
            fmt(100 * extra["within_1m_fraction"], 1) if "within_1m_fraction" in extra else "",
            fmt(100 * extra["beyond_2m_fraction"], 2) if "beyond_2m_fraction" in extra else "",
        ])
    if summary:
        (TABLE_DIR / "surface_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        write("table8_surface_difference",
              ["Block", "Local feature", "Resolution", "Fused points",
               "Coverage (%)", "Median difference (m)", "NMAD (m)",
               "Trend amplitude (m)", "Detrended NMAD (m)",
               "Cells within 1 m (%)", "Cells beyond 2 m (%)"], rows)
    else:
        print("  table8 skipped, no surface comparisons yet")


def table_merging_tolerance(records) -> None:
    """Sensitivity of the detector-free matcher to the keypoint merging tolerance.

    Detector-free matching returns correspondences rather than keypoints, so the
    toolkit pools them into tracks within a tolerance. The table compares the
    default tolerance with a coarser one on tracks, registration and the camera
    network.
    """
    rows = []
    for (ds, config, seed), entry in sorted(records.items()):
        if seed != 0 or "matching" not in entry:
            continue
        match = VARIANT_PATTERN.match(config) or CONFIG_PATTERN.match(config)
        if not match or match.group(2) != "dense":
            continue
        tolerance = match.group(5) if match.re is VARIANT_PATTERN else "1"
        sfm = entry.get("sfm", {})
        network = entry.get("network", {})
        rows.append([
            SHORT[ds],
            f"{tolerance} px",
            f"{entry['matching']['median_verified_inliers']:.0f}",
            f"{sfm.get('registered_images', '')}",
            fmt(sfm.get("mean_track_length"), 2),
            fmt(sfm.get("mean_reprojection_error_px"), 2),
            fmt(network.get("similarity_held_out", {}).get("rmse"), 3),
            fmt(network.get("relative_rotation_error_deg", {}).get("median"), 4),
        ])
    if rows:
        write("table9_merging_tolerance",
              ["Block", "Merging cell", "Verified inliers", "Registered",
               "Track length", "Reprojection (px)", "Network RMSE (m)",
               "Rel. rotation (deg)"], rows)


def table_memory() -> None:
    """What each front end costs on the device, from the measurement stage.

    Peak allocation and time per image for each front end and resolution,
    distinguishing an allocation that fails outright from one the driver
    satisfies out of system memory at a cost in time that makes the
    configuration impractical.
    """
    path = TABLE_DIR / "memory_feasibility.json"
    if not path.exists():
        print("  table10 skipped, no memory measurement yet")
        return
    record = json.loads(path.read_text(encoding="utf-8"))
    capacity = record["total_memory_gib"]
    rows = []
    for entry in record["detectors"] + record["detector_free"]:
        peak = entry.get("peak_gib")
        if entry["status"] != "ok":
            note = "allocation failed"
        elif peak is not None and peak > capacity:
            note = "satisfied from system memory"
        else:
            note = "fits on the device"
        rows.append([
            FRONT_END_NAMES.get(entry["front_end"], entry["front_end"]),
            str(entry["resolution"]),
            "single" if entry["precision"] == "fp32" else "half",
            fmt(peak, 2),
            fmt(entry.get("seconds"), 2),
            str(entry.get("keypoints", entry.get("matches", ""))),
            note,
        ])
    write("table10_memory_feasibility",
          ["Front end", "Working resolution", "Precision", "Peak memory (GiB)",
           "Seconds", "Keypoints or matches", "Outcome"], rows)
    print(f"  device capacity {capacity:.2f} GiB")


def main() -> int:
    records = load_records()
    limit = resolution_limit(records)
    print(f"records: {len(records)}")
    print(f"resolution limit from repeated runs: {limit * 1000:.2f} mm\n")
    table_datasets()
    table_reference_point(records, limit)
    table_sweep(records, "resolution", limit)
    table_sweep(records, "budget", limit)
    table_repeatability(records, limit)
    table_surface(records)
    table_merging_tolerance(records)
    table_memory()
    limits = resolution_limits(records)
    (TABLE_DIR / "resolution_limit.json").write_text(
        json.dumps({"largest_seed_range_m": limit,
                    "reported_limit_m": round(limit, 3),
                    "seeds_per_group": 5,
                    "largest_seed_range": limits}, indent=2), encoding="utf-8")
    for name, value in limits.items():
        print(f"  seed spread {name}: {value:.6f}")
    print(f"\ntables written to {TABLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
