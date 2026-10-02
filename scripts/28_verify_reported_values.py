"""Check every number the article quotes against the record that produced it.

One entry per value that appears in the article or its supporting information.
A value that drifts because an analysis was rerun is reported here.
"""
import csv
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis"
RECORDS = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
import _style as fs  # noqa: E402

checks = []


def check(label, got, want, tol=0.051):
    try:
        ok = abs(float(got) - float(want)) <= tol
    except (TypeError, ValueError):
        ok = got == want
    checks.append((ok, label, got, want))


def same(label, got, want):
    checks.append((got == want, label, got, want))


support = json.loads((ANALYSIS / "pair_support_summary_operating.json").read_text(
    encoding="utf-8"))

d1 = support["D1_154_building"]
d2 = support["D2_111_nadir"]
det1 = [v["precision"] for k, v in d1.items() if "loftr" not in k]
det2 = [v["precision"] for k, v in d2.items() if "loftr" not in k]
check("precision, detector-based, both blocks, minimum", min(det1 + det2), 0.93, 0.005)
check("precision, detector-based, both blocks, maximum", max(det1 + det2), 1.00, 0.005)
check("precision, detector-based, D1, minimum", min(det1), 0.98, 0.005)
check("precision, detector-free, D1", d1["loftr-dense-r1024-kall"]["precision"], 0.81)
check("precision, detector-free, D2", d2["loftr-dense-r1024-kall"]["precision"], 0.86)

# recovery at 25 to 35 degrees in D1, the three jointly learned configurations
recovered = []
for key in ("superpoint-lightglue-r2048-k8192", "aliked-lightglue-r2048-k8192",
            "disk-lightglue-r2048-k8192"):
    recovered.append(100.0 - d1[key]["classes"]["convergence"][3]["failed_pct"])
check("recovered at 25 to 35 degrees, lowest", min(recovered), 24.0, 0.5)
check("recovered at 25 to 35 degrees, highest", max(recovered), 33.0, 0.5)

# the half-loss angle, quoted only inside the angles a block contains
HALF = {
    "disk-lightglue-r2048-k8192": (19.2, 16.4, 22.3),
    "aliked-lightglue-r2048-k8192": (23.0, 19.3, 26.3),
    "superpoint-lightglue-r2048-k8192": (23.9, 20.6, 27.4),
    "loftr-dense-r1024-kall": (36.6, 31.5, 43.6),
}
for key, (angle, low, high) in HALF.items():
    entry = d1[key]
    same(f"half-loss reached in D1, {key}", entry["half_angle_reached"], True)
    check(f"half-loss angle in D1, {key}",
          entry["half_angle_accepted_deg"], angle, 0.06)
    check(f"half-loss interval low in D1, {key}",
          entry["half_angle_interval"]["low_deg"], low, 0.06)
    check(f"half-loss interval high in D1, {key}",
          entry["half_angle_interval"]["high_deg"], high, 0.06)
for key in ("rootsift-lightglue-r2048-k8192", "rootsift-nn_ratio-r2048-k8192",
            "doghardnet-lightglue-r2048-k8192"):
    same(f"half-loss not reached in D1, {key}",
         d1[key]["half_angle_reached"], False)
check("D1 ceiling on the fitted angle",
      d1["rootsift-lightglue-r2048-k8192"]["half_angle_ceiling_deg"], 61.1, 0.05)
check("unreached share for the learned descriptor in D1",
      100 * d1["doghardnet-lightglue-r2048-k8192"]
      ["half_angle_interval"]["unreached_share"], 96.0, 0.05)
for key in d2:
    same(f"half-loss not reached in D2, {key}", d2[key]["half_angle_reached"], False)
check("D2 ceiling on the fitted angle",
      d2["rootsift-lightglue-r2048-k8192"]["half_angle_ceiling_deg"], 45.0, 0.05)

# reference-consistent share beyond 120 m in D2, and the one-pair lead
beyond = {v["method"]: v["classes"]["baseline"][5]["supportable_pct"]
          for v in d2.values()}
check("reference-consistent beyond 120 m, lowest", min(beyond.values()), 3.0, 0.6)
check("reference-consistent beyond 120 m, highest", max(beyond.values()), 6.9, 0.6)
check("reference-consistent beyond 120 m, detector-free", beyond["LoFTR"], 6.9, 0.06)
check("reference-consistent beyond 120 m, best detector-based",
      max(v for k, v in beyond.items() if k != "LoFTR"), 6.9, 0.06)
check("candidate pairs beyond 120 m in D2",
      d2["loftr-dense-r1024-kall"]["classes"]["baseline"][5]["count"], 101, 0.5)

# the tilted subset of D2
tilted = [v["classes"]["convergence"][4]["supportable_pct"] for v in d2.values()]
check("reference-consistent in the tilted subset, lowest", min(tilted), 56.0, 0.6)
check("reference-consistent in the tilted subset, highest", max(tilted), 58.0, 0.6)

# aggregate intervals quoted in the text
check("D1 SuperPoint aggregate", d1["superpoint-lightglue-r2048-k8192"]["failed_pct"],
      35.4, 0.05)
check("D1 SuperPoint interval, low", d1["superpoint-lightglue-r2048-k8192"]["failed_low"],
      29.3, 0.4)
check("D1 SuperPoint interval, high",
      d1["superpoint-lightglue-r2048-k8192"]["failed_high"], 41.6, 0.4)
check("D1 RootSIFT aggregate", d1["rootsift-lightglue-r2048-k8192"]["failed_pct"],
      6.1, 0.05)

# the common resolution table
rows = list(csv.DictReader((ANALYSIS / "common_sets.csv").open(encoding="utf-8")))


def shared(dataset, config):
    for row in rows:
        if row["dataset"] == dataset and row["config"] == config:
            return 1000.0 * float(row["held_out_rmse_shared_m"])
    raise KeyError(config)


check("D1 at 1024, RootSIFT", shared("D1_154_building",
                                     "rootsift-lightglue-r1024-k8192"), 39.0, 0.5)
check("D1 at 1024, DoG + HardNet", shared("D1_154_building",
                                          "doghardnet-lightglue-r1024-k8192"), 40.6, 0.6)
check("D1 at 1024, SuperPoint", shared("D1_154_building",
                                       "superpoint-lightglue-r1024-k8192"), 43.9, 0.5)
check("D1 at 1024, DISK", shared("D1_154_building",
                                 "disk-lightglue-r1024-k8192"), 132.5, 0.5)
check("D1 at 1024, ALIKED", shared("D1_154_building",
                                   "aliked-lightglue-r1024-k8192"), 227.9, 0.5)
check("D2 at 1024, DoG + HardNet", shared("D2_111_nadir",
                                          "doghardnet-lightglue-r1024-k8192"), 46.3, 0.5)
check("D2 at 1024, LoFTR", shared("D2_111_nadir",
                                  "loftr-dense-r1024-kall"), 75.6, 0.5)

# the interior orientation relation, reported on the five detector-based
# configurations of the near-nadir block
interior = json.loads((ANALYSIS / "interior_orientation.json").read_text(encoding="utf-8"))
relation = interior["relation"]
five = relation["D2_111_nadir | without LoFTR"]
check("D2 configurations behind the relation", five["n"], 5, 0.5)
check("D2 rho, principal distance against surface median",
      five["spearman_c_vs_surface_median"], -1.0, 0.001)
check("D2 fitted slope", five["slope_m_per_px"], -0.022, 0.0006)
check("D2 slope, leave-one-out minimum", five["slope_leave_one_out_min"],
      -0.027, 0.0006)
check("D2 slope, leave-one-out maximum", five["slope_leave_one_out_max"],
      -0.021, 0.0006)
check("D2 predicted slope", five["predicted_slope_m_per_px"], -0.025, 0.0006)
check("D2 principal distance spread", five["c_spread_px"], 14.6, 0.06)
check("D2 flying height behind the prediction", five["flying_height_m"], 94.0, 0.5)
check("D2 reference principal distance", five["reference_principal_distance_px"],
      3699.5, 0.5)
with_loftr = relation["D2_111_nadir | all"]
check("D2 rho including the detector-free run",
      with_loftr["spearman_c_vs_surface_median"], -1.0, 0.001)
check("D2 slope including the detector-free run",
      with_loftr["slope_m_per_px"], -0.020, 0.0006)
six = relation["D1_154_building | without LoFTR"]
check("D1 configurations behind the relation", six["n"], 6, 0.5)
check("D1 rho, without detector-free", six["spearman_c_vs_surface_median"],
      -0.49, 0.006)
check("D1 rho, leave-one-out minimum", six["spearman_leave_one_out_min"],
      -0.70, 0.006)
check("D1 rho, leave-one-out maximum", six["spearman_leave_one_out_max"],
      -0.10, 0.006)
check("D1 fitted slope", six["slope_m_per_px"], -0.018, 0.0006)
check("D1 predicted slope", six["predicted_slope_m_per_px"], -0.019, 0.0006)
check("largest repeated-run spread of the principal distance",
      max(v["c_range_px"] for v in interior["repeatability"].values()), 0.585, 0.001)

# the on-board decomposition, over the thirty-six configurations of the design
onboard = json.loads((ANALYSIS / "onboard_decomposition.json").read_text(encoding="utf-8"))
check("D1 on-board RMS", onboard["D1_154_building"]["rms_3d_mm"], 41.0, 0.05)
check("D2 on-board RMS", onboard["D2_111_nadir"]["rms_3d_mm"], 207.5, 0.05)
check("share carried by the tilted subset",
      100 * onboard["D2_111_nadir"]["share_of_squared_difference_from_tilted"],
      78.4, 0.05)
check("median difference, tilted",
      onboard["D2_111_nadir"]["median_difference_tilted_mm"], 643.2, 0.5)
check("median difference, near-nadir",
      onboard["D2_111_nadir"]["median_difference_nadir_mm"], 87.0, 0.5)
check("configurations in the on-board comparison, D2",
      onboard["D2_111_nadir"]["configurations_compared"], 36, 0.5)
check("rho over configurations, D2",
      onboard["D2_111_nadir"]["spearman_reference_vs_onboard"], -0.69, 0.006)
check("rho over the near-nadir cameras, D2",
      onboard["D2_111_nadir"]["spearman_reference_vs_onboard_nadir_only"], 0.19, 0.006)
check("near-nadir cameras used, D2",
      onboard["D2_111_nadir"]["nadir_cameras_used"], 102, 0.5)
check("rho over configurations, D1",
      onboard["D1_154_building"]["spearman_reference_vs_onboard"], 0.91, 0.006)
check("rho over the near-nadir cameras, D1",
      onboard["D1_154_building"]["spearman_reference_vs_onboard_nadir_only"],
      0.99, 0.006)

# the surface gate and the common mask
mask = json.loads((ANALYSIS / "common_surface_mask.json").read_text(encoding="utf-8"))
summary, gate = mask["summary"], mask["rows"]
ordinary = [r for r in gate if not (r["dataset"] == "D1_154_building"
                                    and r["method"] == "LoFTR")]
check("gate retention, lowest of the twelve",
      min(r["kept_by_gate_pct"] for r in ordinary), 96.3, 0.05)
check("gate retention, highest of the twelve",
      max(r["kept_by_gate_pct"] for r in ordinary), 97.6, 0.05)
check("beyond half a metre, lowest of the twelve",
      min(r["beyond_0p5m_pct"] for r in ordinary), 3.1, 0.05)
check("beyond half a metre, highest of the twelve",
      max(r["beyond_0p5m_pct"] for r in ordinary), 14.0, 0.05)
check("beyond two metres, lowest of the twelve",
      min(r["beyond_2m_pct"] for r in ordinary), 1.7, 0.05)
check("beyond two metres, highest of the twelve",
      max(r["beyond_2m_pct"] for r in ordinary), 2.5, 0.05)
loftr = next(r for r in gate if r["dataset"] == "D1_154_building"
             and r["method"] == "LoFTR")
check("gate retention, detector-free in D1", loftr["kept_by_gate_pct"], 21.7, 0.05)
check("beyond half a metre, detector-free in D1", loftr["beyond_0p5m_pct"], 88.0, 0.05)
check("beyond two metres, detector-free in D1", loftr["beyond_2m_pct"], 56.8, 0.05)
check("median before the gate, detector-free in D1",
      loftr["median_before_gate_m"], -0.185, 0.0006)
check("median after the gate, detector-free in D1",
      loftr["median_own_m"], -0.307, 0.0006)
check("dispersion before the gate, detector-free in D1",
      loftr["nmad_before_gate_m"], 2.950, 0.0006)
check("dispersion after the gate, detector-free in D1",
      loftr["nmad_own_m"], 0.559, 0.0006)
check("cells before the gate, detector-free in D1",
      loftr["cells_before_gate"], 1263767, 0.5)

check("common cells, D1", summary["D1_154_building"]["cells_coarse_common"],
      130970, 0.5)
check("common cells, D2", summary["D2_111_nadir"]["cells_coarse_common"],
      258802, 0.5)
check("common cells before the gate, D1",
      summary["D1_154_building"]["cells_coarse_common_before_gate"], 196563, 0.5)
check("common cells before the gate, D2",
      summary["D2_111_nadir"]["cells_coarse_common_before_gate"], 262955, 0.5)
check("strict common cells, D1",
      summary["D1_154_building"]["cells_strict_common"], 3052, 0.5)
check("strict common cells, D2",
      summary["D2_111_nadir"]["cells_strict_common"], 3072, 0.5)
check("footprint spanned by the common mask, D1",
      100 * summary["D1_154_building"]["footprint_share_coarse_common"], 69.2, 0.5)
check("footprint spanned by the common mask, D2",
      100 * summary["D2_111_nadir"]["footprint_share_coarse_common"], 79.3, 0.5)
check("local component factor on the common mask, D1",
      summary["D1_154_building"]["local_component_factor_coarse_common"],
      1.39, 0.005)
check("local component factor on the common mask, D2",
      summary["D2_111_nadir"]["local_component_factor_coarse_common"],
      1.27, 0.005)
check("local component factor before the gate, D1",
      summary["D1_154_building"]["local_component_factor_coarse_common_before_gate"],
      1.31, 0.005)
check("local component factor before the gate, D2",
      summary["D2_111_nadir"]["local_component_factor_coarse_common_before_gate"],
      1.4, 0.005)

# the reference solution and the positional references
meta = json.loads((ANALYSIS / "onboard_metadata.json").read_text(encoding="utf-8"))
for dataset, images in (("D1_154_building", 154), ("D2_111_nadir", 111)):
    entry = meta[dataset]
    check(f"exposures in the timestamp file, {dataset}",
          entry["exposures_in_timestamp_file"], images, 0.5)
    check(f"embedded positions repeating the timestamp file, {dataset}",
          entry["embedded_position_matches_timestamp_file"], images, 0.5)
    same(f"every exposure at a fixed solution, {dataset}",
         entry["satellite_status_flags"], {"50": images})
check("median vertical standard deviation, D1",
      1000 * meta["D1_154_building"]["sigma_vertical_m_median"], 70.0, 0.5)
check("median vertical standard deviation, D2",
      1000 * meta["D2_111_nadir"]["sigma_vertical_m_median"], 24.0, 0.5)
check("antenna offset, vertical, D1",
      meta["D1_154_building"]["antenna_offset_vertical_mm_median"], 188.0, 0.5)
targets = meta["D1_154_building"]["surveyed_target_check"]
check("surveyed targets in D1", targets["n"], 3, 0.5)
check("target agreement in plan", targets["rms_plan_m"], 0.046, 0.0006)
check("target agreement in height", targets["rms_height_m"], 0.008, 0.0006)
for dataset, sigma in (("D1_154_building", 0.78), ("D2_111_nadir", 0.59)):
    check(f"reprojection error of the reference solution, {dataset}",
          meta[dataset]["reference_project"]["reprojection_error_px"], sigma, 0.006)
    check(f"images in the reference solution, {dataset}",
          meta[dataset]["reference_project"]["registered_images"],
          meta[dataset]["reference_project"]["images"], 0.5)
    check(f"components of the reference solution, {dataset}",
          meta[dataset]["reference_project"]["components"], 1, 0.5)

# what the three jointly learned configurations recover above thirty-five degrees
for method, above_fifty in (("SuperPoint", 2), ("ALIKED", 0), ("DISK", 0)):
    key = next(k for k, v in d1.items() if v["method"] == method)
    classes = d1[key]["classes"]["convergence"]
    check(f"recovered between 35 and 50 degrees, {method}",
          round(classes[4]["count"] * (100 - classes[4]["failed_pct"]) / 100), 0, 0.5)
    check(f"recovered beyond 50 degrees, {method}",
          round(classes[5]["count"] * (100 - classes[5]["failed_pct"]) / 100),
          above_fifty, 0.5)
    check(f"reference-consistent beyond 50 degrees, {method}",
          classes[5]["supportable_pct"], 0.0, 0.001)
check("pairs between 35 and 50 degrees in D1",
      d1["rootsift-lightglue-r2048-k8192"]["classes"]["convergence"][4]["count"],
      225, 0.5)
check("pairs beyond 50 degrees in D1",
      d1["rootsift-lightglue-r2048-k8192"]["classes"]["convergence"][5]["count"],
      72, 0.5)
check("the classical detector recovers every pair between 25 and 50 degrees, D1",
      max(d1["rootsift-lightglue-r2048-k8192"]["classes"]["convergence"][i]["failed_pct"]
          for i in (3, 4)), 0.0, 0.001)
check("the learned descriptor beyond 50 degrees in D1",
      d1["doghardnet-lightglue-r2048-k8192"]["classes"]["convergence"][5]["failed_pct"],
      62.5, 0.05)

# the two adjusted principal distances of each reconstruction
interior_rows = list(csv.DictReader(
    (ANALYSIS / "interior_orientation.csv").open(encoding="utf-8")))
gaps = sorted(abs(float(r["fx_minus_fy_px"])) for r in interior_rows)
check("configurations whose two principal distances agree within a pixel",
      sum(1 for g in gaps if g < 1.0), 11, 0.5)
check("largest difference between the two principal distances", gaps[-1], 3.1, 0.05)
check("second largest difference between the two principal distances",
      gaps[-2], 1.9, 0.05)

# the four look directions of the convergent block, as circular means
_classes, _means = fs.look_direction_classes("D1_154_building")
for index, want in enumerate((41.0, 131.0, 220.0, 313.0)):
    check(f"look direction {index + 1} of D1", _means[index], want, 0.5)
for index, want in enumerate((32, 32, 29, 30)):
    check(f"images in look direction {index + 1} of D1",
          sum(1 for v in _classes.values() if v == index), want, 0.5)
_separations = [(_means[(i + 1) % 4] - _means[i]) % 360 for i in range(4)]
check("largest separation between look directions", max(_separations), 93.3, 0.5)
check("smallest separation between look directions", min(_separations), 88.5, 0.5)

# the size of the design, against the manifests
_pattern = re.compile(r"(.+?)-(lightglue|nn_ratio|dense)-(r\d+|native)-k(\d+|all)$")
for _dataset, _want in (("D1_154_building", 36), ("D2_111_nadir", 36)):
    _folder = RECORDS / "manifests" / _dataset
    _stems = [q.stem for q in _folder.glob("*.json") if _pattern.match(q.stem)]
    check(f"configurations in the design, {_dataset}", len(_stems), _want, 0.5)
_nn = [q for q in (RECORDS / "manifests"
                   / "D1_154_building").glob("rootsift-nn_ratio-*.json")]
check("settings of the nearest-neighbour variant", len(_nn), 2, 0.5)

# the pooled detector-free observations against the largest keypoint store
census = json.loads((RECORDS / "tables"
                     / "keypoint_census.json").read_text(encoding="utf-8"))
for dataset, factor in (("D1_154_building", 2.38), ("D2_111_nadir", 2.98)):
    pooled = census[f"{dataset}/loftr-r1024-kall"]["median"]
    largest = max(v["median"] for k, v in census.items()
                  if k.startswith(dataset) and "loftr" not in k)
    check(f"pooled observations against the largest keypoint store, {dataset}",
          pooled / largest, factor, 0.01)
    check(f"pooled observations against the adopted limit, {dataset}",
          pooled / 8192, factor * largest / 8192, 0.01)

# the gate sensitivity
gate = json.loads((ANALYSIS / "surface_gate_sensitivity.json").read_text(
    encoding="utf-8"))["summary"]
for dataset in ("D1_154_building", "D2_111_nadir"):
    for key in ("0.5", "2", "none"):
        check(f"median ordering against the one-metre gate at {key}, {dataset}",
              gate[dataset]["median_m"]["order_against_one_metre"][key], 1.0, 0.001)
    factors = gate[dataset]["local_component_factor"]
    low, high = (1.33, 1.70) if dataset == "D1_154_building" else (1.20, 1.58)
    check(f"smallest local component factor over the gates, {dataset}",
          min(factors.values()), low, 0.006)
    check(f"largest local component factor over the gates, {dataset}",
          max(factors.values()), high, 0.006)
    worst = min(
        min(gate[dataset][q]["order_against_one_metre"][k]
            for k in ("0.5", "2"))
        for q in ("trend_amplitude_m", "detrended_nmad_m"))
    # in the convergent block the trend ordering moves at half a metre
    same(f"trend and local orderings hold at a neighbouring gate, {dataset}",
         worst >= 0.82, dataset == "D2_111_nadir")

# the image projections behind the surveyed-target check
_projections = sorted(t["projections"] for t in targets["targets"])
check("smallest number of images measuring a target", _projections[0], 43, 0.5)
check("largest number of images measuring a target", _projections[-1], 92, 0.5)

# the four separations between consecutive look directions of D1
for index, want in enumerate((90.0, 88.0, 93.0, 89.0)):
    check(f"separation {index + 1} between consecutive look directions of D1",
          _separations[index], want, 0.5)
check("the four separations sum to a full turn", sum(_separations), 360.0, 0.05)

# how thinly an aggregated cell of the common set is populated
_constituents = json.loads(
    (ANALYSIS / "common_surface_constituents.json").read_text(encoding="utf-8"))
for dataset, want in (("D1_154_building", 4.0), ("D2_111_nadir", 2.0)):
    check(f"constituents behind a common cell, {dataset}",
          _constituents[dataset]["median_per_configuration"], want, 0.01)
    check(f"constituents behind a common cell, sparsest at the cell, {dataset}",
          _constituents[dataset]["median_of_the_per_cell_minimum"], 1.0, 0.01)

# the vertical part of the camera-centre discrepancy against its horizontal part
_ratio = json.loads(
    (ANALYSIS / "vertical_to_horizontal_ratio.json").read_text(encoding="utf-8"))
check("smallest vertical to horizontal ratio, all configurations",
      _ratio["all_configurations"]["minimum"], 0.08, 0.005)
check("largest vertical to horizontal ratio, all configurations",
      _ratio["all_configurations"]["maximum"], 0.54, 0.005)
check("smallest vertical to horizontal ratio, the four drawn",
      _ratio["drawn_in_the_residual_figure"]["minimum"], 0.15, 0.005)
check("largest vertical to horizontal ratio, the four drawn",
      _ratio["drawn_in_the_residual_figure"]["maximum"], 0.31, 0.005)

# the two models the detector-free run of D1 returned
_split = json.loads(
    (ANALYSIS / "detector_free_model_split.json").read_text(encoding="utf-8"))
check("images in the larger detector-free model of D1", _split["larger"], 125, 0.5)
check("images in the smaller detector-free model of D1", _split["smaller"], 49, 0.5)
check("images of the smaller model absent from the larger",
      _split["only_in_the_smaller"], 29, 0.5)
check("images registered in neither model", 154 - _split["union"], 0, 0.5)

# what the difference-of-Gaussians detector loses when it carries a learned
# descriptor
_dog = d1["doghardnet-lightglue-r2048-k8192"]["classes"]["convergence"]
_root = d1["rootsift-lightglue-r2048-k8192"]["classes"]["convergence"]
check("DoG with a learned descriptor, lost between 25 and 35 degrees",
      _dog[3]["failed_pct"], 1.5, 0.05)
check("DoG with a learned descriptor, lost between 35 and 50 degrees",
      _dog[4]["failed_pct"], 6.2, 0.05)
for index, label in ((3, "25 to 35"), (4, "35 to 50")):
    check(f"RootSIFT lost between {label} degrees", _root[index]["failed_pct"],
          0.0, 0.001)

# in every class of the convergent block a detector-based configuration retains
# more than the detector-free one
for which in ("convergence", "baseline"):
    _ahead = []
    for index in range(len(d1["loftr-dense-r1024-kall"]["classes"][which])):
        _free = d1["loftr-dense-r1024-kall"]["classes"][which][index]
        if not _free["count"]:
            continue
        _best = min(v["classes"][which][index]["failed_pct"]
                    for k, v in d1.items() if "loftr" not in k)
        _ahead.append(_free["failed_pct"] < _best)
    same(f"a detector-based configuration retains more in every {which} class "
         "of D1", any(_ahead), False)

# what the image metadata records of the two flights
_acq = {d: meta[d]["acquisition"] for d in ("D1_154_building", "D2_111_nadir")}
for dataset, images, date in (("D1_154_building", 154, "2023-09-26"),
                              ("D2_111_nadir", 111, "2022-06-08")):
    entry = _acq[dataset]
    check(f"images carrying metadata, {dataset}", entry["images"], images, 0.5)
    same(f"acquisition date, {dataset}", entry["first_exposure"][:10], date)
    same(f"acquisition ends on the same day, {dataset}",
         entry["last_exposure"][:10], date)
    same(f"one camera body across the block, {dataset}",
         entry["body_serial_constant"], True)
    same(f"one platform across the block, {dataset}",
         entry["platform_identifier_constant"], True)
    check(f"images accounted for by the gimbal settings, {dataset}",
          sum(entry["gimbal_pitch_deg"].values()), images, 0.5)
same("the same camera body flew both blocks",
     _acq["D1_154_building"]["body_serial"]
     == _acq["D2_111_nadir"]["body_serial"], True)
same("the same platform flew both blocks",
     _acq["D1_154_building"]["platform_identifier"]
     == _acq["D2_111_nadir"]["platform_identifier"], True)
same("the two flights differ in firmware",
     _acq["D1_154_building"]["camera_firmware"]
     != _acq["D2_111_nadir"]["camera_firmware"], True)
# the nine tilted images of the near-nadir block, as the gimbal recorded them
_tilted = sum(count for angle, count in _acq["D2_111_nadir"]["gimbal_pitch_deg"].items()
              if abs(float(angle) + 45.0) < 1.0)
check("tilted images in the near-nadir block", _tilted, 9, 0.5)
_near_nadir = sum(count for angle, count in _acq["D1_154_building"]["gimbal_pitch_deg"].items()
                  if abs(float(angle) + 90.0) < 1.0)
check("near-nadir images in the convergent block", _near_nadir, 31, 0.5)
check("distance between the two blocks",
      meta["block_separation"]["distance_m"] / 1000.0, 3.8, 0.05)

# the OpenDroneMap release that computed both reference solutions
_history = {d: meta[d]["processing_history"] for d in ("D1_154_building",
                                                       "D2_111_nadir")}
for dataset, cell in (("D1_154_building", 0.0874), ("D2_111_nadir", 0.1041)):
    same(f"reference solution computed by OpenDroneMap 3.6.2, {dataset}",
         _history[dataset]["project_last_saved_version"], "3.6.2")
    check(f"reference elevation cell size, {dataset}",
          _history[dataset]["elevation_resolution_m"], cell, 0.0001)

# recovery of the two most convergent classes of the convergent block
_d1 = support["D1_154_building"]
_conv = {k: v["classes"]["convergence"] for k, v in _d1.items()}
# RootSIFT under either matcher loses none of the two most convergent classes
for key in ("rootsift-lightglue-r2048-k8192", "rootsift-nn_ratio-r2048-k8192"):
    for index, label in ((3, "25 to 35"), (4, "35 to 50")):
        check(f"{key} lost between {label} degrees",
              _conv[key][index]["failed_pct"], 0.0, 0.001)
# and more than one configuration recovers every pair in both
same("more than one configuration recovers every pair in both classes",
     sum(1 for k, c in _conv.items()
         if c[3]["failed_pct"] == 0.0 and c[4]["failed_pct"] == 0.0) > 1, True)

# the three difference-of-Gaussians configurations hold the three smallest
# camera-centre discrepancies in every block and at both extraction resolutions
_dog = ("rootsift-lightglue", "rootsift-nn_ratio", "doghardnet-lightglue")
for _dataset in ("D1_154_building", "D2_111_nadir"):
    for _res in ("r1024", "r2048"):
        _ranked = []
        for _path in sorted(
                (RECORDS / "evaluation" / _dataset)
                .glob(f"*-{_res}-k8192.json")):
            if "__seed" in _path.stem:
                continue
            _value = json.loads(_path.read_text(encoding="utf-8")).get(
                "similarity_held_out", {}).get("rmse")
            if _value:
                _ranked.append((_value, _path.stem))
        _ranked.sort()
        _top = {name.rsplit("-", 2)[0] for _, name in _ranked[:3]}
        same(f"the three smallest discrepancies are difference-of-Gaussians, "
             f"{_dataset} at {_res}", _top <= set(_dog), True)

# the extremes over the whole sweep, beside the values at the adopted setting
_sweep = json.loads((ANALYSIS / "family_scope_convergence.json").read_text(
    encoding="utf-8"))
check("worst difference-of-Gaussians loss between 35 and 50 degrees",
      _sweep["difference_of_gaussians"]["worst_35_50_pct"], 16.9, 0.05)
check("pairs the best jointly learned configuration recovers between 35 and 50",
      _sweep["jointly_learned"]["best_35_50_recovered"], 33, 0.5)

# the share of accepted pairs that miss the threshold, as one in n
for dataset, one_in in (("D1_154_building", 5.0), ("D2_111_nadir", 7.0)):
    precision = support[dataset]["loftr-dense-r1024-kall"]["precision"]
    check(f"accepted pairs of detector-free matching that miss the threshold, "
          f"{dataset}", 1.0 / (1.0 - precision), one_in, 0.5)

# the tilted subset of the near-nadir block, detector-free against the range
# of the detector-based methods
_tilted = {k: v["classes"]["convergence"][4]["supportable_pct"]
           for k, v in support["D2_111_nadir"].items()
           if v["classes"]["convergence"][4]["count"]}
_detector_based = [v for k, v in _tilted.items() if "loftr" not in k]
check("detector-free reference-consistent share in the tilted subset",
      _tilted["loftr-dense-r1024-kall"], 55.9, 0.05)
check("lowest detector-based share in the tilted subset",
      min(_detector_based), 55.9, 0.05)
check("highest detector-based share in the tilted subset",
      max(_detector_based), 58.5, 0.05)

# the footprint of the common aggregated set, after and before the gate
_mask = json.loads((ANALYSIS / "common_surface_mask.json").read_text(
    encoding="utf-8"))["summary"]
for dataset, after, before in (("D1_154_building", 0.69, 1.00),
                               ("D2_111_nadir", 0.79, 0.81)):
    check(f"footprint of the common aggregated set after the gate, {dataset}",
          _mask[dataset]["footprint_share_coarse_common"], after, 0.005)
    check(f"footprint of the common aggregated set before the gate, {dataset}",
          _mask[dataset]["footprint_share_coarse_common_before_gate"],
          before, 0.005)

# the sign of the relation between adjusted principal distance and the height
# offset of the delivered surface
for _key, _want in (("D2_111_nadir | without LoFTR", -1.0),
                    ("D2_111_nadir | all", -1.0)):
    same(f"the principal-distance relation is a reversal, {_key}",
         relation[_key]["spearman_c_vs_surface_median"] < 0, True)
    same(f"the fitted slope is negative, {_key}",
         relation[_key]["slope_m_per_px"] < 0, True)

# the delivered share of the requested keypoint limit, smallest and largest
_census = json.loads((RECORDS / "tables"
                      / "keypoint_census.json").read_text(encoding="utf-8"))
_shares = []
for _name, _entry in _census.items():
    if "rootsift" not in _name and "doghardnet" not in _name:
        continue
    _requested = _name.rsplit("-k", 1)[-1]
    if not _requested.isdigit():
        continue
    _shares.append(100.0 * _entry["median"] / int(_requested))
check("smallest delivered share of the requested keypoint limit",
      min(_shares), 68.0, 0.6)
check("largest delivered share of the requested keypoint limit",
      max(_shares), 78.0, 0.6)

bad = [c for c in checks if not c[0]]
for ok, label, got, want in checks:
    if not ok:
        print(f"  MISMATCH  {label}: article says {want}, record gives {got}")
print(f"{len(checks) - len(bad)} of {len(checks)} quoted values reproduce")
