# Image matching and pair geometry in convergent and near-nadir UAV blocks

Code and data for the article

> A. Janowski, M. Hüsrevoğlu, A. E. Karkınlı. *Learned and hand-crafted image
> matching in convergent and near-nadir unmanned aerial vehicle blocks.*
> Submitted to The Photogrammetric Record.

Six methods of establishing image correspondence (RootSIFT, DoG+HardNet,
SuperPoint, ALIKED and DISK with LightGlue, and the detector-free LoFTR) are
compared on two DJI Phantom 4 RTK blocks under one candidate pair list per
block, one geometric verification and one orientation procedure. Every run is
evaluated against a reference solution computed with OpenDroneMap. The
repository holds the imagery, the pipeline, the reference solutions, the record
written by every run, the derived analyses and the scripts that draw the
figures of the article.

## Layout

    scripts/                 the pipeline, numbered in the order it runs
    scripts/odm/             the reference solution: OpenDroneMap run, export,
                             and the measurement of the surveyed targets
    scripts/figures/         one script per figure of the article
    D1_154_building/, D2_111_nadir/
      D1_images/, D2_images/ the 154 and 111 images
      RTK_data/              the on-board GNSS logs and the timestamp file
      exports/               the reference solution in the form the pipeline
                             reads: camera poses, calibration, surface model,
                             orthophoto, tie points, and for D1 the surveyed
                             targets with their measurements in the imagery
    odm_runs/D1/, odm_runs/D2/
                             the OpenDroneMap projects, reduced to their
                             settings and log, the oriented block, the surface
                             model and the processing report
    results/parsed_inputs/   exposure record of every image
    results/pairs/           candidate pair list of each block
    results/manifests/       settings of every matching run
    results/pairwise/        one row per candidate image pair and run
    results/sfm_reports/     sparse reconstruction report of every run
    results/evaluation/      camera-network assessment of every run
    results/reference_geometry/  reference poses and pair geometry of both blocks
    results/calibration/     measured runtime and memory of each method
    results/tables/          result tables and the surface summary
    results/surface/         signed height-difference raster of every surface comparison
    results/analysis/        reference-consistency, on-board position, interior
                             orientation, common-set and surface-gate analyses,
                             the contact sheet of the target measurements and
                             the image pair, stations and terrain drawn in the
                             graphical abstract
    results/figures/         the figures as printed

## Order of the pipeline

1. `prepare_inputs.py` reads the timestamp files and writes the exposure
   records and the candidate pair lists.
2. `odm/run_odm.py D1` and `D2` compute the reference solutions with
   OpenDroneMap; `odm/odm_to_exports.py` writes them into `exports/`, and
   `odm/measure_targets.py` measures the surveyed targets in the imagery.
3. `01_reference_geometry.py` derives the reference poses and pair geometry.
4. `00_calibrate_operating_points.py` and `02`–`14` extract features, match,
   reconstruct, assess the camera network and run dense matching;
   `09r_surface_against_reference.py` compares the dense surfaces with the
   reference surface model.
5. `22`–`27`, `31`, `33` and `36` are the analyses reported in the article,
   `10_build_tables.py` writes the tables and `scripts/figures/` the figures.
6. `28_verify_reported_values.py` checks every value quoted in the article
   against the records.

## What runs from the repository alone

| Script | Output |
|---|---|
| `prepare_inputs.py` | `results/parsed_inputs/`, `results/pairs/` |
| `odm/odm_to_exports.py D1`, `D2` | `exports/` of each block, from `odm_runs/` |
| `odm/measure_targets.py` | `D1_154_building/exports/target_measurements.csv` |
| `01_reference_geometry.py` | `results/reference_geometry/` |
| `10_build_tables.py` | `results/tables/` |
| `23_pair_support_summary.py` | reference-consistency summary, intervals, half-loss angles |
| `24_interior_orientation.py` | principal distance against surface height offset |
| `25_onboard_decomposition.py` | on-board positions against reference centres |
| `26_common_sets.py` | comparisons on shared cameras and at 1024 pixels |
| `27_common_surface_mask.py` | surface comparison on the common aggregated set |
| `30_test_reference_epipolar.py` | synthetic check of the reference essential matrix |
| `31_onboard_metadata.py` | on-board positions, target check and acquisition metadata |
| `33_surface_gate_sensitivity.py` | surface statistics at other gate thresholds |
| `28_verify_reported_values.py` | checks every value quoted in the article against the records |
| `scripts/figures/*.py` | `results/figures/` |

For example:

    python scripts/28_verify_reported_values.py
    python scripts/figures/fig03_pair_stratification.py

Feature extraction, matching, reconstruction and dense matching (`00`, `02`–`09`,
`09r`, `11`–`14`, `22` and the model-split part of `36`) write feature, match and
reconstruction stores of several tens of gigabytes to `workspace/`, or wherever
the `DEEPMATCHING_WORKSPACE` environment variable points, and are rerun from the
imagery. Running OpenDroneMap again writes its full projects, dense clouds
included, to `odm_runs/` or wherever `ODM_PROJECTS` points.

## Environment

The results were produced with Python 3.14 and the packages in
`requirements.txt`. Matching uses
[LightGlue](https://github.com/cvg/LightGlue), detector-free matching uses
[Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization),
and reconstruction uses pycolmap 4.0.4 with COLMAP. The reference solutions were
computed with [OpenDroneMap](https://github.com/OpenDroneMap/ODM) 3.6.2 from its
Docker image. Each run records the PyTorch and OpenCV versions and the graphics
device in its manifest.

## Figures

| Script | Figure of the article |
|---|---|
| `fig01_block_geometry.py` | Figure 1 |
| `fig02_operating_point.py` | Figure 2 |
| `fig03_pair_stratification.py` | Figure 3 |
| `fig04_reference_dependence.py` | Figures 4 and 5 |
| `fig05_residual_fields.py` | Figure 6 |
| `fig06_surface_maps.py` | Figure 7 |
| `fig07_repeatability_propagation.py` | Figure 8 |
| `figS_supporting.py` | Figures S1 and S2 |
| `fig00_graphical_abstract.py` | graphical abstract |

## Size

The repository is about 2.6 GB, most of it the imagery; no file exceeds 100 MB.
Pushing it in more than one commit, for example the imagery separately, keeps
each push within the size GitHub accepts.

## Licence

The code is released under the MIT licence (`LICENSE`). The imagery and the data
are released under the Creative Commons Attribution 4.0 International licence.

## Citation

Please cite the article; `CITATION.cff` gives the reference in machine-readable
form.
