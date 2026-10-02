#!/usr/bin/env python3
"""Shared output locations for the controlled matching experiment.

Small, durable outputs stay with the project so that they travel with the code:
run manifests, per-pair metric tables, reconstruction reports and the reference
geometry. Bulk intermediates that can be regenerated from those, and that grow
to tens of gigabytes, live in a separate workspace directory on a volume with
room for them.

The workspace location can be overridden with the DEEPMATCHING_WORKSPACE
environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Durable, small outputs kept inside the project.
RESULTS = PROJECT_ROOT / "results"

# Bulk intermediates. Regenerable, so they do not need to sit next to the code.
DEFAULT_WORKSPACE = PROJECT_ROOT / "workspace"
WORKSPACE = Path(os.environ.get("DEEPMATCHING_WORKSPACE", DEFAULT_WORKSPACE))

FEATURES_DIR = WORKSPACE / "features"
MATCHES_DIR = WORKSPACE / "matches"
SFM_DIR = WORKSPACE / "sfm"
DENSE_DIR = WORKSPACE / "dense"

# Small outputs, inside the project.
MANIFESTS_DIR = RESULTS / "manifests"
PAIRWISE_DIR = RESULTS / "pairwise"
SFM_REPORTS_DIR = RESULTS / "sfm_reports"
REFERENCE_DIR = RESULTS / "reference_geometry"
CALIBRATION_DIR = RESULTS / "calibration"
EVALUATION_DIR = RESULTS / "evaluation"


def ensure_workspace() -> None:
    for directory in (FEATURES_DIR, MATCHES_DIR, SFM_DIR, DENSE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for directory in (
        MANIFESTS_DIR,
        PAIRWISE_DIR,
        SFM_REPORTS_DIR,
        REFERENCE_DIR,
        EVALUATION_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
