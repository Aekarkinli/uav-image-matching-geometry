#!/usr/bin/env python3
"""Supporting figures, produced by the same code as the published ones.

The two figures that map object-space quantities show a selected subset of the
front ends, because a full set at journal width would not be legible. The
configurations left out are not omitted from the study, so they are drawn here
with identical settings and identical scales, and referred to from the article.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fig05_residual_fields as residual  # noqa: E402
import fig06_surface_maps as surface  # noqa: E402


def main() -> int:
    print("S1, camera-position residual fields not shown in the residual-field figure")
    residual.main(columns=["doghardnet", "loftr"], name="figS1_residual_fields")

    print("S2, surface difference maps not shown in the surface-difference figure")
    surface.main(columns=[("doghardnet", "doghardnet-lightglue-r2048-k8192"),
                          ("disk", "disk-lightglue-r2048-k8192"),
                          ("loftr", "loftr-dense-r1024-kall")],
                 name="figS2_surface_maps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
