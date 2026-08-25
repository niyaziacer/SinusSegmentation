#!/usr/bin/env python3
"""Local, Slicer-free unit test for segmentation_core.region_growing.

Builds a synthetic phantom (a bone shell around an air cavity, plus a thin
ostium channel leaking to "exterior" air) and checks that segment_region
recovers a volume close to the analytic sphere volume without leaking.

Run with: python3 scripts/test_region_growing.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SinusSegmentation"))

from segmentation_core.region_growing import segment_region  # noqa: E402

SPACING_MM = (0.5, 0.5, 0.5)
SHAPE = (80, 80, 80)
CENTER = (40, 40, 40)
CAVITY_RADIUS_VOX = 10  # -> analytic volume = 4/3 * pi * (10*0.5mm)^3
BONE_HU = 700.0
AIR_HU = -900.0
TISSUE_HU = 40.0


def build_phantom():
    volume = np.full(SHAPE, TISSUE_HU, dtype=np.float32)
    zz, yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1], 0:SHAPE[2]]
    dist = np.sqrt((zz - CENTER[0]) ** 2 + (yy - CENTER[1]) ** 2 + (xx - CENTER[2]) ** 2)

    bone_shell = (dist >= CAVITY_RADIUS_VOX) & (dist < CAVITY_RADIUS_VOX + 3)
    volume[bone_shell] = BONE_HU

    cavity = dist < CAVITY_RADIUS_VOX
    volume[cavity] = AIR_HU

    # Exterior air, far from the cavity, fills the rest of the volume outside
    # a large "head" region -- if the algorithm ever leaked out this far
    # (crop radius is only 25mm = 50 voxels at 0.5mm spacing, well inside the
    # 80-voxel volume) the resulting volume would be enormous.
    exterior = dist > 35
    volume[exterior] = AIR_HU

    # A thin one-voxel-wide ostium channel connecting the cavity to exterior
    # air, to prove the morphological opening severs it.
    channel = np.zeros(SHAPE, dtype=bool)
    channel[CENTER[0], CENTER[1], CAVITY_RADIUS_VOX - 1:SHAPE[2]] = True
    volume[channel] = AIR_HU

    return volume


def analytic_volume_cm3():
    radius_mm = CAVITY_RADIUS_VOX * SPACING_MM[0]
    volume_mm3 = 4.0 / 3.0 * math.pi * radius_mm ** 3
    return volume_mm3 / 1000.0


def main():
    volume = build_phantom()
    expected_cm3 = analytic_volume_cm3()

    result = segment_region(
        volume_hu=volume,
        spacing_mm=SPACING_MM,
        seed_index=CENTER,
        crop_radius_mm=25.0,
        min_size_voxels=1000,
        opening_radius_vox=1,
    )

    print(f"success={result.success} reason={result.reason}")
    print(f"voxels={result.volume_voxels} volume_cm3={result.volume_cm3:.3f} "
          f"expected_cm3={expected_cm3:.3f}")

    assert result.success, f"segmentation failed: {result.reason}"

    rel_error = abs(result.volume_cm3 - expected_cm3) / expected_cm3
    print(f"relative_error={rel_error:.3%}")
    assert rel_error < 0.10, "computed volume too far from analytic sphere volume"

    # Leak guard: the mask must stay inside the local crop, never reach the
    # "exterior air" region far from the seed.
    exterior_touch = result.mask & (
        np.sqrt(
            (np.arange(SHAPE[0])[:, None, None] - CENTER[0]) ** 2
            + (np.arange(SHAPE[1])[None, :, None] - CENTER[1]) ** 2
            + (np.arange(SHAPE[2])[None, None, :] - CENTER[2]) ** 2
        )
        > 35
    )
    assert not exterior_touch.any(), "segmentation leaked into exterior air"

    print("OK: synthetic phantom test passed")


if __name__ == "__main__":
    main()
