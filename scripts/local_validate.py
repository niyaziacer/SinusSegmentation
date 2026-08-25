#!/usr/bin/env python3
"""Sanity-check that the user's real DICOM series loads into a coherent HU
volume before any of this is tried inside 3D Slicer.

No seed coordinates are needed here -- seeds are placed interactively inside
the running Slicer module. This script only exercises the DICOM loading path
and reports numbers a human can sanity-check (shape, spacing, HU range).

Usage: python3 scripts/local_validate.py [dicom_dir]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dicom_io import load_series  # noqa: E402

DEFAULT_DICOM_DIR = "/mnt/c/Users/LENOVO/Desktop/3d_slicer_kurs/ct_paranasal"


def main():
    dicom_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DICOM_DIR
    volume, spacing_mm, first_ds = load_series(dicom_dir)

    # Real scanner data commonly has a handful of single-pixel saturation
    # artifacts (metal/dental restorations reading as the int16 max, 32767)
    # and an out-of-reconstruction-field padding value (often -2048). Neither
    # is a loading bug, so report percentiles rather than raw min/max for the
    # sanity checks -- they're robust to a few outlier pixels.
    p01, p50, p999 = np.percentile(volume, [0.1, 50, 99.9])

    print(f"dicom_dir           = {dicom_dir}")
    print(f"num_slices          = {volume.shape[0]}")
    print(f"volume_shape (z,y,x)= {volume.shape}")
    print(f"spacing_mm (z,y,x)  = {spacing_mm}")
    print(f"hu_min / hu_max (raw) = {volume.min():.1f} / {volume.max():.1f}")
    print(f"hu_p0.1 / hu_median / hu_p99.9 = {p01:.1f} / {p50:.1f} / {p999:.1f}")

    air_voxels = int(((volume >= -1024) & (volume <= -300)).sum())
    bone_voxels = int((volume >= 300).sum())
    saturated_voxels = int((volume >= 32767).sum())
    print(f"voxels_in_air_range(-1024..-300)  = {air_voxels}")
    print(f"voxels_bone_like(>=300 HU)        = {bone_voxels}")
    print(f"voxels_saturated(>=32767 HU)      = {saturated_voxels} "
          f"({100 * saturated_voxels / volume.size:.4f}% -- expected: tiny, metal/dental artifact)")

    print(f"PatientPosition     = {getattr(first_ds, 'PatientPosition', 'n/a')}")
    print(f"Modality            = {getattr(first_ds, 'Modality', 'n/a')}")

    # p0.1 is intentionally not checked against a tight range: out-of-field
    # padding values vary by scanner vendor (seen so far: -2048 and -3024),
    # neither of which is a loading bug. The median is a more robust check
    # that we're looking at real air/soft-tissue HU values, not garbage.
    assert volume.shape[0] > 50, "unexpectedly few slices for a head CT"
    assert -1100 < p50 < 200, "median HU outside expected air/soft-tissue range"
    assert p999 > 100, "high-percentile HU too low for a CT containing bone"
    assert air_voxels > 10000, "very little air-range tissue found -- check rescale slope/intercept"

    print("OK: DICOM series loads into a plausible HU volume")


if __name__ == "__main__":
    main()
