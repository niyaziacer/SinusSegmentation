"""Shared DICOM-series loader for the local (non-Slicer) helper scripts."""

from pathlib import Path

import numpy as np
import pydicom


def load_series(dicom_dir: str):
    """Read every DICOM image file in dicom_dir (any/no extension) into a
    single HU volume, sorted by slice position.

    Returns (volume, spacing_mm, first_dataset) where volume has axes
    (z, y, x) and spacing_mm is (spacing_z, spacing_y, spacing_x).
    """
    candidates = sorted(p for p in Path(dicom_dir).iterdir() if p.is_file())
    if not candidates:
        raise SystemExit(f"No files found in {dicom_dir}")

    datasets = []
    skipped = 0
    for p in candidates:
        try:
            ds = pydicom.dcmread(str(p))
            ds.pixel_array  # force-decode to catch non-image files (e.g. DICOMDIR)
        except Exception:
            skipped += 1
            continue
        datasets.append(ds)

    if not datasets:
        raise SystemExit(f"No readable DICOM images found in {dicom_dir}")
    if skipped:
        print(f"(skipped {skipped} non-image file(s) in {dicom_dir})")

    datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))

    rows, cols = datasets[0].Rows, datasets[0].Columns
    volume = np.zeros((len(datasets), rows, cols), dtype=np.float32)

    slope = float(getattr(datasets[0], "RescaleSlope", 1.0))
    intercept = float(getattr(datasets[0], "RescaleIntercept", 0.0))

    for i, ds in enumerate(datasets):
        volume[i] = ds.pixel_array.astype(np.float32) * slope + intercept

    px_spacing = [float(v) for v in datasets[0].PixelSpacing]
    z_positions = [float(ds.ImagePositionPatient[2]) for ds in datasets]
    slice_spacing = float(np.median(np.diff(z_positions))) if len(z_positions) > 1 else 1.0
    spacing_mm = (abs(slice_spacing), px_spacing[0], px_spacing[1])

    return volume, spacing_mm, datasets[0]
