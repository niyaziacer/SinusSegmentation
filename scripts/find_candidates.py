#!/usr/bin/env python3
"""Explore a real DICOM volume for candidate air-filled cavities and run
segment_region on each one, as a dry run of the algorithm before trying it
in 3D Slicer (where seeds are placed by hand instead).

This has no ground-truth anatomy labels -- it just finds internal air
pockets (excluding the large exterior/background air component that touches
the image border) and reports how segment_region behaves when seeded at
each one's centroid. Useful to sanity-check the algorithm on real data and
to get approximate seed locations to try first inside Slicer.

Usage: python3 scripts/find_candidates.py [dicom_dir] [--png out.png]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SinusSegmentation"))

from _dicom_io import load_series  # noqa: E402
from segmentation_core.region_growing import segment_region  # noqa: E402

DEFAULT_DICOM_DIR = "/mnt/c/Users/LENOVO/Desktop/3d_slicer_kurs/ct_paranasal"
HU_RANGE = (-1024.0, -300.0)
MIN_CANDIDATE_VOXELS = 300
MAX_CANDIDATE_VOXELS = 60000


def find_candidates(volume_hu, spacing_mm):
    air_mask = (volume_hu >= HU_RANGE[0]) & (volume_hu <= HU_RANGE[1])
    labeled, num = ndimage.label(air_mask, structure=np.ones((3, 3, 3), dtype=int))

    # The exterior/background air is whichever component touches the volume
    # border; excluding it (and anything of similar size) leaves internal
    # cavities: sinuses, but also the trachea/nasal cavity/mouth -- this is
    # a rough filter, not a sinus classifier.
    border_labels = set()
    border_labels.update(np.unique(labeled[0, :, :]))
    border_labels.update(np.unique(labeled[-1, :, :]))
    border_labels.update(np.unique(labeled[:, 0, :]))
    border_labels.update(np.unique(labeled[:, -1, :]))
    border_labels.update(np.unique(labeled[:, :, 0]))
    border_labels.update(np.unique(labeled[:, :, -1]))
    border_labels.discard(0)

    sizes = ndimage.sum(np.ones_like(labeled), labeled, index=np.arange(1, num + 1))

    candidates = []
    for label_id in range(1, num + 1):
        if label_id in border_labels:
            continue
        size = int(sizes[label_id - 1])
        if not (MIN_CANDIDATE_VOXELS <= size <= MAX_CANDIDATE_VOXELS):
            continue
        centroid = ndimage.center_of_mass(labeled == label_id)
        candidates.append((size, tuple(int(round(c)) for c in centroid)))

    candidates.sort(key=lambda c: -c[0])
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dicom_dir", nargs="?", default=DEFAULT_DICOM_DIR)
    parser.add_argument("--top", type=int, default=12, help="how many largest candidates to test")
    parser.add_argument("--png", default=None, help="optional path to save an overlay figure")
    args = parser.parse_args()

    print(f"Loading {args.dicom_dir} ...")
    volume, spacing_mm, _ = load_series(args.dicom_dir)
    print(f"volume shape (z,y,x)={volume.shape}  spacing_mm (z,y,x)={spacing_mm}")

    print("Finding internal air-cavity candidates ...")
    candidates = find_candidates(volume, spacing_mm)
    print(f"found {len(candidates)} candidate air pockets in the "
          f"[{MIN_CANDIDATE_VOXELS}, {MAX_CANDIDATE_VOXELS}] voxel size range")

    top = candidates[: args.top]
    results = []
    print(f"\n{'#':>3} {'raw_vox':>8} {'seed(z,y,x)':>18} {'status':>10} {'vol_cm3':>8}  reason")
    for idx, (raw_size, seed) in enumerate(top):
        result = segment_region(
            volume_hu=volume,
            spacing_mm=spacing_mm,
            seed_index=seed,
            hu_range=HU_RANGE,
            crop_radius_mm=25.0,
            min_size_voxels=1000,
            opening_radius_vox=1,
        )
        status = "OK" if result.success else "FAIL"
        print(f"{idx:>3} {raw_size:>8} {str(seed):>18} {status:>10} "
              f"{result.volume_cm3:>8.2f}  {result.reason or ''}")
        results.append((seed, raw_size, result))

    if args.png:
        _save_overlay_png(volume, spacing_mm, results, args.png)
        print(f"\nSaved overlay figure to {args.png}")


def _save_overlay_png(volume, spacing_mm, results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    successful = [(seed, r) for seed, _, r in results if r.success]
    if not successful:
        print("(no successful candidates to plot)")
        return

    # Show every axial slice that contains at least one successful segment,
    # capped at a reasonable grid size.
    z_slices = sorted({seed[0] for seed, _ in successful})
    z_slices = z_slices[:12]

    n = len(z_slices)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    combined_mask = np.zeros(volume.shape, dtype=bool)
    for _, r in successful:
        combined_mask |= r.mask

    for ax, z in zip(axes, z_slices):
        ax.imshow(volume[z], cmap="gray", vmin=-1000, vmax=1000)
        overlay = np.ma.masked_where(~combined_mask[z], combined_mask[z])
        ax.imshow(overlay, cmap="autumn", alpha=0.5)
        ax.set_title(f"z={z}")
        ax.axis("off")

    for ax in axes[len(z_slices):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


if __name__ == "__main__":
    main()
