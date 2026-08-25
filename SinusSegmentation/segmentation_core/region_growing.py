"""Seeded, bounded region growing for air-filled paranasal sinus cavities.

Pure numpy/scipy implementation with no Slicer/VTK dependency, so it can be
unit-tested outside 3D Slicer (see scripts/test_region_growing.py).

Paranasal sinuses connect to the nasal cavity/exterior air through a small
ostium. A plain HU-threshold + connected-component pass on the whole volume
leaks through that ostium and merges the sinus with all exterior air into one
component. This module avoids that geometrically: growing only happens
inside a small crop around the seed point, so it can never reach exterior
air in the first place. A morphological opening additionally severs any
thin neck that survives within the crop before labeling -- but only as much
opening as is actually needed (see segment_region), since real CT air/bone
boundaries are irregular at the voxel scale and every extra bit of opening
erodes true boundary detail that a uniform-radius re-dilation can't recover.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .anatomy import (
    DEFAULT_HU_RANGE,
    DEFAULT_MIN_SIZE_VOXELS,
    DEFAULT_OPENING_RADIUS_VOX,
    DEFAULT_LEAK_VOLUME_CM3,
)

# How far (mm) to search for a nearby air voxel if the seed itself landed on
# bone/mucosa rather than inside the air cavity.
SEED_SEARCH_RADIUS_MM = 5.0


@dataclass
class RegionGrowingResult:
    mask: np.ndarray  # bool array, same shape as the input volume
    success: bool
    reason: Optional[str]  # None on success; else a short machine-readable code
    seed_index_used: Optional[Tuple[int, int, int]]
    bbox: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]  # (lo, hi), hi exclusive
    volume_voxels: int = 0
    volume_mm3: float = 0.0
    volume_cm3: float = 0.0


def _crop_bounds(shape: Sequence[int], seed_index: Sequence[int],
                  spacing_mm: Sequence[float], crop_radius_mm: float) -> Tuple[tuple, tuple]:
    lo = []
    hi = []
    for axis in range(3):
        radius_vox = max(1, int(round(crop_radius_mm / spacing_mm[axis])))
        lo.append(max(0, seed_index[axis] - radius_vox))
        hi.append(min(shape[axis], seed_index[axis] + radius_vox + 1))
    return tuple(lo), tuple(hi)


def _nearest_air_voxel(air_mask: np.ndarray, seed_local: Sequence[int],
                        spacing_mm: Sequence[float], max_search_mm: float) -> Optional[Tuple[int, int, int]]:
    coords = np.argwhere(air_mask)
    if coords.size == 0:
        return None
    diffs = (coords - np.array(seed_local)) * np.array(spacing_mm)
    dists = np.linalg.norm(diffs, axis=1)
    best = np.argmin(dists)
    if dists[best] <= max_search_mm:
        return tuple(int(v) for v in coords[best])
    return None


def _component_at_opening(air_mask, seed_local, opening_radius):
    """Applies plain morphological opening (erosion then dilation by the same
    amount -- NOT reconstruction/propagation, which would just flood right
    back through whatever neck the erosion severed) and returns the labeled
    component touching the seed, or None if the seed isn't in it at this
    opening radius."""
    if opening_radius == 0:
        opened = air_mask
    else:
        opened = ndimage.binary_opening(air_mask, iterations=opening_radius)
    if not opened[seed_local]:
        return None
    labeled, _ = ndimage.label(opened, structure=np.ones((3, 3, 3), dtype=int))
    component_label = labeled[seed_local]
    if component_label == 0:
        return None
    return labeled == component_label


def segment_region(
    volume_hu: np.ndarray,
    spacing_mm: Sequence[float],
    seed_index: Sequence[int],
    hu_range: Tuple[float, float] = DEFAULT_HU_RANGE,
    crop_radius_mm: float = 25.0,
    min_size_voxels: int = DEFAULT_MIN_SIZE_VOXELS,
    opening_radius_vox: int = DEFAULT_OPENING_RADIUS_VOX,
    leak_volume_cm3: float = DEFAULT_LEAK_VOLUME_CM3,
) -> RegionGrowingResult:
    """Segment one air-filled cavity around a seed voxel.

    volume_hu: 3D numpy array of Hounsfield units.
    spacing_mm: per-axis voxel spacing (mm), in the same axis order as volume_hu.
    seed_index: integer (a0, a1, a2) index into volume_hu, in the same axis order.

    Escalates morphological opening one voxel at a time, from none up to
    opening_radius_vox, stopping as soon as a radius no longer looks like a
    leak (grown volume under leak_volume_cm3). This keeps the most true
    volume for the (common) case where the crop alone already prevents
    leaking, while still cutting a real thin connection (e.g. a patent
    ostium) when a gentler radius doesn't stop it.
    """
    shape = volume_hu.shape
    seed_index = tuple(int(v) for v in seed_index)

    if any(not (0 <= seed_index[a] < shape[a]) for a in range(3)):
        return RegionGrowingResult(
            mask=np.zeros(shape, dtype=bool), success=False,
            reason="seed_outside_bounds", seed_index_used=None, bbox=None,
        )

    lo, hi = _crop_bounds(shape, seed_index, spacing_mm, crop_radius_mm)
    crop_slices = tuple(slice(lo[a], hi[a]) for a in range(3))
    crop = volume_hu[crop_slices]
    seed_local = tuple(seed_index[a] - lo[a] for a in range(3))

    air_mask = (crop >= hu_range[0]) & (crop <= hu_range[1])
    voxel_volume_mm3 = float(np.prod(spacing_mm))

    if not air_mask[seed_local]:
        found = _nearest_air_voxel(air_mask, seed_local, spacing_mm, SEED_SEARCH_RADIUS_MM)
        if found is None:
            return RegionGrowingResult(
                mask=np.zeros(shape, dtype=bool), success=False,
                reason="seed_not_in_air",
                seed_index_used=tuple(seed_index), bbox=(lo, hi),
            )
        seed_local = found

    best_filled = None
    best_voxels = 0
    best_reason = None

    for radius in range(0, opening_radius_vox + 1):
        component_mask = _component_at_opening(air_mask, seed_local, radius)
        if component_mask is None:
            continue

        raw_voxels = int(component_mask.sum())
        if raw_voxels < min_size_voxels:
            if best_filled is None:
                best_reason = "too_small"
                best_voxels = raw_voxels
            continue

        closed = ndimage.binary_closing(component_mask, iterations=radius + 1)
        filled = ndimage.binary_fill_holes(closed)
        voxels = int(filled.sum())
        volume_cm3 = voxels * voxel_volume_mm3 / 1000.0

        best_filled = filled
        best_voxels = voxels
        best_reason = "possible_leak" if volume_cm3 > leak_volume_cm3 else None
        if best_reason is None:
            break  # gentlest non-leaking opening found -- stop eroding further

    if best_filled is None:
        return RegionGrowingResult(
            mask=np.zeros(shape, dtype=bool), success=False,
            reason=best_reason or "seed_not_in_air",
            seed_index_used=tuple(lo[a] + seed_local[a] for a in range(3)), bbox=(lo, hi),
            volume_voxels=best_voxels,
            volume_mm3=best_voxels * voxel_volume_mm3,
            volume_cm3=best_voxels * voxel_volume_mm3 / 1000.0,
        )

    mask_full = np.zeros(shape, dtype=bool)
    mask_full[crop_slices] = best_filled
    volume_mm3 = best_voxels * voxel_volume_mm3
    volume_cm3 = volume_mm3 / 1000.0

    return RegionGrowingResult(
        mask=mask_full,
        success=best_reason is None,
        reason=best_reason,
        seed_index_used=tuple(lo[a] + seed_local[a] for a in range(3)),
        bbox=(lo, hi),
        volume_voxels=best_voxels,
        volume_mm3=volume_mm3,
        volume_cm3=volume_cm3,
    )
