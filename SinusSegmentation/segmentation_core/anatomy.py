"""Static table describing the paranasal sinus regions this module segments.

Kept separate from the algorithm and from Slicer glue code so both can
reference the same names, colors and default parameters.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SinusRegion:
    id: str  # stable key, e.g. "right_maxillary"
    name_tr: str  # matches the naming the user already used manually in Slicer
    name_en: str
    label_value: int  # segment label value used in the multi-label export
    color_rgb: tuple  # 0..1 floats, for the Segmentation node display
    crop_radius_mm: float  # local region-growing crop radius around the seed
    enabled_by_default: bool
    # Per-region "this looks like a leak" cutoff (cm3): a single global cutoff
    # is too loose for smaller sinuses -- e.g. sphenoid leaking down into the
    # nasopharynx via the sphenoethmoidal recess landed at ~17-21 cm3, well
    # under a generic 30 cm3 cutoff, even though real sphenoid volume is
    # rarely above ~10 cm3. Tighter numbers here make the escalating-opening
    # loop actually try harder (or flag possible_leak) instead of accepting
    # a plausible-looking but wrong volume.
    leak_volume_cm3: float


SINUS_REGIONS = (
    SinusRegion("right_maxillary", "sağ sinus maxillaris", "right maxillary sinus",
                1, (0.20, 0.60, 0.20), 25.0, True, 35.0),
    SinusRegion("left_maxillary", "sol sinus maxillaris", "left maxillary sinus",
                2, (0.90, 0.80, 0.30), 25.0, True, 35.0),
    SinusRegion("right_frontal", "sağ sinus frontalis", "right frontal sinus",
                3, (0.75, 0.30, 0.20), 25.0, True, 20.0),
    SinusRegion("left_frontal", "sol sinus frontalis", "left frontal sinus",
                4, (0.75, 0.35, 0.30), 25.0, True, 20.0),
    SinusRegion("right_sphenoid", "sağ sinus sphenoidalis", "right sphenoid sinus",
                5, (0.20, 0.55, 0.75), 18.0, True, 12.0),
    SinusRegion("left_sphenoid", "sol sinus sphenoidalis", "left sphenoid sinus",
                6, (0.30, 0.65, 0.85), 18.0, True, 12.0),
    SinusRegion("right_ethmoid", "sağ sinus ethmoidalis", "right ethmoid sinus",
                7, (0.65, 0.45, 0.75), 10.0, True, 10.0),
    SinusRegion("left_ethmoid", "sol sinus ethmoidalis", "left ethmoid sinus",
                8, (0.75, 0.55, 0.85), 10.0, True, 10.0),
)

# Default air-cavity Hounsfield range and morphology parameters.
DEFAULT_HU_RANGE = (-1024.0, -300.0)
DEFAULT_MIN_SIZE_VOXELS = 1000
# This is a ceiling, not a fixed amount applied to every region: segment_region
# escalates opening one voxel at a time and stops as soon as the result no
# longer looks like a leak, so raising this only helps genuinely patent/leaky
# cases without costing volume accuracy on well-contained ones.
DEFAULT_OPENING_RADIUS_VOX = 3
DEFAULT_LEAK_VOLUME_CM3 = 30.0


def region_by_id(region_id: str) -> SinusRegion:
    for region in SINUS_REGIONS:
        if region.id == region_id:
            return region
    raise KeyError(f"Unknown sinus region id: {region_id!r}")
