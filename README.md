# SinusSegmentation

A [3D Slicer](https://www.slicer.org/) scripted extension that segments the
paranasal sinuses (maxillary, frontal, sphenoid, ethmoid — right/left) from a
head CT scan, using seed-point-assisted region growing and morphology, and
computes per-sinus volume (cm³) and surface area (mm²).

## Why not plain thresholding?

Paranasal sinuses are air-filled cavities connected to the nasal cavity (and
therefore exterior air) through a small opening, the ostium. A naive
HU-threshold + connected-component pass leaks through the ostium and merges
the sinus with all the air around the patient's head into one giant blob —
exactly the problem you run into doing this by hand in Segment Editor with
Threshold + Islands. This extension avoids that geometrically: growing is
bounded to a small crop around each seed point (so it can never reach
exterior air), and a morphological opening severs any thin neck that
survives inside the crop, before the region is kept and smoothed.

## Algorithm

For each sinus, given one seed point placed inside its air cavity:

1. Crop a local sub-volume around the seed (25 mm radius for maxillary,
   frontal and sphenoid; 10 mm for ethmoid — tuned against a real patent-airway
   CT where 15 mm let ethmoid growth leak into the nasal cavity, see
   `scripts/find_candidates.py`).
2. Threshold the crop to an air HU range (default −1024…−300).
3. Morphological opening to sever thin ostium/nasal-cavity connections.
4. Connected-component labeling; keep only the component touching the seed.
5. Reject the result if it's smaller than a minimum size (default 1000
   voxels, matching manual "Islands" filtering) — reported as a failed seed
   rather than silently accepted.
6. Morphological closing + hole-filling to smooth the kept region.
7. Flag (don't silently accept) results larger than ~30 cm³ as a possible
   leak, as a belt-and-suspenders check on top of step 1's geometric guard.

The algorithm itself (`SinusSegmentation/segmentation_core/`) is plain
numpy/scipy with no Slicer dependency, so it's unit-testable outside Slicer —
see [Development](#development) below.

Volume and surface area are computed with Slicer's built-in
`SegmentStatistics` module (labelmap volume + closed-surface area plugins)
rather than a hand-rolled marching-cubes pass.

## Installation

1. Clone or download this repository.
2. In 3D Slicer: **Edit → Application Settings → Modules → Additional module paths**,
   add the `SinusSegmentation/SinusSegmentation` folder (the one containing
   `SinusSegmentation.py`), and restart Slicer.
   (Alternatively, use the **Extension Wizard** module → *Select Extension* →
   pick the repository's `SinusSegmentation` folder.)
3. The module appears under the **Segmentation** category as "Sinus Segmentation".

## Usage

1. Load your CT volume.
2. Open the **Sinus Segmentation** module.
3. Select the input volume (and optionally an existing segmentation node;
   otherwise a new one is created).
4. For each sinus you want to segment, click **Tohum yerleştir** ("Place
   seed") and click once inside that sinus's air cavity in a slice view.
5. Click **Segmentasyonu Çalıştır** ("Run Segmentation").
6. Per-sinus volume and surface area appear in the results table; use
   **CSV olarak dışa aktar** to export them.

Default thresholds/parameters are under **Gelişmiş Parametreler** (Advanced).

## Development

```bash
# Pure-algorithm unit test (synthetic phantom, no Slicer needed):
python3 scripts/test_region_growing.py

# Sanity-check a real DICOM series loads into a plausible HU volume:
python3 scripts/local_validate.py /path/to/dicom/dir
```

Inside Slicer, the module's own self-test (Extension Wizard's "Reload and
Test", or `slicer.modules.sinussegmentation.selfTest()`) builds a synthetic
phantom and checks the full node round-trip end to end.

## Roadmap

- Publish to the Slicer Extensions Manager catalog (needs an `s4ext`
  descriptor and extension-index CI build — intentionally left out of v1 to
  keep local development simple; for now, load the module locally as
  described above).
- Per-region threshold/crop-radius overrides in the GUI (currently only
  global Advanced parameters).
- Optional automatic seed suggestion, as an alternative to manual placement.

## License

MIT — see [LICENSE](LICENSE).
