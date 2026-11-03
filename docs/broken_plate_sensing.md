# Broken-plate stationary sensing study

The real broken plate can be segmented in suitable views, but is **not yet an
accepted runtime CRA object**. Preserve the original failures and accepted v2.
No wheels, navigation acceptance, main merge or primary-controller change.

## Captures and reproducible diagnostic

Frozen native runtime `4a35d39` captured three new 18-source stationary runs
with the original 28 cm plate mesh (2.60 cm high), unchanged LiDAR profile/mount,
Owl camera raised .60 m, unchanged lens, and native 1920 x 1200 RGB sampling.
The new `--stationary-prop-rgb-width` option defaults to the old 640 width;
1280/1920 are guarded to stationary restaurant prop tests without humans/wheels.
Actual calibration is re-read for the chosen output size. This is real rendered
sampling, not upscaled old images, a new camera or an environment upgrade.

All three native component gates FAILED `target_uniquely_observed`: the original
floor filter still removes returns below 5 cm. Their expected guard exceptions,
18 complete fields, zero wheel commands and unchanged source hashes are saved.
These are completed failed tests, not successful perception gates.

The separate offline diagnostic fits a robust floor plane from current raw
returns in an explicitly declared floor band. It searches 2--50 mm residuals,
groups current returns within .12 m, then queries the existing SAM checkpoint
with an image grid and up to sixteen sensor-only group seeds. The 2 mm search
threshold is **not** a new sensor-accuracy assumption or runtime floor gate.
Groups with three points can be queried diagnostically, but the original
eight-point association requirement is reported separately and remains intact.
No old scan points are accumulated to manufacture support.

Fixture bounds enter only after extraction and inference, to score outputs.
They never supply prompts, geometry, labels, mass or risk. Camera/LiDAR source
times and current projection/depth visibility are retained. The floor-band and
frame placement still depend on the declared Isaac ego-pose/world-floor oracle.
The diagnostic preserves the original height allowance (3 cm per measurement,
plus floor fit/extrapolation); no candidate is admitted to control or CRA.

## Results

Image gates here mean model quality >= .85, >= 200 pixels, <= 40% image area,
no border truncation and >= 98% connected component. A passed image gate is not
verified whole-object ownership or reliable physical dimensions.

| View | Raw plate-region returns | RGB candidate frames | RGB plus >=8 visible near-floor returns | Runtime component |
| --- | --- | --- | --- | --- |
| 1.1 m, original axis | 0 | 18/18 | 0/18 | FAIL |
| 2.8 m, south / 90 degrees | 3--4 | 10/18 | 0/18 | FAIL |
| 2.8 m, diagonal / 60 degrees | 9--10 | 18/18 | 18/18 | FAIL |

The close view uses automatic image-grid proposals. Far views recover the plate
through raw near-floor seeds; ordinary filtered obstacle prompts cannot do so.
Diagonal masks have 607--644 pixels and model quality .8659--.8768. The observed
return residual is roughly 22--26 mm, still unresolved under the original height
allowance. Returns mainly sample one narrow rim strip (last XY eigenvalue ratio
.00718), not a complete plate footprint or volume. Do not feed those extents
to mass estimation as though they were whole-object dimensions.

The existing silhouette diagnostic returns a continuous outline cue in the
accepted image masks. Its range changes with view: .430--.530 close,
.616--.655 south, .568--.589 diagonal. This is a view-dependent 2D proxy,
not a calibrated sharpness comparison, contact CRA factor, or plate phi.
No new mass, fall-height, thermal, complete four-factor or navigation claim.

The three viewpoints are a development study, not repeatability or moving-camera
validation. Real empty-floor/reflection controls and a bounded near-floor
association policy remain necessary before runtime integration. A plausible
next step is a separate channel combining these RGB candidates with current
raw returns and temporal ownership checks, preserving the main floor filter
and uncertainty. Thin-object dimensions need their own validation as well.

## Validation and artifacts

- 109 focused tests passed; full regression **3708 passed in 151.98 s**.
- Ruff across `robot`, `scripts`, `tests` and Git diff checks passed.
- All 106 indexed accepted-v2 files and all original asset-file hashes match.
  Its manifest remains SHA256
  `5f340caeab502955d2a54310079e30d24e88ae5c1b5c71e9b562d5cc83ec0b1c`.
- Captures, original failed gates, replay data, masks, figure and logs:
  `outputs/broken-plate-sensing/`; main figure `plate-sensing-results.png`.
- Asset: Broken Plate (low poly), Akimkhan, CC BY 4.0, original Poly by Google
  credit. See `assets/restaurant/sketchfab_contact_props/README.md`.

Reproduce a fresh capture using `scripts/evaluation/run_restaurant_demo.py`
with `--config configs/restaurant_broken_plate_sensing.json`, `--stage stationary`,
one of `--case near_rgb`, `south_rgb`, `diagonal_rgb`, and a fresh `--output`.
Source must be clean and committed. Expected current component result is FAIL.

Replay with `scripts/evaluation/audit_near_floor_sensing.py --source <captures>`
and `--output <fresh-audit> --sam --evaluation-fixture
scenes/generated/restaurant_mask_diagnosis_broken_plate_v1.json`.
The diagnostic never imports into the runtime worker or changes an old report.
