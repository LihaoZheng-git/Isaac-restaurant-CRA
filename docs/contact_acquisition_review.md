# Contact acquisition review after v3

The completed v3 navigation video is preserved. A byte-identical sharing copy,
credits, original README and summary are in `outputs/final demo/v3-pi-share/`.
`preservation.json` records the hashes and original paths. The separate
`outputs/contact-acquisition-review/preservation-before.json` protects all 319
pre-existing final-demo files. This experiment does not replace that recording.

## Diagnosis

Exact saved-mask replay reproduced all 116 target cues: 56 stationary bin,
58 stationary crate and each first live acquisition. Both runs already used
the local convex multiscale proxy, and live operation already retained its
first accepted intrinsic contact estimate. Freezing it again is not a repair.

The bin's first live mask has a thin pixel spur at its upper edge, giving raw
cue 1.0. The crate's maximum (0.600162) is a lower outer corner, not proof of
its broken interior rim. The previously reported stationary median contact
factors 0.117775/0.120053 have overlapping ranges. Stationary acceptance proved
dataflow and association, not a required sharpness ranking. Subsequent live
candidate masks vary too: the crate has only two verified image-group sources
before static-location support takes over. A longer contact acquisition window
would therefore not automatically repair this mission.

Two alternatives were evaluated without changing their parameters:

- Existing same-SAM crop refinement accepts the representative stationary bin
  view, but rejects both first live views under the original agreement/topology
  gates. Its fallback retains the bad bin cue. No gates were weakened.
- Dyadic contour smoothing at 1/2/4 times the original smoothing changes cue
  strength but retains the reversed first-live ordering, and weakens some
  resolved triangle controls. It is not promoted.

All results and failed initial diagnostic/test attempts remain under
`outputs/contact-acquisition-review/`. `mask-diagnosis.json` contains source
hashes and exact scoring maxima; `live-following-views.json` records subsequent
verified candidates. These are offline sensor replays, not new motion tests.

## Bounded opt-in correction

`--contact-pixel-denoise` selects a standard one-pixel binary opening of a copy
of the accepted contact component, using a 3x3 cross. The entire correction is
rejected if it removes more than 2% of component pixels, splits the component,
introduces/preserves unresolved holes, disappears or expands the original mask.
Rejection keeps the original contour. The 98% retained area follows the existing
dominant-component resolution budget. No component is chosen for a desired score.

Only the contour used by the local contact proxy changes. Original SAM mask,
LiDAR ownership, occupancy, mass, original timestamps/confidence/uncertainty,
specialist-part maximum, other factors and controller remain unchanged. Existing
static-property retention then retains this intrinsic cue with its original
clock, expiry and confidence decay; current velocity and interaction remain live.
There is no fixed final contact factor or fixed phi.

This explicitly assumes one-pixel-wide protrusions are unresolved visual detail.
A real very thin tip can also be removed: this is a visible contour proxy, not
physical sharpness, material hardness or proof of safety. No class-name override,
model training, new weights or environment change is involved. The default is
off, and the native CLI currently permits only separate zero-wheel prop tests.

On the exact two saved first live images, raw cue bin/crate changes from
1.000000/0.600162 to 0.309927/0.568647. It removes 6/803 and 4/2004 pixels,
respectively. This improves those particular observations; it is not a new
full-mission score or guarantee of ordering across views. All 114 saved target
stationary masks retain overlapping cue ranges (bin 0.124594--0.578055,
crate 0.263089--0.568201). The corrected medians are 0.378465/0.394840.

## Validation and reproduction

`tests/unit/test_contact_boundary_resolution.py` checks rotated/rescaled smooth
and pointed controls, a multi-pixel spur, whole-correction rejection, immutable
source masks/clocks, specialist maximum, default-off CLI and retained contact
with live kinetic and unchanged non-contact factors/occupancy. Original local
geometry, mask refinement, grouping and static-retention regressions also apply.

Replay all actually assessed groups, including non-target groups:

```powershell
python scripts/evaluation/audit_contact_geometry.py --source outputs/service-demo-v3-open-aisle/bin-static-v1 outputs/service-demo-v3-open-aisle/crate-static-v1 outputs/service-demo-v3-open-aisle/long-motion-v1 --pixel-denoise --output outputs/contact-acquisition-review/all-current-replay.json
```

Use a new output path on repeats. The stored run audits 56/68/58 current groups;
the crate stationary source includes ten non-target groups. Retained LiDAR-only
rows are not presented as fresh mask assessments. Replay reports raw cues,
never substituted live factors or a re-executed phi.

Native recipe `configs/restaurant_contact_boundary_resolution.json` adds only
the explicit correction to the original two 60-source zero-wheel prop commands:

```powershell
python scripts/evaluation/run_restaurant_demo.py --config configs/restaurant_contact_boundary_resolution.json --case bin-static --stage stationary --output outputs/contact-acquisition-review/bin-static-v1
python scripts/evaluation/run_restaurant_demo.py --config configs/restaurant_contact_boundary_resolution.json --case crate-static --stage stationary --output outputs/contact-acquisition-review/crate-static-v1
```

Fresh native outcomes are pending at implementation. No new wheel trial or
replacement demo is accepted. Preserve every prior failure, v3 archive and
draft/main/primary/PR117 hold. A reliable contact-demo contrast still requires
source-matched native evidence; the broken interior rim remains outside this
outer-contour estimator's demonstrated coverage.
