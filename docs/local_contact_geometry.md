# Local contact geometry correction

The old global silhouette proxy can score a smooth elongated object above a
broken one. It takes a maximum over elongation, circularity deficit, concavity
and unsigned polygon turns. The vase replay demonstrates the elongation
confound; unsigned turns also fail to distinguish an inward notch from an
outward point. These are limitations of the proxy, not proof of faulty CRA
aggregation or failed image--LiDAR association.

The user authorized improving this estimator. Separate default-OFF
`--local-contact-geometry` selects a local convex boundary proxy during
stationary image-object validation. It cannot enable wheels or human trials.
Legacy default estimates and all primary/main/PR117 holds remain unchanged.

## Measurement and mapping

`robot/perception/local_contact_geometry.py` processes only the accepted outer
mask contour. It normalizes winding, resamples to 512 equally spaced boundary
positions, and applies Gaussian smoothing with sigma max(0.8 pixels, 0.003
perimeter). At each position it measures signed turns across three arc-length
supports: max(3 pixels, 0.015/0.03/0.06 perimeter). Numerical resampling adds no
sensor evidence. An exactly collinear cusp gets its turn sign only from a
nondegenerate larger support, preventing signed-zero errors at narrow tips;
an inward slit remains inward.

The continuous strength is the smaller of the two fine-scale turns, mapped
linearly from 25 to 155 degrees and clipped to [0,1]. Its persistence multiplier
maps fine/coarse turning-angle ratio from 0.35 to 0.75, clipped to [0,1]. Smooth
curvature tends to accumulate with support length; a concentrated corner
persists. A nearby concavity can cancel the coarsest turn, so strength still
requires two positive finer turns. The maximum product over boundary positions
is the object cue. These fixed, configurable thresholds are engineering
assumptions, not trained physical sharpness or injury calibration.

Global elongation, circularity and concavity are retained as diagnostics only
in this mode. Texture edges are not inputs. The original pixel, confidence,
connected-component, truncation and hole gates remain. Unavailable masks remain
unavailable; the approved factor prior policy still provides numeric scores.
Contact confidence and uncertainty intervals are unchanged. An accepted
specialist exposed-part cue still takes the maximum with the shape cue; local
smoothness cannot suppress it. Sources/timestamps, mass, current motion,
potential/thermal priors, interaction estimate, CRA aggregation, fixed lambda,
occupancy, safety margins and controller checks are unchanged.

The approach uses standard contour operations; background references are
[OpenCV shape descriptors](https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html)
and [He and Yung's curvature-scale-space corner detector](https://hub.hku.hk/bitstream/10722/46475/1/91677.pdf).
This is an engineering implementation motivated by local/multiscale geometry,
not a reproduction of that paper or a validated contact-injury model.

## Independent tests and archived-image diagnosis

Tests cover circles, smooth ellipses/capsules, blunt rectangles, triangles,
nail-like tips and jagged stars at four rotations and three resolutions.
Smooth shapes stay below 0.2; pointed fixtures exceed 0.5. Single-pixel spurs,
inward slits, contour winding, class-name independence, uncertainty/provenance,
specialist preservation and original rejection gates are checked. An early
thin-tip failure exposed the signed-zero defect; the fix resolves sign from
larger observed supports rather than weakening the test. End-to-end unit checks
verify contact changes in grouped CRA while raw IDs/point counts, other factor
scores, mass and lambda remain identical.

Replay of the 64 original accepted vase masks gives median raw cues:

| Recorded resolution | Version | Legacy cue | Local cue |
| --- | --- | --- | --- |
| 640 | Intact | 0.453778 | 0.204854 |
| 640 | Broken | 0.362345 | 0.352309 |
| 1920 | Intact | 0.449666 | 0.154293 |
| 1920 | Broken | 0.329448 | 0.215084 |

These are replayed visual cues, not newly executed CRA or physical sharpness.
The stronger responses occur at observed rim/corner locations. Resolution and
view dependence remain; a break hidden inside a silhouette can still be missed.
No vase category, authored break location or desired score is an estimator input.

Additional archived props were examined after fixing the initial parameters:
intact/broken crate local medians 0.223943/0.321903 (16 supported updates each).
The bin remains problematic: only 6/16 masks pass the unchanged interior-boundary
gate, and an acute mask protrusion near its lid yields a high local cue (median
0.849935). This cannot establish physical sharpness of the bin. Preserve that
limitation; the new option is not accepted for arbitrary masks or final demos.

## Reproduction and native validation

`scripts/evaluation/audit_contact_geometry.py --source <capture> ... --output
<fresh-json>` replays original masks, verifies source joins, records hashes,
and saves both estimators' evidence without altering input artifacts.

Use `configs/restaurant_local_contact_geometry.json` with the existing
`scripts/evaluation/run_restaurant_demo.py`, `--stage stationary` and a fresh
output directory. It reuses the same licensed vase scenes, scale, camera pose,
LiDAR, SAM/mass/part weights and ownership gates. Cases are `intact_vase`,
`broken_vase`, `intact_vase_rgb1920`, `broken_vase_rgb1920`.

Fresh native results and final validation are recorded below after completion.
This is prior-assisted contact-factor validation; potential/thermal remain
labeled priors, mass is uncalibrated, and Isaac supplies the ego reference.
No new complete four-factor, navigation or universal sharpness claim.
