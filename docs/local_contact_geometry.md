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

## First native repetition and boundary diagnosis

At `bf544c3` all four 18-source stationary component gates passed, with 16
accepted object assessments each and frozen sources. The 640-pixel pair had
contact medians intact/broken 0.097883/0.113385. However, the 1920-pixel repeat
reversed the ordering again: 0.105705/0.103038. These component passes establish
dataflow, not successful sharpness ranking. Preserve all four original captures
in `outputs/local-contact-geometry/`; the high-resolution failure is not erased.

Mask inspection found a small stair-stepped corner on the intact rim. A full
image SAM pass uses a fixed 1024-pixel model input; simply increasing sensor
resolution does not force its small-object mask to resolve the actual boundary.
Changing the curvature support floors or requiring all three turns to be
positive did not solve the recorded pair consistently; neither experiment was
integrated. A crop of the existing PACO exposed-edge model detected 14/16 broken
high-resolution images, but its masks covered much of the vase body rather than
the rim. That is invalid evidence for this purpose. Its proposed runtime wiring
was withdrawn, with code and results preserved under ignored diagnostic output.

## Contact-only same-image mask refinement

Separate default-OFF `--contact-mask-refinement` requires the stationary local
contact mode. After the existing three-source image--LiDAR object qualification,
the same audited SAM checkpoint receives an original RGB crop with 25% padding.
The prompt is the interior distance-transform maximum and the original mask
bounding box. No class, authored break, scene geometry, desired score, or label
selects a mask. The full image is restored in the shared backend afterwards,
including after inference exceptions. At most 16 verified objects are refined
per frame; the code does not resize the sensor image or invent new pixels.

Candidates require model quality >=0.85 and original-parent IoU >=0.85, as well
as the original contact pixel, connected-component, border and hole gates.
Crop truncation also rejects. Selection maximizes parent agreement, not contact
score or quality; plausible masks must be mutually equivalent at IoU >=0.85.
Failure retains the original contact mask. Accepted refinement changes only the
outline used by the contact proxy. Original object masks, mass inputs, raw
LiDAR IDs/points, ownership, all other factor evidence and clocks remain intact.
Full-frame specialist evidence still takes its original maximum. No cropped
PACO outputs enter runtime. Confidence cannot exceed the original parent's.
These are configurable correspondence assumptions for separated props, not a
general crowded-scene proof. Diagnostics save chosen contour, mask hash,
source time, candidate rejection reasons and actual extra inference time.

Recorded-frame replay with these fixed settings refines all 64 vase masks.
Raw cue medians intact/broken are 0.234409/0.362267 at640 and
0.077676/0.165295 at1920. The bin refines15/16; all16 then have usable contact
outlines, median0.459891 versus0.849935 among the six earlier usable masks.
That moderate bin cue describes visible corners, not physical sharpness.
Replay is not fresh Isaac acceptance. Fresh matched component runs use
`configs/restaurant_contact_refinement.json` and new output directories.

The source-joined replay also retains an adverse control: refined intact crate
cue0.458182 exceeds broken crate0.321903 (the latter retains original masks
after refinement rejection). This is not a general broken-object classifier.
Corners, occlusion, mask uncertainty and viewpoint can dominate; internal edges
and material remain unmeasured. Do not promote the option as universal contact
perception or silently replace accepted-demo settings. Reproduce these controls
with `scripts/evaluation/audit_contact_geometry.py --refine-sam-checkpoint
outputs/sam2_integration/sam2.1_t.pt` plus the original source folders. Original
recorded factor/phi fields in that report remain historical, not recalculated
runtime claims.

At `c6455fd`, the four fresh vase component runs passed, but one intact1920
frame156 acquired a thin floor-grout filament and saturated the cue at1.0.
Median ordering alone concealed this failure: intact/broken cues were
0.073104/0.158108 at1920 and0.234409/0.381829 at640. Preserve these captures.
The extra line extended26pixels outside its original associated mask while
overall overlap remained acceptable. All other selected high-resolution
refinements extended at most1pixel. The corrected admission now requires EVERY
new mask pixel to stay within max(2pixels, 2% of original mask maximum span)
of the original mask. This configurable image-correspondence allowance rejects
the entire candidate; no trimming, score clamp or synthetic contour is applied.
It can reject a real thin feature missing from the original mask, a limitation
of bounded same-object refinement. A synthetic connected-filament test with
IoU>0.99 verifies that overlap alone cannot bypass this gate.

The first bin launch at this revision failed its preflight before Isaac because
the old generated scene indexed the earlier asset README hash. Only the README
changed. Preserve the failure; regenerate a separate scene with the same base,
mesh and placement instead of rewriting the original manifest or bypassing it.

## Final native validation and unresolved ranking

Corrected revision `26d6502` passed three fresh 18-source stationary components,
with frozen source/scene hashes and zero wheel commands. Each has 16 accepted
image--LiDAR object assessments after two qualification sources. **These are
perception/dataflow passes, not contact-risk contrast acceptance.**

| Current native case | Contact available | Refined masks | Median visual cue | Median CRA contact factor | Median phi |
| --- | --- | --- | --- | --- | --- |
| Intact vase, 1920 | 16/16 | 11/16 | 0.101993 | 0.094247 | 0.586053 |
| Broken vase, 1920 | 16/16 | 16/16 | 0.121700 | 0.089997 | 0.584110 |
| Bin, 640 | 9/16 | 5/16 | 0.323450 (supported only) | 0.179272 (all 16) | 0.624918 |

The corrected intact cue spans 0.042466..0.283697, broken 0.099080..0.174942;
there is substantial overlap. No vase cue saturates in these final runs.
The raw median ordering improves over the original global proxy, but the
**final contact factor and phi STILL reverse the desired vase ordering**.
The bin's nine available cues span 0.302236..0.917825; seven updates use the
explicit contact prior. It is not a consistently resolved low-contact control.

The factor calculation is operating as configured. Contact interaction is 0.15
in these records. The first two accepted observations have motion-limited
confidence 0.4875/0.625, later at most 0.75, and a 0.25 uncertainty prior increases
low-confidence risks. Small changes in dominant-component fraction also affect
the contact uncertainty interval/confidence. The intact vase's weak and variable
visual separation is therefore insufficient to establish conservative risk
ordering. Do not remove startup rows, reduce the prior, alter confidence, clamp
scores or use a vase/broken label to force the comparison. A good median in the
earlier 640 pair does not repair this later adverse result.

The geometric correction removes a real elongation confound, and boundary
admission removes a demonstrated false-tip source. It does **not** solve
general visual sharpness: the visible interior ceramic rim is largely absent
from an outer-silhouette representation, and small mask changes still dominate.
The existing exposed-edge specialist was also unsuitable in the crop diagnosis.
The next meaningful perception step is validated local rim/edge evidence with
smooth rims, texture/shadow and hollow-container negatives. Further tuning of
CRA priors or selecting only a favorable frame is not supported by this result.
Keep the options default-OFF and unpromoted to the final navigation demo.

Final regression: **3766 unit tests passed in 148.57 s**, **79 focused tests
passed**, Ruff across `robot scripts tests` and Git diff checks passed. The
source-joined replay retains every original mask and report, including the
filament and failed control cases. All 106 indexed accepted-v2 files and all
original licensed asset hashes match. The regenerated bin asset layer is
byte-identical to its predecessor (SHA256
`26b2f06fd76869c70090c3c4210cbb56e0d9ea53b2b656e8a5124b4ca1f8a092`).
Final run directories are `outputs/local-contact-geometry/refined-{intact,broken}-rgb1920-v2`
and `refined-bin-v2`; `final-native-summary.json` records exact outcomes.
Median full field builds are about 2.81/2.74 s for the high-resolution pair;
these paced tests are not real-time validation. Potential/thermal priors, mass
calibration limitations, the ego-pose oracle and all main/primary/PR117 holds
remain unchanged.
