# Bin and crate comparison with the long service route

## Scope and protocol

The user requested the same isolated pass for the rounded bin, a comparison
against the broken crate, and a new long service demonstration with both props.
Existing archived V3 videos and every earlier failed trial are retained.

The separate branch is `codex/cra-bin-crate-service-demo`, native source
`0047241450f674036def336f2b26d6296d536602`. The prior isolated 5 cm crate
run uses `8f19084d07858a3b3865655cd0cd84248b4d584b`.

The isolated pair shares authored robot pose (15.65, 4.05, 0), prop XY center
(17.50, 3.40), original mesh scales, elevated camera, 4 m straight waypoint,
N12 sensor MPC, analytical CBF, source clocks and contact denoising. Independent
60-source and own-run 60-source parked checks precede wheels.

The two meshes have different size. The bin's near edge is 21.4 cm farther
from the route. Their vision angle and range match, while occupied geometry,
uncertainty and interaction differ. Isolating the causal effect of CRA would
require a separate matched-geometry comparison.

The score comparison reuses the parked native unique fixture matcher. Navigation
measurements replay stored intrinsic mass/contact and current LiDAR membership.
Missing-speed rows remain in the all-observation statistics and are also reported
separately. Shared progress intervals are fixed at 0.25 m. Authored fixture bounds
identify rows for post-run evaluation only.

The waypoint nominal uses localization and remains obstacle-unaware. Camera and
LiDAR supply surrounding geometry and evidence to CRA, the field, MPC and CBF.
Field lambda is fixed at 1.25. The isolated N12 pair has explicit guidance
tracking disabled. The N6 long mission uses the previously approved guidance
tracking objective with weight 100 in the Carter unicycle adaptation. Those
controller settings must accompany comparisons between the two protocols.
The authored-world clearance monitor is passive in these recipes. Navigation,
benchmark thresholds and PhysX target contacts are reported separately. Simulated
ego localization, estimated rough mass, stored property priors and unavailable
factor priors remain explicit.

## Isolated results

One driven run per prop has completed.

| Measurement | Rounded bin | Broken crate |
| --- | ---: | ---: |
| Reached 4 m waypoint | Yes | Yes |
| Navigation simulation time | 26.9167 s | 32.0000 s |
| Final position error | 9.636 cm | 9.649 cm |
| Braking translation | 2.729 mm | 2.610 mm |
| Maximum lateral offset alongside prop | 2.311 cm away | 8.02 cm away |
| Minimum padded chassis-to-mesh-box clearance | 20.426 cm | 3.287 cm |
| Legacy benchmark-circle clearance | 14.378 cm | −4.642 cm |
| Target PhysX contacts in navigation and braking | None reported | None reported |
| Accepted visual parked sources | 58/60 | 37/60 |
| Retained original mass/contact with current LiDAR support | 120/250 fields | 79/307 fields |
| Retained rows with live relative-speed estimate | 120/120 | 76/79 |
| Retained rows also meeting the original 20 cm whole-center diagnostic | 59/120 | 12/79 |

The original crate benchmark-circle criterion fails and is preserved. Independent
mesh-box clearance and PhysX event reporting are additional observations.
The retained-property count establishes intermittent current-point support.

| Median score | Bin parked | Crate parked | Bin navigation | Crate navigation |
| --- | ---: | ---: | ---: | ---: |
| Intrinsic visual outline cue | 0.16502 | 0.53558 | 0.25852 | 0.53558 |
| Conservative contact factor | 0.09395 | 0.13400 | 0.15189 | 0.17267 |
| Four-factor CRA phi | 0.58592 | 0.60423 | 0.61462 | 0.62338 |

All five shared 25 cm progress intervals have higher crate median contact
and phi. Crate-minus-bin median differences are 0.02078 for contact and
0.00951 for phi. The phi difference ranges from 0.00922 to 0.00964 across
these intervals. Shared intervals cover the first 1.25 m of the route, while
the props are centered about 1.85 m ahead of the start. The final shared
interval contains just one qualified crate field.

Observed distributions overlap. Navigation phi spans 0.61243–0.64992 for
the bin and 0.62193–0.84626 for the crate. The crate's three unavailable-speed
rows produce the upper value through the existing interaction/kinetic prior
handling. Its intrinsic cue stays 0.53558 on every retained row. Restricting
the displayed subset to supported speed gives crate phi 0.62193–0.63345.
The all-row result remains preserved alongside that diagnostic subset.

This local demonstration uses repeated correlated observations from one run
per object and supports the observed median ordering for these views. Physical
sharpness calibration, generalization, mass calibration and continuous
whole-object tracking remain open evaluations.

The pair audit found zero original-source/current-point errors. All selected
target points lie within the corresponding mesh box plus the original 3 cm
allowance.

Native sources are under
`F:/CRA-experiments/service-v3-balanced-pair/bin-short-motion` and
`F:/CRA-experiments/crate-aisle-balance/setback5-motion`.
Comparison data and image are
`outputs/service-demo-v3-balanced-contact/matched-bin-crate-scores.json`
and `matched-bin-crate-scores.png`. The complete paired short video is saved
separately as `matched-bin-crate-short.mp4` and its smaller share copy.

## First long service trial

The full trial retains the V3 N6 controller, route and human profile and uses
crate (17.50, 3.40) and bin (21.20, 3.40). Six fresh prerequisites and the
mission's own parked block passed. The first startup attempt was interrupted
and preserved. `startup-v2` supplied the passing prerequisite.

The completed run in
`F:/CRA-experiments/service-v3-balanced-pair/long-motion-v1` **failed** its
300 s duration bound, reaching 1/8 guides. It recorded 1,882 live fields,
410 committed while translating. Final local pose was
(9.418233, −0.145627, −0.587774 rad), ahead of the crate. Braking drift
was 0.384 mm. Recorded authored-world minimum clearance was −3.536 cm.
The installed PhysX crate reporter recorded zero target contacts. These
observations retain their separate meanings.

The execution receipt records frozen sources, failed report and a traceback
flag. The saved traceback is the runner's explicit failed-wheel-gate exception.
The route result remains failed.

Earlier read-only prefixes show a rightward turn toward the crate and a
sustained controller-selected hold. At source 80.1 s, the recorded world pose
was about (17.103, 3.981, −0.480 rad). The specific active inequality remains
unresolved for this prefix because exact successful planner inputs were
outside its recorded scope.

The completed failed-run evidence audit found two fresh crate associations
at source 48.8 and 49.7 s, with phi 0.60256 and 0.69957. There were 37
retained current-point crate rows from 49.9 to 55.2 s, with constant cue
0.36336, median contact 0.15975 and median phi 0.61729. Relative speed was
supported in 34 of those rows. All selected points met the mesh-box allowance,
and both fresh plus 14/37 retained rows met the 20 cm center diagnostic.

From source 66 s onward, all 1,463 fields lacked a fresh or retained
crate-specific property association. Occupied sensor geometry and property
reuse have separate mechanisms. Establishing a cause requires the associated
controller records. The bin was observed ahead from a distance on 1,081
sources. Carter stopped before reaching its nearby navigation window.
See `failed-long-motion-v1-prop-coverage.json` and
`long-motion-v1-hold-prefix.json` in the balanced-contact output directory.

## One-change 20 cm crate setback

A read-only placement survey used every accepted earlier corrected-V3 pose,
its settled route anchor and the padded chassis against the actual crate box.
The following minima would result if that recorded trajectory were held fixed.

| Candidate crate center y | Prior recorded path clearance | Sampled anchored nominal clearance |
| --- | ---: | ---: |
| 3.10 m | 29.28 cm | 29.96 cm |
| 3.15 m | 24.29 cm | 24.96 cm |
| 3.20 m | 19.29 cm | 19.96 cm |
| 3.25 m | 14.29 cm | 14.96 cm |
| 3.30 m | 9.30 cm | 9.96 cm |
| 3.40 m | 0 cm, rectangles overlap | 0 cm, rectangles overlap |

The bin at y=3.40 gives 21.18 cm on the same recorded path. The candidate
y=3.20 restores about 20 cm of swept-chassis space on that previous path,
while moving the crate 20 cm toward the route compared with the earlier
successful V3 layout at y=3.00.

The separately authored relief20 scene changes only crate placement and the
crate-relative zero-wheel observation pose. Bin, route, controller, camera,
limits, property rules and evaluation tolerances remain at their V3 settings.
Its native source revision remains frozen at `0047241`. Its separate recipe
and scene hashes identify the placement change.

All six fresh relief20 prerequisites passed. The final mission in
`F:/CRA-experiments/service-v3-balanced-pair/relief20/long-motion-v1` also
**failed** its 300 s duration bound, reaching 1/8 guides. It recorded 1,691
live fields, 394 while translating. Final local pose was
(9.538725, 0.139377, 0.199002 rad). Braking drift was 0.586 mm and the
authored-world minimum clearance was 22.766 cm. The installed PhysX crate
reporter recorded zero target contacts. The execution receipt confirms frozen
sources and preserves its failed report and traceback flag.

The second run turns away from the right-hand crate, then makes little progress.
Between 55 and 85 navigation seconds, all 1,801 recorded commands are zero
with reason `waiting_for_fresh_valid_plan`, while the waypoint nominal still
requests approximately 0.186 m/s forward. All 150 late field records in the
saved diagnostic prefix are numerically ready.

Exact replay of seven original sensor fields at source 62.0, 65.1, 70.1,
75.1, 80.1, 85.1 and 90.1 s identifies a specific rejection. Hold,
recorded waypoint and slow 0.025 m/s candidates fail the dense padded
configuration-space geometry test. Their unpadded endpoints, discrete-time
CBF conditions and immediate analytical CBF pass. Rejecting occupied cells
attribute to ID 7000000418, mapped offline to an archived four-point
stationary-background region centered at world (17.32217, 3.41075).
That region retains the approved 0.30 m positional allowance from source
48.5 s. Its applied risk is 0.79457 and its fixed unknown-risk baseline
0.68359.

This establishes why the checked candidates were rejected and the controller
waited for an accepted plan. The feasibility finding covers those tested
candidates. Broader trajectory feasibility remains open. The offline association of those four
points with the nearby crate mesh remains an evaluation observation. See
`relief20-hold-prefix.json` for original source hashes, exact checks and
the distinction between footprint inflation and spatial/yaw query padding.

The completed evidence audit found three fresh crate associations at source
45.0, 46.6 and 53.4 s, with median phi 0.61111 and range 0.59262–0.61874.
There were 40 retained current-point rows at 46.7–52.4 s with cue 0.41305,
median contact 0.16348 and median phi 0.61908. Thirty-six rows had supported
speed. All selected points met the original mesh-box allowance. Two of the
fresh rows and 19/40 retained rows met the whole-center diagnostic.
From source 66 s onward, all 1,276 fields lacked a fresh or retained
crate-specific property association.

The short comparison video is available. Both closer-layout failures remain
recorded alongside the subsequent accepted mission below. The second
failure's exact geometry rejection is documented above. The first trial
has the earlier command/pose diagnosis and a different turn direction.
See `failed-relief20-prop-coverage.json`.

The prepared `audit_local_passes.py` uses the actual settled nominal polyline
for a subsequent accepted full mission.
Approach is the 1 m preceding each prop's upstream box edge. Alongside spans
the projected mesh depth. Positive deviation is away from the corresponding
prop. Fresh and retained scores remain separate, with live field denominators,
speed-supported counts and original property clocks. These windows were fixed
before final native results.

## Accepted full mission at the original V3 prop positions

A new scene layer restores the original complete-V3 placement, crate
(17.50, 3.00) and bin (21.20, 3.20). The composed prop bounds were compared
against the archived original scene, with the equality checks recorded in
`original_v3_layout_parity.json`. The long run uses the V3 N6 controller.
The isolated scored pair above uses N12 and the closer common center.

All six fresh prerequisites and the mission's own parked block passed.
The native mission in
`F:/CRA-experiments/service-v3-balanced-pair/original-v3/long-motion-v1`
then **passed** all eight guides and final arrival in 159.0333 simulation
seconds. Final goal error was 9.753 cm, within the original 10 cm tolerance.
Braking drift was 2.164 mm, minimum authored-world clearance 35.207 cm and
minimum recorded CBF margin 0.07601. It recorded 1,122 fields, 1,106
while translating. Both installed PhysX target reporters recorded zero
bin or crate contacts in parked, navigation and braking phases.

The final execution receipt passed with frozen source revision
`0047241450f674036def336f2b26d6296d536602` and no traceback.
Native launch and execution took 7,360.94 wall seconds, so this remains
a paced simulation demonstration.

The parallel-human validation passed with 63 current observed parallel
rows and a 38.448 cm excursion from the actual anchored nominal polyline
during nearby walking. There were 28 supported moving-kinetic rows.
Median steady human velocity error was 0.06292 m/s and median relative-speed
error 0.01507 m/s against independent simulation truth. The native one-way
walk and the active human-envelope containment checks passed.

The original-source/current-point and fresh contact coverage audits passed.
They identify 7 fresh visual bin assessments and 5 fresh visual crate
assessments. Their fresh median outline cues are 0.31537 and 0.46425,
contact factors 0.12419 and 0.12598, and phi 0.60530 and 0.61506. Fresh
score distributions overlap. The bin's observed fresh outline cue spans
0.04669–0.73404, while the crate spans 0.32679–0.49955.

| Retained current-point measurements | Rounded bin | Broken crate |
| --- | ---: | ---: |
| Qualified rows among 1,122 fields | 83 | 311 |
| Source time range | 71.5–83.5 s | 52.3–113.4 s |
| Intrinsic visual cue | 0.29134 | 0.49955 |
| Median conservative contact factor | 0.15435 | 0.17197 |
| Median phi | 0.61567 | 0.63136 |
| Phi range | 0.61362–0.61743 | 0.62077–0.83106 |
| Rows with relative-speed support | 83 | 290 |
| Original 20 cm center diagnostic passed | 54/83 | 223/311 |
| Maximum original property age | 12.10 s | 61.30 s |

Every selected target point met the original mesh-box plus 3 cm allowance.
The crate's 21 unavailable-speed rows account for the high phi excursions.
On its 290 speed-supported retained rows, phi spans 0.62077–0.64553,
with median 0.63054. The original contact cue stays constant. These rows
reuse one property source per prop, with live motion/interaction and the
recorded original source ages.

Prop-local windows expose an important availability limit. The bin has
retained evidence on all 35 approach fields and all 10 alongside fields.
The crate has retained evidence on 25/34 approach fields and on none of
the 27 alongside fields. It also has two fresh approach observations,
and there are zero fresh alongside observations.
The larger 311-row mission total includes intermittent observations after
the pass. Property support beside the crate remains intermittent.

The robot follows an almost straight line beside both props. Relative to the
settled anchored nominal polyline, its center lies 0.28–0.51 cm toward the
crate during the alongside window and 0.30–0.44 cm toward the bin. The
maximum approach deviation toward either prop is 0.83 cm. This final
original-layout mission supports full service-route completion and periods
of scored object use. The separate N12 closer-prop trials provide the
measured visible bend comparison.

The fresh and retained score differences are observed outcomes for different
objects, positions, source ages and interactions. A causal ablation isolating
the effect of contact score would require matched geometry and control inputs.
See `original-v3-local-pass-scores.json`, the native
`contact-coverage.json` and `static-prop-audit.json`.

The actual-root motion summary contains 263 nearby-walking samples. During
walking within 2 m, it measured no translation-hold interval below
0.02 m/s and one 0.0167 s slow interval below 0.05 m/s. Minimum
human-center distance was 1.42173 m. These measurements describe the
nearby-human portion of this one run. Whole-mission pauses remain in the
source recordings.

The camera audit passed field/score/mask and video-timing parity for
1,670 native frames, 1,308 accepted masks and 67 colored tracks.
Geometry provenance passed for 34,969 region references and 705 archives.
Motion and fragment provenance passed for 890 ambiguous-motion proofs
and 4,141 fragment proofs. Human association audits retain 218 current
body joins, 20 uncertain-human rows and zero recovery events.

The completed presentation is saved in
`outputs/final demo/v3-bin-crate-validated/`. Its left panel shows the
recorded perception image and accepted colored evidence. The right panels
show the map with recorded trajectories and the native follow camera.

`restaurant-service-v3-share.mp4` is the share copy and
`restaurant-service-v3-map.mp4` is the original composed export.
`matched-bin-crate-short-share.mp4` contains the complementary closer
N12 comparison. The full export retains all 1,601 navigation and stopping
frames in source order plus 60 credit frames, at 10 fps. The full frame,
source-hash and video decode audit passed. The short copy separately passed
full decode for all 371 frames at 10 fps.

The composer verified the original 96 V3 and 12 V1 archive files during
assembly. The new package includes its manifest, exact frame mapping,
source score JSON, asset credits and preview images.

## Reproducing the layouts

Original generated restaurant layers and licensed prop assets must be present.
Use the established environment to run
`scripts/scene_creation/build_service_pair_layout.py` with
`--layout original-v3 --output-directory scenes/generated/service_pair_repeat
--recipe outputs/service-demo-v3-balanced-contact/repeat.json`.
Both output paths must be fresh. The builder verifies parent hashes and
composed bounds, then authors additive layers and the launcher recipe.
The `relief20` mode reproduces the second failed placement for further study.
The initial closer placement uses
`scripts/scene_creation/build_restaurant_service_v3.py --balanced-contact`
and `configs/restaurant_service_demo_v3_balanced_contact.json`.

For each derived recipe, run `scripts/evaluation/run_restaurant_demo.py`
with `--config`, `--case`, `--stage` and a fresh `--output`.
Run `bin-static` and `crate-static` at the stationary stage, then
`long-idle-component` and `long-parallel-component` at the component stage,
then `long-parallel` and `long-parallel-near` at the stationary stage.
Finally launch `long-parallel` at the motion stage, supplying all four
matching gates using `--stationary-source`, `--component-source`,
`--near-stationary-source` and `--actor-wait-source`. The launcher verifies
the committed runtime and exact scene/recipe hashes. Motion also repeats
its own 60-source parked check. Run Isaac processes sequentially.

The reusable authoring script was promoted after the native trials. Its
five focused regressions and real-scene parity audit reproduce both layouts.
Validated native runtime provenance remains the recorded `0047241` revision.

## Software validation and preservation

The targeted paired-score regression tests passed, 3 tests, and their Ruff
checks passed. The native-source full unit run passed 3,982 tests in
265.85 s with temporary files located on the repository volume. The final
suite, including reproducible overlay-builder tests and service CLI scope
regressions, passed **3,993 tests in 339.33 s**. Repository Ruff and
`git diff --check` also passed.

Two prior environment-sensitive attempts are preserved. The first reported
3,701 passed and 281 temporary-directory errors. The next reported
3,980 passed and two cross-volume Windows path failures. Moving the test
temporary directory to the repository volume produced the complete passing
run. Native diagnostic conditions, acceptance thresholds and evidence scores
were retained throughout.

All new figures, raw diagnostic records and videos use fresh output paths.
The final checksum audit verified all 408 earlier archive files and all
20 files in the new package. Main/primary and other
historical promotion holds continue until explicitly revised.
