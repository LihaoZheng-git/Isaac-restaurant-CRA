# Live CRA navigation integration

The accepted next target is continuous robot navigation with refreshed physical
evidence and safety fields. No scheduled observation pauses belong in the final
restaurant route. Safety stops remain permitted. Existing category-free wheel
results used frozen perception and must not be relabeled as live integration.

## Sensor epochs and background inference

`robot/perception/sensor_epochs.py` keeps a bounded rigid ego-pose history,
interpolates translation and rotation without extrapolation, and maps scan-time
chassis points into the camera-time optical frame. It retains pose source/status,
rejects future/stale sensor pairs and large pose gaps, and derives actual robot
translation velocity from historical poses. The existing centroid tracker already
compensates ego motion when supplied the matching scan-time pose; its static-world
test now explicitly covers a translating/turning robot.

`robot/perception/live_inference.py` runs one background inference job without an
accumulating backlog. Worker inputs must be immutable copies; all Isaac/USD reads
and wheel commands remain on the simulation thread. Completed results keep their
original sensor timestamp. Late, future, failed and missing results are rejected,
not retimed or converted to zero-risk evidence. The default result-age gate is
0.35 s, matching the existing live-field limit.

The stationary Isaac diagnostic is
`scripts/diagnostics/live_sensor_epoch_isaac.py`. It captures real camera/LiDAR
metadata with only zero wheel commands, checks stationarity, and preserves the
on-disk scene. Transient renderer/physics changes to the in-memory root are
reported separately; no layer save is called. The earlier audit wrongly tested
unchanged in-memory serialization across normal initialization and is retained
as a rejected report.

The observed camera annotator metadata is empty in this installation. Camera
timestamps are wrapper **read times**, not independent acquisition timestamps.
The bounded simulation integration uses this declared pose-time approximation;
it cannot claim hardware synchronization, per-ray LiDAR deskew or calibrated
moving-camera projection accuracy. USD poses/extrinsics remain oracle references.
Reprojection/association quality must be validated during the driving diagnostic.

Core validation: **51 focused tests**, **1,572 full unit tests**, repository-wide
Ruff. No package installation, dependency upgrade, new wheel trial or saved scene
change occurred in this first step. Current controls, fixed lambda 1.25, CRA
confidence/aggregation and unknown handling are unchanged.

The final stationary audit passed eight distinct LiDAR/camera reads with 0.10 s
pair age, 0.00096 m robot drift and 0.00154 degrees rotation drift. The scene file
was unchanged. Artifact: `outputs/live_cra/sensor_epoch_stationary_final/report.json`.

## Remaining integration gates

1. A separately named moving-camera RGB plume method, with registration quality,
   source association and continuous cue validation. Preserve the older fixed
   clean-reference method and its stationary-only gate.
2. Refreshed SAM/mass/contact/geometry/motion evidence joined to the existing
   transactional Safe-SAGE field runtime. Account for changing field values in
   MPC/CBF through its temporal derivative; preserve all unassessed occupancy.
3. Stationary combined readiness, then actual bounded wheel trials with repeated
   accepted perception/field updates, goal completion, safety margins and stopping.
   Record processing latency and safety stops; do not claim real-time performance
   merely because the simulator can wait for computation.

## Registered moving-view plume component

`robot/perception/moving_camera_plume.py` is separate from the old stationary
method. Independent background corners, forward/backward optical flow, RANSAC
homography, spatial coverage, source consistency, view coverage and photometric
checks validate registration. The initial clean reference is warped through
accepted view changes; source/disocclusion regions are excluded, and enough
usable air region must remain. A poor alignment yields unknown evidence.

The method requires an **explicitly validated clean reference**, not an arbitrary
first frame. It retains that reference's ID/time and a declared age limit.
It accepts static visible plume cues, so upward motion is not a prerequisite.
The registered foreground masks feed the unchanged continuous area/extent/
persistence mapping, with confidence capped at 0.55. No trained weights, class
labels, temperature or aerosol identity are supplied. Negative cue observations
require an explicit validated domain. Shared positive foreground or duplicate
instance claims across a batch remain ambiguous and unknown.

Validated domains so far: synthetic camera translations with static/no visible
plume, and historical stationary Isaac captures with explicitly labeled pixel
shifts. The historical replay preserves frozen LiDAR timestamps and declares its
0.5 s stationary gate; the live source-age default remains 0.15 s. Results were
0.0000 (no plume) and 0.4520 (persistent plume). These are **not actual moving-
camera sensor validation**; driving integration must still test real pose/view
changes. Significant parallax, newly revealed backgrounds, poor texture,
lighting changes, opaque plume lookalikes and faint cues remain limitations.

Reproduction: `scripts/evaluation/moving_plume_replay.py --capture
outputs/sam2_integration/unknown_matte_plume_stationary --pixel-shift 2 1 --output
outputs/live_cra/plume-replay-positive.json`; the matched no-plume capture is
`unknown_matte_ground_stationary`. The earlier stationary method is unchanged.
Validation: 52 focused tests, **1,589 full unit tests**, repository-wide Ruff.
The actual moving-camera and continuous-navigation gates remain pending at that
component revision. The integration results below supersede only the navigation
part of that statement, not moving-view four-factor perception acceptance.

## Live integration development branch

`codex/cra-live-four-factor-runtime` joins fresh RGB/LiDAR observations to SAM
instance association, category-free rough mass, support/landing evidence, tracked
motion, visible-outline/part contact proxies and registered plume evidence.
`robot/evaluation/live_object_perception.py` builds transactional fields from
these inputs. `scripts/demos/cra_live_object_isaac_demo.py` develops that path with
a separated matte sphere and optional opaque visual puffs in a session layer.
It never saves or enriches the restaurant scene.

The worker receives copied observations. It selects one nearby observable
geometric track, keeps its unknown class and retains every unassessed obstacle
return with conservative risk. Native sensor pixels feed mass/contact/fusion;
only plume registration uses a calibrated 140-degree pinhole remap. Invalid
remap pixels are unavailable. If the default SAM prompt fails the original
quality/association gates, one retry may use an actual observed return from the
nearest 3D component (configurable 0.04 m connectivity radius). It does not
override an ambiguity rejection or provide an authored mask, class or color.

The live plume opt-in uses background above the object to exclude reflective
floor appearance changes. Independent background must cover at least 15% of
the image and inlier coverage at least 10%, in addition to the configured domain
coverage, feature/inlier, local-source, IoU and photometric gates. After a broken
consecutive registration chain, an opt-in retry may register the original
validated clean reference directly; it does not invent a new clean reference
and still needs three fresh accepted observations. The legacy defaults remain
full-image background and no reference recovery. These additions improve
stationary validation but do not yet establish robust moving-view plume recall.

Point heights are expressed in a gravity-aligned scan heading frame using the
declared oracle pose reference. This prevents chassis pitch/roll from turning
the floor into apparent above-ground occupancy. Ground height is a declared
world-ground reference; fall height still requires observed surface evidence or
the separately gated local ground-continuity prior. Object height is never fall
height. LiDAR acquisition within the header-to-read interval is assumed for this
static fixture: a conservative union of observed placements is used for field
occupancy, not per-ray deskew. The renderer still reports an internal history
limit warning despite the process requesting 128 history transforms; that
setting is not proof that sensor timing is resolved.

Live fields retain their source epochs and temporal derivatives. MPC runs in a
separate bounded worker. A delayed plan and every new field generation undergo
the same exact DTCBF and dense geometry acceptance checks at the current state;
the analytical CBF also checks each commanded physics step. Stale/failed inputs
cause recorded safety stops. There is no scheduled observation pause and the
waypoint nominal remains obstacle-unaware. Lambda stays 1.25.

The live fixture explicitly enables positive geometry retention for 0.5 s and
an optional static-scene occupancy memory prior through LiDAR blind regions.
Old source times remain recorded; absent observations do not clear occupancy or
refresh physical factors. This memory is not suitable for moving humans. Exact
cell coalescing, KD-tree connectivity/nearest-cell calculations and direct sparse
solutions of the same five-point equations reduce computation. Legacy backends
remain the defaults, equation residual gates remain in force, and focused tests
compare the optional backends with their originals.

### Verified positive-plume navigation result

`outputs/live_cra/live-positive-v3/report.json` passed the bounded navigation
gate: actual waypoint arrival, 0.25870 m minimum oracle fixture clearance,
0.00239 m stopping drift, 548 fresh field commits and 384 commits during
translation. Its stationary four-factor gate passed 12 distinct observations
in **prior mode** before any nonzero wheel command. The last stationary visible
plume cue was approximately 0.41. On-disk scene content was unchanged.

This is **not completed live four-factor perception validation**. Only 39/548
live updates had an accepted visual instance; only 1/548 had four-factor
readiness. Registration failed during view changes and LiDAR/instance association
was lost later. Unknown factors remained unknown, aggregate phi was 1.0 in all
but the first live update (0.98887),
and the conservative controller continued with live occupancy and the explicit
static memory prior. About half of the command samples were safety stops, most
while awaiting a fresh plan. Navigation took about 58 simulation seconds and
882 wall seconds at an explicit 0.075 simulation-seconds/wall-second rate limit.
Median field-build time was 1.36 wall seconds (95th percentile 1.63 s); the
maximum accepted source-to-commit age was 0.283 s in simulation time. The
separate `evidence-audit.json` records this distinction from navigation success.
No real-time or uninterrupted-motion claim is supported.

Earlier live runs failed clearance, duration or stationary readiness gates and
remain rejected artifacts (`live-none-2`, `live-none-3`, `live-none-4`, and
`live-positive-final`). Positive clearance in the duration-limited run did not
count as arrival. Puffs produce real simulated LiDAR returns and are retained;
the plume/no-plume pair is not a thermal-only causal navigation comparison.

Current verification: **1,615 full unit tests passed**, repository-wide Ruff
passed. The runtime branch remains unmerged pending its intended perception
acceptance. The next bounded work is to audit actual moving-view RGB/LiDAR
visibility and instance association, then improve valid plume registration
coverage without weakening unknown handling, confidence or CRA aggregation.
Stationary controls must pass again before another wheel trial. Restaurant
enrichment, moving actors and matched CRA/controller comparisons remain later
work.

The matched pre-retention no-plume trial subsequently also reached the goal:
`outputs/live_cra/live-none-final/report.json` passed with 540 fresh fields,
0.26640 m minimum fixture clearance and 0.00225 m stopping drift. It showed
the same limited moving-view four-factor coverage. Both endpoint successes
remain conservative navigation results rather than complete live perception.

## Bounded temporal mass/contact evidence

The user approved preserving useful earlier object estimates across brief
visual association gaps. The opt-in `TemporalObjectEvidence` store now retains
mass for at most **3 s** and visible contact proxies for at most **1 s** from
their original camera source time. The demo enables this path; the worker API
and older callers remain opt-in. `--disable-evidence-retention` provides the
legacy baseline. These limits and identity/uncertainty parameters are
configurable engineering assumptions, not calibrated physical guarantees.

Borrowing requires the same currently observed geometric track, consecutive
observation counts, agreement with the source's world-position/velocity
prediction, bounded point-count change and no close competing track. Contact
also requires a bounded source-to-current viewpoint change (15 degrees by
default, using the declared chassis-position reference). Broken continuity,
disappearance, ambiguity, excessive viewpoint change or expiry removes the
retained evidence. These checks do not prove identity through crowded merges
or occluded crossings; that remains outside the separated-object domain.

Every borrowed scalar retains its source, timestamp and original camera-frame
ID. It becomes a **property-persistence PRIOR**, its confidence halves per
second of source age, and its interval widens. Repeated borrowing never renews
the original timestamp. A stronger accepted contact cue can survive a weaker
current silhouette only until the original cue expires; it cannot be refreshed
by that weaker observation. Fresh available mass replaces its earlier estimate.

The adapter's explicit retained-property age allowances affect only marked
prior mass and visual contact inputs. Motion, route interaction, plume,
support and fall height retain their independent freshness rules and are not
stored here. Missing fall-height/plume evidence can therefore still give
aggregate phi=1 even when retained mass/contact support valid factor-specific
diagnostics. No confidence cap, unknown fallback, aggregation, lambda or
controller constraint was weakened to reduce phi.

`--evidence-gap-diagnostic` is stationary-only and cannot be combined with
`--motion`. After the ordinary 12-observation readiness gate, it deliberately
suppresses SAM output for 13 distinct real RGB/LiDAR epochs, then restores the
model for five epochs. This is labeled fault injection, not natural occlusion.
`outputs/live_cra/temporal-gap-stationary/evidence-gap-report.json` passed:
mass was retained for all 13 gap frames; contact was retained for four, then
expired in the remaining nine. Contact expired earlier because its stronger
source cue predated the first dropped output; the lease was not reset at the
drop. Full stationary readiness recovered after model output resumed. All
wheel commands were zero and the scene file was unchanged.

Validation at this refinement: **1,635 unit tests passed**, repository-wide
Ruff passed. Tests cover expiry, uncertainty decay, track reset/discontinuity,
nearby competitors, viewpoint changes, stronger-cue preservation, separate
dynamic-input freshness and rollback on a failed field transaction. The live
report now separates navigation acceptance from per-factor availability and
counts retained evidence separately from fresh instance estimates.

The follow-up `outputs/live_cra/temporal-positive-live/report.json` recorded
549 field commits, 380 during translation. Mass was available in 118 updates
(43 fresh, 75 retained); contact was available in 56 (27 fresh, 29 retained,
including stronger older cues). This demonstrates actual short-gap reuse,
not improved mass accuracy or restored long-term visibility. Potential was
ready in 11 updates, thermal in one, and complete four-factor readiness in one.

That trial **failed navigation acceptance**: its 60 s simulation deadline
expired about 0.236 m from the goal. Minimum oracle fixture clearance was
0.28311 m and stopping drift 0.00450 m, but those values do not count as arrival.
It took 911 wall seconds under the declared pacing. Preserve the failed report;
neither longer property leases nor reduced confidence/fallback conservatism
were used to make it pass. The bounded retention repair is validated by unit,
stationary fault-injection and actual moving sensor evidence; full moving-view
perception and consistent navigation completion remain unresolved. PR #117
remains a draft rather than a completed restaurant-readiness gate.

## Return-time scan audit and ego-motion correction

The current Pandar wrapper now copies GMO per-return time offsets, frame-of-
reference, motion-compensation state, model-to-app transform and start/end
poses. The same validity mask filters XYZ and times, so asynchronous consumers
cannot accidentally pair times with different returns. These are simulator
sensor records, not measured odometry. The authored sensor profile, mounting,
10 Hz rate and accumulation remain unchanged.

The installed output was audited as **SENSOR / NONCOMPENSATED**, with an identity
model-to-app transform and return offsets spanning approximately 0–0.1 s. Its
GMO start/end poses describe the final renderer tick, not the entire accumulated
scan. The [NVIDIA GMO contract](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/docs/source/generic_model_output/generic_model_output.html)
defines element times relative to the header timestamp; the
[sensor-pose helper](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/api/function__lidar_config_helper_8h_1a31c74a3fa965f9a818bee5eea79406db.html)
identifies the quaternion order as `(x, y, z, w)`.

`deskew_uncompensated_lidar` maps each return through its retained acquisition-
time chassis pose into a common header-time chassis frame. Translation is
interpolated linearly and orientation by Slerp. Missing/future poses, excessive
history gaps, mixed provenance, unsupported frames/compensation, nonidentity
model transforms and scans exceeding the configurable 0.15 s duration are
rejected. It does not correct independently moving objects, invent missing
returns, or extrapolate. Pose/extrinsic references remain **ORACLE** here.
The live demo enables this bounded path; `--disable-lidar-deskew` keeps the
legacy whole-scan start-time transform/interval-union baseline. Other callers
are unchanged.

`scan-audit-stationary`, `deskew-stationary` and `deskew-none-stationary` passed
their 12-frame zero-wheel development gates in prior mode. Snapshot capture
is explicit via `--save-sensor-snapshots`, including raw returns, times, RGB,
calibration, aligned transforms and accepted masks. Geometry diagnostics now
preserve the strict support/landing rejection, sphere fit and ground-plane
rejection; SAM diagnostics preserve candidate quality, area, boundary, owned
coverage and foreign-return contamination without accepting rejected masks.

The eight-second `deskew-moving-audit` is a **duration-limited diagnostic**, not
arrival acceptance: 74 live fields, 69 during translation, 37 fresh instances
and one complete factor assessment. Its minimum oracle fixture clearance was
0.33804 m. Partial curved-surface coverage and failed plume registration remained
unknown. An independent, explicitly oracle-selected sphere audit on the
no-plume moving captures compares the same raw returns with/without correction:
for example, frame 168's 95th-percentile radial discrepancy fell from 0.0436 m
to 0.0205 m; other frames remained mixed. This is a bounded geometry diagnostic,
not a calibrated sensor accuracy result or an input to mass/CRA estimation.

Actual later mask replay found high-quality proposals that touched the image
boundary or included too many other-track returns. Those rejections are kept;
unknown class is not the cause. The front camera can lose the sphere during
avoidance turns, and the top-mounted LiDAR loses lower-surface coverage on close
approach. Neither missing view establishes low physical risk.

Offline trials of descriptor matching, rotation-seeded flow, tiled features and
larger reprojection thresholds did not reliably restore the plume coverage
gate. None of those experimental changes is integrated, and registration,
confidence, fallback, aggregation and controller acceptance gates were not
relaxed. A replacement moving-view plume method is a separate research choice.
The unresolved renderer-history warning and approximate RGB acquisition clock
remain explicit. Full moving-view four-factor acceptance is still pending.

The subsequent `deskew-none-live` 60 s trial failed its arrival deadline:
589 fields (361 during translation), minimum oracle clearance 0.21544 m,
stopping drift 0.00073 m, and final goal distance approximately 0.825 m.
There were 29 fresh instances, six potential-ready updates and one thermal/
complete assessment. Short-gap retention was used 20 times for mass and
21 for contact. The trial's final legacy report boundary string predates the
report-label correction; its saved scan metadata records the actual per-return
deskew. Preserve this failure and the snapshot/oracle audit rather than
reinterpreting conservative motion as arrival or full perception readiness.

At this refinement, **1,655 unit tests passed** and repository-wide Ruff passed.
The demo now records its maximum simulation duration explicitly and permits
an opt-in budget up to 120 s; the default remains 40 s. An explicitly longer
90 s run is separate from the failed 60 s comparison and cannot establish a
speed improvement or erase that failure. The thermal method choice and full
live perception acceptance remain unresolved; PR #117 remains a draft.

## Fresh verification of delayed optimizer candidates

At a 0.1 simulation-seconds/wall-second pacing limit, the separate
`deskew-none-live-90` trial exposed optimizer starvation: median solve latency
was 2.855 wall seconds, above its 0.2 simulation-second candidate budget.
259 results were rejected for source age. It did not reach the goal and finally
stopped on repeated stale perception results; clearance was 0.26937 m. Another
stationary Kit process overlapped its final portion, so this is a development
load/failure diagnostic, not a controlled performance comparison.

`SafeSAGEPlanExecutionLease` separates candidate computation age from current
execution verification. `--planner-result-max-age` defaults to the legacy
0.2 s; an explicit bounded opt-in permits up to 0.5 s. Original optimizer
timestamps are retained and repeated verification never renews that source-age
cap. A delayed result is only a proposal: its entire command horizon must pass
the unchanged exact DTCBF and dense padded geometry checks from the **current
state against the current accepted field**. Every field generation invalidates
that verification; periodic checks occur after 0.15 s even if the field is
unchanged. Successful verification lasts at most the first 0.2 s interval,
and the analytical CBF still checks every commanded physics step. Failed,
expired, future or unverified candidates stop. Physical input/field age limits,
confidence, phi, lambda, arrival and clearance gates are unchanged. This is a
bounded unicycle/runtime adaptation, not exact paper equivalence or a new
real-time guarantee.

The stationary lease-code gate passed with zero wheels. **1,665 full unit tests
passed**, including candidate expiry, original-clock preservation, changed/stale
fields, failed verification and short execution leases; Ruff passed.

`verified-plan-reuse-none-live`, with a 0.5 s candidate cap at the explicitly
paced 0.1 rate, reached the actual waypoint in approximately 45.6 simulation
seconds (557 wall seconds). It committed 390 fields, 290 during translation,
with 0.22916 m minimum oracle fixture clearance and 0.00252 m stopping drift.
107 of 1,485 nonzero commands used candidates older than 0.2 s. Maximum
executed source age was 0.2833 s, verification age 0.1500 s and field age
0.2667 s; source clocks were not rewritten. The separate command-lease audit
records these values. There were still 1,107 waiting-for-plan stops and 144
rejections after full-horizon revalidation. No uninterrupted-motion claim is
supported, and the earlier failed trials remain failures.

This is a **conservative navigation pass, not complete moving-view perception**:
there were 18 fresh instances, three potential-ready updates and one complete
assessment.

The `verified-plan-reuse-positive-live` repeat also reached the actual waypoint,
with 356 fields (264 during translation), 0.19535 m minimum fixture clearance,
and 0.00225 m stopping drift, taking approximately 41.5 simulation seconds and
503 wall seconds. It retained 41 mass and 27 contact observations, but had just
13 fresh instances, three potential-ready updates and one complete assessment.
Puffs contribute actual LiDAR occupancy: differing times/clearances between
these controls do **not** establish a thermal-only causal effect.

These two passes validate the bounded navigation/reuse behavior, not complete
moving-view perception. The user has been asked to choose between a bounded
Isaac-trained plume segmenter feeding the existing continuous cue score and
real steam footage first. No replacement model, weights, dataset, license or
moving-camera accuracy is claimed as integrated. Camera truncation and limited
close-range LiDAR coverage also remain explicit. PR #117 remains a draft pending
its intended perception gate; restaurant enrichment is still later work.
