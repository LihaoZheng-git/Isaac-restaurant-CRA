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
