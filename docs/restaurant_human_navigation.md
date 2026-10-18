# Bounded native-human navigation integration

This is a separate control gate after the parked-robot human perception
benchmark. It uses a separately named 3.9 m centerline service route and one
official Isaac adult human. The standing and crossing episodes share the
robot route, models, factor priors, fixed lambda 1.25 and solver budgets.
Neither episode is a causal comparison of CRA against another navigation
method. PR #117's strict fresh moving-view perception/main hold is unchanged.
The original 3.9466 m route is preserved. Its middle waypoint `(29.0,8.7)`
can overlap the native idle-body envelope: an independent check over recorded
idle poses had -.075161 m minimum extra clearance. The new `human-service`
route uses `(28.8,7.8) -> (28.8,8.7) -> (28.8,9.5)` and had +.120511 m at
the middle pose in that same check. These are authored fixed mission points,
not online obstacle inputs or a guarantee over every future pose.

## Changes and assumptions

`--restaurant-human standing|crossing` opts into native human animation at
SimulationApp startup. The original restaurant files stay unchanged on disk;
native animation, collision and camera additions use the session layer.
Existing static restaurant callers still require their explicit static memory.
The new human caller rejects static occupancy memory and legacy whole-scan
timing. All current LiDAR tracks and unmatched occupancy remain in the field.

The crossing actor starts at `(29.8,8.9,0)` and requests `(27.8,8.9,0)` at
.7 m/s after the robot reaches world y=7.8 m. This trigger controls the actor
only. It never supplies actor position, velocity, class, mass, collision bounds
or desired phi to the estimators or controller. Native auto avoidance is off.
An independent authored raster/capsule monitor checks placement and aborts;
it is not perception or safety-filter input. Its sampled straight-path
precheck had .302145 m extra clearance; that precheck is not an executed walk.

The audited COCO person model can select the configurable 70 [45,100] kg
adult mass PRIOR only with fresh, unambiguous same-frame SAM/LiDAR association.
The default interval is configurable with `--human-mass-prior KG LOWER UPPER`.
Original catalog-network outputs remain in diagnostics. No human mass is
measured, learned from the rig, or calibrated by this trial. Intrinsic mass
alone may persist for 3 original-source seconds under identity/competitor
gates, with a 1.5 s confidence half-life. Source time is never renewed by
borrowing. The original engineering mass interval remains a prior.

Contact is reassessed from current visible geometry. It can be high for human
outlines and is not physical sharpness, hardness or biological injury severity.
Old contact, support, plume, motion and static/translated body bounds cannot
be borrowed in this mode. Unsupported potential and thermal retain labeled
.25 factor priors. Height is never used as fall height. Every track still
gets four numeric scores and computed phi; this does not mean four measured
physical factors or low aggregate risk for a standing person.

Positive occupancy expires after .15 s. The existing nearby-track-loss stop,
.35 s inference/field source-age caps, exact current-field MPC/DTCBF/dense
geometry checks, command leases and per-physics-step analytical CBF stay in
force. A copied scan may wait at most .05 s for actually recorded return poses;
the original scan/RGB epochs remain unchanged. Robot pose/extrinsics/gravity
are oracle references; RGB time is a read-time approximation. Ego deskew does
not correct articulated actors. Paced simulation is not real-time acceptance.

## Validation status

The first new zero-wheel episode stopped on a diagnostic-camera API error.
The corrected 18-frame episode then failed its human gate: early SAM proposals
did not meet the unchanged .85 model-quality threshold and only one fresh human
mass prior was accepted. A longer 36-frame observation window is evaluated
separately, without lowering mask/association or numeric-policy thresholds.
Failed runs are retained under `outputs/restaurant_service/human-nav-stationary-*`.
The 36-frame v3 human checks passed, but the legacy unrelated bowl gate failed.
The new mode selects the human as its mandatory component; other bowl/prop
failures and priors remain recorded, and all physical occupancy is retained.

`human-standing-nav-v1` then failed navigation on a nearby-track loss just before
the endpoint: 233 live fields, 226 translating fields, .086928 m minimum
independent clearance and .003403 m stop drift. A limb/body fragment (track
1058) disappeared while body track 245 remained near its old location. This
trial is not retrospectively accepted. Its native idle root also moved .190240 m;
the subsequent reference explicitly permits bounded standing-idle sway (at most
.25 m root displacement and .12 m/s median root speed), while validating
inferred world velocity and relative speed. It never assigns zero kinetic risk.

The new human-only guard opt-in can resolve a split/rejoined fragment using
at least three current returns within .10 m of its old centroid, a unique
current cluster whose center remains within the existing .25 m association
bound, and recent coexistence. Competing/unclustered support, insufficient
returns, stale/misaligned arrays and actual disappearance still fail. It
transfers no property, shape, class or velocity evidence. Legacy guard callers
retain their original behavior. Merge events and blocked source geometry are
separately recorded. The new standing/crossing retries remain separate trials.
`human-standing-nav-v2` was bounded by an operator interruption after stalling
near the original middle waypoint. Its partial live-field stream/checkpoints
and `operator-abort.json` remain; it has no arrival/stopping acceptance.
The later centerline trials use a 45 s overall bound and a 10 s no-progress
rejection watchdog. A rejected trial cannot count as a successful pass-by.

Before coupled navigation, `human-crossing-actor-v1` passed the transverse
native-human component with zero robot wheels: .688367 m/s median speed,
.301914 m independent layout clearance, .014822 m stop drift, actual endpoint,
articulated feet, grounded root/collider and 54 decoded / 48 distinct RGB frames.
All source scene files remained unchanged. This is actor-only acceptance.

Human motion acceptance additionally requires independently matched current
velocity, at least five supported moving-human kinetic rows (including a
simultaneously moving robot/human observation), bounded
steady world-velocity error, and actual native crossing completion. Navigation
requires all waypoints, positive independent clearance, fresh committed fields
and safe stopping. Stopping conservatively without arrival is a failed
navigation result. Precise trial results will be added after execution.
Runtime safety stops are permitted while the human crosses; hazard observations
must continue during those stops. The gate does not require the robot to keep
moving toward the crossing human.

The full unit suite passed **1,846 tests** and repository Ruff passed. The
first regression invocation hit sandbox temporary-directory setup errors;
rerunning with a dedicated workspace `--basetemp` passed. No dependency or
environment upgrade was made.

## Reproduction

Run `scripts/demos/cra_live_object_isaac_demo.py` with a fresh absolute output
directory, the native-human scene, independent layout survey and:

```text
--restaurant-server --restaurant-human standing --restaurant-route human-service
--persistent-object-risk-map --sam-observed-context-box
--visual-reassessment-interval .2 --stationary-frames 36
--vectorized-fields --stream-live-records --simulation-rate .05
--qp-iteration-limit 1000 --nonlinear-rescue-iterations 20
--planner-result-max-age .5 --immediate-replan
--maximum-seconds 45 --progress-timeout-seconds 10
```

Without `--motion`, wheels remain zero. A wheel episode repeats its own
stationary gate; crossing selects `--restaurant-human crossing`. The complete
per-object audit is append-only `live-fields.jsonl`, while the JSON checkpoint
contains only the latest five fields. Native actor truth is stored separately.

## Executed centerline trials and next diagnostic

`human-standing-centerline-v1` passed all robot waypoints and stopping with
222 live fields (216 translating), .254076 m minimum independent clearance
and .002292 m stop drift. The independent motion audit matched 214 current
human observations; median relative-speed error was .007912 m/s. There were
96 human mass-prior/supported-kinetic rows and 12 supported contact rows, but
zero supported potential or thermal rows. Phi ranged .595922--.683594.
The video decoded 284 distinct frames, and source scene files were unchanged.
This is a prior-assisted standing-idle integration result, not a baseline
comparison or complete sensor-derived four-factor assessment.

`human-crossing-centerline-v1` failed after 131 committed live fields. At
source epoch 19.5, an old fragment at (28.736346,8.829163), last seen at 19.3,
had no current returns within the unchanged .10 m query radius; the current
body cluster was already moving left. The guard stopped correctly under its
then-static query policy. Final abort handling also attempted to poll a canceled
native task; abort is now idempotent and no longer treats that cancellation as
a new move failure. Partial records, blocked source geometry and video remain.

A separate focused replay of that copied scan tests an opt-in bounded moving
fragment hypothesis. It translates the old query by the *current parent
cluster's inferred velocity*, never the authored actor velocity or the lost
swinging fragment's velocity. The parent's previous velocity must be at most
.15 s old, stable within .3 m/s, supported by at least three observations and
at most 2 m/s; query age is at most .35 s and translation at most .25 m.
Unique current positive returns, .10 m query and .25 m center gates still
apply. Exactly three actual current returns supported the shifted query in
the focused replay. This is a kinematic association hypothesis, not observed
free space, transferred semantic identity, or a repaired navigation result.
Insufficient, ambiguous, stale, unstable or absent geometry still stops.

The failed crossing also had **zero supported moving-human kinetic rows**.
World velocity was observed, but the close front-camera view cropped the human's
head and rejected fresh full-instance mass associations. Fixing the continuity
guard alone cannot pass that perception gate. New opt-in camera mount controls
keep the stock defaults, preserve lens calibration and SAM boundary/quality
checks, author only the session layer and re-read actual extrinsics. The
close-view parked-robot benchmark must pass before another wheel crossing.
Pitch/height changes are experimental sensor configurations, not stock Nova
Carter camera validation. A new successful run would not retroactively accept
the failed crossing.

The prospective [final experiment plan](restaurant_final_experiment_plan.md)
separates cue perception, live integration and matched navigation comparisons.
Its baseline identities exist in the experiment contract; all three still need
verified live-runner integration before comparative results can be collected.

### Camera coverage development

The zero-wheel `human-close-camera-v1` uses the explicit parked pose
`(28.8,7.9)`, yaw 90 degrees and a +15 degree left-camera pitch. It completed
native actor motion, but failed perception with three stationary and zero
steady-moving fresh human assessments. `human-close-camera-v2` raised the RGB
origin by .45 m and pitched +10 degrees: the feet left the image, no fresh
human prior passed and the stationary gate stopped before actor movement.
Neither run permits a coupled close-range retry or establishes a useful mount.

An offline lens-envelope design diagnostic uses the recorded stock .3459 m
camera/base height and existing calibrated polynomial lens. Across the tested
pitch/height grid, an authored full-human envelope along the crossing did not
fit when the robot was at y=7.9. At y=7.4, stock height and +15 degrees had
29.4 pixels minimum envelope margin. This is an approximate *authored camera
design hypothesis*, not sensor evidence, visibility reconstruction or a safety
guarantee. Actual rendered mask gates decide acceptance.

The separate `crossing-approach` parked pose is `(28.8,7.4)`, yaw 90 degrees.
Its .7 m/s `human-crossing-approach-camera-v1` passed actor motion and stationary
perception (10 fresh rows), but had only three supported moving rows versus
the required five. Median moving velocity error was .189419 m/s; the whole
benchmark stays failed. These are development attempts, not selected successful
paper repetitions.

`human-crossing-approach-slow-v1` then passed the separate .5 m/s zero-wheel
benchmark: 10 stationary and nine steady-moving fresh visually associated
human mass priors, .103442 m/s median moving velocity error, .501495 m/s
observed native speed, .298639 m independent actor/layout clearance and
.002444 m actor stop drift. All 85 video frames decoded and were distinct;
source scene files stayed unchanged. Maximum result completion age was .35 s.
This validates only the selected parked-view component, not coupled wheels,
all-factor sensing, the faster .7 m/s case, or the failed close view.

The coupled runner preserves its original actor defaults (.7 m/s, robot-y
trigger 7.8). Explicit `--human-crossing-trigger-y` (6.8..7.8) and
`--human-walking-speed` (.5..1) configure only native actor motion. The planned
farther crossing uses trigger 7.0, camera pitch +15 and stock camera height;
robot control, physical evidence, factor priors, full-instance gates and route
remain unchanged. This scenario cannot establish recovery of the original
close crossing, nor is it a stress test of the near-body continuity guard.

Carried bowls are a subsequent component gate: the bowl must follow an actual
hand transform, retain physical occupancy, and acquire a separately associated
visual assessment. An authored bowl height, role or emitted puff never supplies
mass, fall height or thermal scores. Moving-view plume and crowded occlusion
remain unresolved; adding a carried bowl alone does not resolve them.

The installed `omni.anim.behavior.core` 110.1.4 bindings expose
`pickup_object(target, snap_to_hand=False)`, `place_object(target,
placement_target)`, `get_hand_usage_objects()` and hand IK tasks. NVIDIA's
[Behavior API reference](https://docs.omniverse.nvidia.com/kit/docs/behavior-simulation/110.1.1/behavior-simulation/api.html)
documents these capabilities. This identifies a reusable native carrying path;
it is not a tested bowl attachment or a guaranteed carry animation in this rig.
The already audited CC0 Kenney `bowl-soup` mesh can supply rendering geometry.
The next bounded gate should check attachment, posture, retained occupancy and
separate sensor ownership with zero robot wheels before carrying navigation.
