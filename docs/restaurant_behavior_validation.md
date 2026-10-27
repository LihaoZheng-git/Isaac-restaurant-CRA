# Short restaurant behavior validation

The accepted final behavior design is curved avoidance for parallel walking and
checked stop-and-go for perpendicular crossing. Rear passing remains experimental;
none of its failed reports or main/long-service/PR117 holds is changed here.
These short tests precede the separate PI demo design, not a long service route.

The crossing case repeats the accepted left-to-right continuing-away10s fixture,
with current checked-wait admission and strict three checkpoints. Only the
selected eligible crossing may receive applied lambda0, after full stopped-
horizon/source/current-pose admission; physical CRA stays lambda1.25. Rejected
waits rebuild normal fields and cannot count as intentional yields.

The separate oncoming parallel fixture places the free robot at(28.8,7.0),
heading north, with obstacle-unaware local checkpoints(1,0),(2,0),(3.5,0).
The native adult starts at(29.8,10.5), moves to(29.8,7.0) at.5m/s after an
explicit3s navigation delay, and receives normal field lambda1.25 throughout.
No guidance-tracking cost or rear-reference planning is enabled in the first
trial. Authored positions/speed/delay belong only to actor rendering/evaluation;
they never supply runtime factors, masks, motion, geometry or desired phi.
Original restaurant/robot assets remain unchanged. The endpoint is not a claim
that the human stops rigidly; all-phase physical clearance still applies.

The original-layout proposal survey gives1.405545m minimum robot-route clearance
and.302145m human-route clearance; it excludes added prop bounds and is NOT
acceptance. The native setup and continuous independent monitor include all
actual prop bounds and remain mandatory abort/evaluation information only.

Before motion, require a parked parallel-human/sensor component and independent
and own60-frame zero-wheel controller gates. Keep the existing.40+.25m human
envelope, adult mass70[45,100]kg and blunt-contact.1 priors; unsupported potential/
thermal remain labeled.25 priors. The wider.30m prior was a separate rear trial,
not a changed default. No domain/quality/identity/source/freshness or stopping
threshold is relaxed. Preserve the original exact DTCBF/dense geometry and
per-step analytical checks, nominal/MPC speed limits, N6/.2s execution interval,
QP1000/rescue20, .5s candidate limit and .025 simulation pacing. No real-time,
complete fresh four-factor or causal CRA-superiority claim follows.

Parallel acceptance requires actual articulated oncoming walking, independent
grounding/collider/layout checks, supported current human motion/kinetic inputs,
at least three distinct current observed oncoming source epochs, normal
lambda1.25 on all policies, >=.20m lateral excursion while the walking actor is
within2m, actual final goal rejoining, clearance, containment and stopping.
A late curve, restored lambda, positive clearance or proposed reference alone
cannot pass. Crossing retains the existing intentional-wait/execution/source
policy gates. Every failure remains a separate original report.

Source validation passed2,803 unit tests in100.73s and69 final focused tests
in5.41s, including explicit parallel/default actor dispatch and incompatible-
protocol rejection. Ruff, compilation, CLI help and whitespace checks passed.
Native acceptance is pending. Exact frozen invocations and actual
videos will be saved under `outputs/restaurant_behavior_validation/`; ignored
local records are not automatically bundled into a PR or shared externally.

## Stationary component results and overview correction

At frozen eed82a1, both parked moving-actor/sensor components passed with zero
wheel commands: parallel9idle/15movingfresh, median velocity error.049974m/s;
crossing9idle/12movingfresh, error.149543m/s. Actual actor speeds were.516538/
.520357m/s; minimum furniture clearances.395205/.283841m. Grounding, native
articulation, collider readback, parked drift and structural video checks passed.
Both independent controller gates then passed60frames/zero wheels and envelope
containment (147.54/162.15wall seconds). These are not wheel/curve acceptance.

Visual inspection found the new parallel external overview almost black at
eye(25.5,6.0,4.3), despite its nonempty/distinct-frame structural check. Preserve
that original report/video and distinguish it from front-RGB perception, which
passed independently. Presentation readiness FAILED. The render-only correction
uses the proven indoor eye(25.9,6.7,3.2), leaving the robot's actual RGB/LiDAR,
scene/actor/route, physical models, estimators, controller and clocks unchanged.
Repeat the parallel component and visually inspect its new overview; each motion
trial still requires its own fresh60-frame gate. Old independent gates may be
retained only with exact source comparison proving the two observer-eye literals
are the sole functional difference. No physical failure is repaired by video.

The corrected parallel component passed again at a470065 (289.98wall seconds,
9idle/13movingfresh associations, .048418m/s median velocity error). Its actual
overview was visually inspected and shows both native human and robot. All
original physical/sensor checks still passed. Before wheel trials, a report-only
correction makes the unused crossing-plane metadata null in parallel mode;
the exact source audit includes that single reporting expression. It changes
no physical/control behavior or acceptance. New own60-frame gates remain
mandatory before motion; native curved/stop-go results are still pending.

## First parallel navigation result and bounded retry

At frozen cf44564, `parallel-motion-v4` FAILED after17.8 navigation seconds:
1/3 waypoints, progress timeout and only.000777m lateral excursion during the
nearby walking window. Physical clearance was.241402m, braking drift.000872m
and accepted-command analytical margin.381853. The actor/current-motion gate
passed (53 distinct observed oncoming epochs, .087197m/s median steady world
velocity error) and all active body-envelope containment checks passed. This
does not establish arrival or curved avoidance. The launcher returned0 after
the expected failed-acceptance RuntimeError; saved `passed=false` governs.

The trusted native source14.8s replay places the robot query in occupied cells
owned by human envelope3000000000 in both adjacent yaw slices. It returns
`unsafe_configuration_boundary` before optimization. This is not repaired by
waiting longer or by calling restored lambda a successful curve. The observed
mass/contact uses remain labeled priors; potential/thermal remain configured
priors and zero complete fresh four-factor assessments were recorded.

The next bounded parallel retry may use the user's previously approved, separate
checked Laplace-guidance-tracking ablation, beta100, with the same route, actor
timing, lambda1.25, .40+.25m body prior and all original limits/acceptance gates.
All other invocation arguments stay matched. This encourages an earlier checked
turn; it changes controller preferences and must not be called the primary
Safe-SAGE objective or a CRA advantage. It requires its own fresh60-frame gate
before wheels. If no eligible checked reference exists, legacy fallback remains.
Perpendicular crossing stays checked wait/go; the user's latest steering focuses
on parallel walking. A new crossing wheel repeat and the PI demo remain pending.

An evaluation-only companion for the older lower-aisle v9 rear image records
simultaneous center separations2.122/2.763/2.919m at5cm/20cm/maximum29cm bend.
The strongest bend happened after transverse walking ended. Its original failed
report SHA256 remains9fe1f5a3a91ec5729716ffd4e0e14a01dbed8b3eaa50462d19c3bdfe69866fc6;
the companion does not revise old criteria or validate close rear passing.

The weight-only `parallel-tracking-motion-v5` retry at16c0a30 also FAILED:
1/3 waypoints, 24.466667s progress timeout, .000629m nearby bend, .238118m
physical clearance, .000854m braking drift and .387222 minimum accepted-command
analytical margin. Current motion and all active envelope checks passed;51
oncoming epochs were supported, but zero complete fresh four-factor updates.
Its independent+own60 stationary gates passed with zero wheels. Original report
SHA256:2c71eab84cfd2bbcc9cbe415651eea0702d941d4617f42759a72b577215627f9.

Code review explains why weight alone cannot supply proactive parallel steering:
the forecast reference projects the nominal velocity, while the optional
tangent is gated by the transverse-crossing selector. That selector correctly
rejects parallel walking. The earlier source9.8s native snapshot also had
h=1.14333, outside the activation threshold1. Later, source14.4s had h=.27226
but no exact-safe progress reference. The separate bounded follow-up should
explicitly select current visually associated oncoming motion and an away-side
tangent, retain original source times, and test earlier activation as engineering
preferences within the already approved tracking ablation. It must not change
the default crossing selector, primary objective, CRA, lambda or geometry gates.

## Opt-in observed parallel-guidance follow-up

The separate `--parallel-guidance-ablation` adds oncoming-parallel selection to
the existing checked tracking ablation. Defaults remain transverse-only. A
selection needs one visually associated current human track, at least3 consistent
moving observations, >=20 positive LiDAR returns, .2..2m/s speed, <=3m distance,
<=4s constant-relative-velocity encounter and predicted separation<=1.6m.
Observed velocity must lie within30degrees of opposite nominal-route travel,
with >=.15m lateral separation to resolve the away side. The nominal segment
direction is configured route context, never an obstacle-aware nominal command.

An observed human to the route's right selects the left oriented tangent and
vice versa. Current motion/geometry failures, source/identity gaps, competitors,
passing or original8s expiry release the preference; old selection clocks never
renew. CRA/factor provenance and numeric priors are unchanged. Selected field
side and the existing selected-obstacle reference share this preference; every
candidate still needs full exact checks and per-step analytical filtering.
This does not plan a rear pass, infer free space or guarantee a side/convergence.

The prospective native recipe uses beta100, activation_h2, tangent fraction1
and `selected_observed_obstacle`, retaining all other route/timing/body/physical
limits and fixed lambda1.25. These are explicit engineering preferences in a
separate ablation, excluded from the primary comparison. A parked moving-human
component now audits actual current selection and applied side/lambda before
independent and own60-frame controller gates. Native acceptance remains pending.

Final source validation for this follow-up passed2,849 unit tests in97.73s,
71 focused tests in5.54s, repository Ruff, compilation, both CLI help checks
and whitespace checks. Initial sandbox temporary-directory setup errors were
resolved by using a fresh workspace-local pytest directory; no dependencies or
environment settings were upgraded. Native component/controller results must
still be recorded separately before any new curved-avoidance claim.

### Native observed-selection results at 18a19f7

The freely parked native component passed (292.17 wall seconds, 25 distinct
current away-side selections, all actor/sensor/geometry/zero-wheel checks).
Its overview was inspected and shows the native human and robot. The independent
60-frame controller gate passed, followed by each motion run's own 60-frame gate.

| Separate tracking trial | Arrival | Nearby walking bend | Clearance / braking drift | Result |
| --- | --- | --- | --- | --- |
| Original 3 s actor onset | 3/3 checkpoints, 21.9167 s | 0.101554 m | 0.201984 / 0.002345 m | FAILED 0.20 m curve gate |
| Actor onset changed only to 0 s | 0/3, 15.9 s progress timeout | 0.100632 m | 0.211270 / 0.000441 m | FAILED curve and arrival |

Both runs passed current human motion, original-clock away-side selection,
fixed lambda1.25, active body-envelope containment, physical clearance and
stopping checks. The first run had 36 distinct selected sources; the earlier-onset
repeat had 37. The first run's largest nearby bend occurred with the human
1.592782 m away at source18.3 s, so the person was nearby but the bend was mild.
Earlier onset alone did not solve this. No complete fresh four-factor perception
or real-time performance is established. Launcher exit0 with a gate-failure
RuntimeError remains a failed trial.

Original reports, retained byte unchanged:

- `outputs/restaurant_parallel_guidance/motion-v1/report.json`, SHA256
  `9047627f29658b78ddf293b39a2ae7f9756ba8bbd3f637dbf68a4404a6e0760b`.
- `outputs/restaurant_parallel_guidance/motion-onset0-v1/report.json`, SHA256
  `0ed725dbf20cb406e49870a9c0722cba62b2406e3a4b9384d5d9c79417938e51`.

The user approved a further separate variant: intermediate waypoints become
pose-only route guides (final goal remains mandatory), with stronger away-side
reference influence. This is an explicit controller/reference ablation, not a
change to the primary comparison, CRA, physical priors or acceptance gates.
It requires source tests and new stationary controller validation before wheels.
Perpendicular crossing remains checked wait/go; all main/long-service/PR117
holds and earlier failed records remain.

## Separate stronger route-guide variant

The next default-disabled variant uses `--waypoint-mode guides` and
`--guidance-tangent-blend convex` with the observed parallel selector. The
existing pose-only route guide supplies a 1 m lookahead target; intermediate
planes may be passed off the chord within its existing association bound. Only
the final goal must be reached within the original tolerance. The MPC now uses
that actual nominal target rather than a conflicting static checkpoint. Strict
checkpoint callers still receive the same target as before.

The legacy additive reference is `v_goal + a * v_tangent`, clipped to the
original reference speed. The separate convex preference is
`(1-a) * v_goal + a * v_tangent`, where
`a = fraction * sqrt(clip(1-h/activation_h,0,1)) * goal_fade`.
It reduces competing goal pull as the currently observed away tangent becomes
active. This is an engineering travel reference in the approved tracking
ablation, not the paper's unchanged objective or a safety margin. No current
qualified selection means a request-local zero tangent, with no shared planner
mutation, old trajectory reuse or source-clock renewal.

The recipe retains beta100, activation_h2, tangent fraction1, original 3 s actor
onset, route, lambda1.25, body.40+.25m prior, physical/solver limits, freshness,
exact DTCBF/dense geometry, analytical filtering, independent clearance,
containment, stopping and the 0.20 m curve within a 2 m walking encounter gate.
Report route-guide progress separately from exact checkpoint visits. New
independent and own60-frame zero-wheel controller gates are mandatory. Actor,
perception and selector component reuse requires an exact source audit proving
those modules unchanged from the accepted native component. Native acceptance
is pending; all earlier failures and primary-comparison holds remain.

Source validation passed2,869 units in107.39 s, 48 focused tests in7.13 s,
repository Ruff, compilation, CLI help and whitespace checks. Two trusted local
native selected-field snapshots also retained exact-safe plan success under
both additive and convex route-guide references. Convex requested stronger
away rotation within the same .8 rad/s limit; original field bytes/epochs were
unchanged. This is offline same-input reference evidence, not counterfactual
navigation or a substitute for the new native stationary/motion gates.

### First convex guide result and bounded rejoin retry

At frozen abeb768, the independent and own60-frame zero-wheel gates passed.
`outputs/restaurant_parallel_route_guides/motion-v1/report.json` FAILED overall:
1/3 route guides, 13.5333 navigation seconds, `nearby_track_lost`, despite a
0.207672 m bend during the nearby walking window. Current human motion,
away-side selection (37 distinct epochs), body containment, 0.306906 m minimum
physical clearance, 0.118516 minimum accepted analytical margin and 0.004370 m
braking drift passed. No complete fresh four-factor updates occurred. Original
report SHA256 remains
`dae02c4a1c3f67cbc5fab338db6d546627071a19c898be431be44173752efab2`.

The stop is justified by unresolved track327: 16 positive returns observed once
at source14.3 s, centroid(29.863761,9.002218). At source21.6 s the returning robot
entered its unchanged1.2 m watch radius. Whole-surface positive correspondence
failed (maximum distance0.214581 m versus0.10 m limit); the child lacked the
history needed for bounded missing-fragment coverage. Its original 3D minimum
gap to the main human cluster was0.223076 m, outside even the existing0.20 m
clustering option. The original RGB frame visibly truncates the person at the
image edge. Neither class membership, a wider arbitrary cluster, expiry, nor
the human's later departure can erase this unresolved observation.

A separate bounded parameter retry changes only the pose-only route-guide
lookahead from1 to2 m. It keeps the fragment, guard, source clocks, actor,
lambda, priors and all safety/acceptance limits unchanged. An offline nominal-
only rejoin from the recorded post-encounter pose predicts minimum distances
1.170440/1.222557 m to the retained point for1/2 m lookahead. This is a geometric
hypothesis for a gentler return, not evidence of free space or native acceptance.
New independent and own60 gates precede its wheels. Preserve the failed1 m
result even if the retry completes; no general fragment-tracking fix is claimed.
