# Simple-object CRA and Safe-SAGE test

**Latest gate (2026-10-08): PASS for all three matched Isaac wheel trials.**
The controller correction below resolves the earlier navigation stall. This
accepts a static single-object controller bridge with measured LiDAR geometry
and explicitly controlled oracle factor profiles. The separate sensor-only
assessment remains unknown/saturated (`phi=1.0`), not a complete four-factor
sensor result. The historical failed gate is retained below.

This bounded test adds one blue, static, 0.25 m radius sphere to the anonymous
USD session layer. The saved restaurant scene is not edited. The waypoint goal
is 2.50 m ahead of the initial robot pose and the sphere lies on the straight
nominal line, about 1.30 m ahead. No bypass path is supplied to the nominal.

## Evidence boundary

Pandar returns supply the target's occupancy geometry, visible dimensions and
centroid motion. The known fixture position selects the controlled track and
its radius supplies an independent clearance monitor. Neither authored center
nor radius is supplied as obstacle geometry to MPC or CBF. Other restaurant
objects are outside this single-target field and clearance acceptance scope.
Coordinate transforms and robot ego pose come from the USD simulation state
(oracle references). The world-Z-zero floor reference and nominal 0.20 m/s
approach speed are configured priors. Visible height is not fall height;
no full-volume sphere fit or authored collision mesh is fed to the filters.

The sensor record preserves unavailable mass, contact and thermal inputs. The
support/landing estimator is invoked, but missing support coverage remains
unknown; object height is never substituted. A stationary target has nonzero
relative speed when the nominal robot approaches. The robot camera also saves
an actual RGB view for inspection.

Three separately labeled **oracle experimental profiles** exercise the real
physical-factor adapter. These are illustrative controlled inputs, not vision
model outputs or validated physical safety estimates:

| Input | Low | Medium | High |
|---|---:|---:|---:|
| Mass, kg | 0.25 | 0.25 | 0.25 |
| Support-to-landing drop, m | 0 | 0 | 0 |
| Relative approach speed, m/s | 0.20 | 0.20 | 0.20 |
| Visual exposed-edge proxy | 0.02 | 0.40 | 0.80 |
| Thermal cue | 0.01 | 0.01 | 0.01 |
| Interaction evidence | 1 | 1 | 1 |
| CRA phi | 0.02985 | 0.40603 | 0.80201 |
| Outward flux magnitude | 0.53582 | 0.98724 | 1.46241 |

Confidence 1 in these profiles means the software receives exactly the chosen
experimental inputs. It does not mean physical calibration is perfect. The
contact proxy is explicitly enabled and keeps its named proxy basis. Physical
sharpness and hardness are not invented. Default CRA uncertainty handling and
its 0.5–1.7 flux mapping are unchanged.

A separate constant-parameter baseline uses risk 0.40 and flux magnitude 0.98.
It has no estimated four-factor profile. Geometry, controller settings, goal,
configured passing-side guidance, and lambda 1.25 match the CRA cases. The legacy class-policy
configuration uses a different flux range and some class-specific lambdas;
those values are not mixed into this comparison.

## Validation and execution

The script first captures twelve matched stationary scans, checks robot drift,
preserves the saved scene, checks oracle four-factor readiness with the existing
readiness validator, and builds finite fields and valid initial MPC predictions.
Wheel motion is rejected if any stationary prerequisite fails.

The initial coarse 0.20 m / 10-yaw setup stopped safely at an intermediate
discrete configuration boundary, despite independent sphere clearance remaining
positive. Offline reconstruction of its saved field matched exactly and
reproduced that stop. A finer 0.10 m / 16-yaw grid resolves that particular false
unsafe query. The measured footprint and its 0.05 m policy margin are unchanged.
The test uses twelve integration and geometry-check substeps per 0.20 s MPC
interval. The optional MPC integration mode predicts the curved turn produced
by successive Euler execution steps, including its analytical Jacobians;
geometry checks follow the same curve. A single coarse Euler endpoint and a
straight interpolation between endpoints were insufficient in the numerical
replay. The legacy MPC default remains one integration step. This is still a
kinematic prediction, not a model of wheel inertia or slip. The analytical CBF
rechecks every executed physics-step command. Root and joint velocities are
cleared during trial resets. These settings are recorded in the output report.

The comparison retains the existing command-tracking objective. An exploratory
optional state-tracking objective did not complete the numerical waypoint and
made early Isaac motion computation take several minutes. That attempt was
stopped and is not an accepted motion result. No avoidance trajectory is
supplied to the nominal.

The field and CRA assessment remain frozen during each run, with explicit
static-obstacle temporal derivative zero. This isolates the controller bridge;
it does not test continuous moving-camera factor estimation or dynamic fields.
Acceptance requires goal completion, target policy-footprint clearance above
0.02 m, observed MPC intervention, and no MPC/CBF/clearance failure. Commands,
poses, margins, solve times, path length and simulated duration are saved.

```powershell
$env:ISAAC_SIM_HEADLESS = '1'
python -u scripts/demos/cra_simple_object_isaac_demo.py --output outputs/simple_object/stationary
python -u scripts/demos/cra_simple_object_isaac_demo.py --motion --output outputs/simple_object/motion
```

Use the existing Isaac/YOLO environment. No package, checkpoint or scene
upgrade is needed. Inspect report JSON and failure JSON, not just the process
exit status: Isaac shutdown can mask an exception's exit code.

## Camera model audit

An offline replay of the actual saved stationary RGB image detected the sphere
as `sports_ball` with confidence 0.806. That class is outside the current ABO
mass estimator's supported classes. The exposed-edge checkpoint returned no
edge detections; this is not validated negative contact evidence. The sphere
is not an open-container source supported by the current plume demonstration,
and no clean plume reference was supplied. Thermal evidence remains unknown.
The audit performed no live camera–LiDAR association and cannot establish
sensor-domain mass, contact or thermal accuracy. The corrected audit records
the RGB and both model hashes in
`outputs/simple_object/vision-audit-corrected/report.json`.

The older positive contact replay utility hardcodes a PACO-specific scope in
its report and requires a positive edge. Its exploratory sphere replay is not
an acceptance result; use the explicitly scoped corrected audit above.

## Initial results (2026-10-05)

Stationary validation passed with 754 target returns, 0.00142 m position drift,
0.00105 degrees yaw drift, all four controlled predictions valid, fixed lambda,
and the saved scene unchanged. The separate sensor phi remains 1.0 and is not
accepted as a complete four-factor sensor result. Artifacts are in
`outputs/simple_object/stationary-pass/`.

The final matched wheel run used the existing command-tracking objective,
0.10 m / 16-yaw fields, and 12 integration/geometry substeps. Its stationary
gate passed with 753 target returns, 0.00140 m / 0.000739 degrees drift and
the saved restaurant file unchanged. All trial resets had identical measured
start poses within the configured tolerance.

| Wheel trial | CRA/control score | Minimum policy-footprint clearance | Goal reached | Termination |
|---|---:|---:|---|---|
| Controlled low | 0.02985 | 0.13404 m | No | 30 s duration limit |
| Constant baseline | 0.40000 | 0.13002 m | No | 30 s duration limit |
| Controlled high | 0.80201 | 0.13870 m | No | 30 s duration limit |

All three had 150 MPC interventions, zero backup-filter interventions and
positive checked MPC/CBF margins. They stopped about 1.66–1.72 m from the goal.
No monitored target-footprint overlap occurred during the recorded trajectories.
This does not establish collision freedom for unmodeled restaurant objects or
unobserved stopping transients. The overall result is **FAIL**, not a successful
pass-by and not a complete four-factor sensor result. Increasing CRA did not
resolve the common progress failure; clearance ordering is not monotonic
across all three controls.

Mean MPC solve times were 1.99 / 1.91 / 1.86 s (low / baseline / high), with
maxima 37.30 / 43.16 / 34.83 s. These diagnostic timings were collected with
other validation work active, not as an isolated performance benchmark.
Simulation pauses stepping while MPC computes; this is not real-time control
validation. The next bounded task is to diagnose near-stopped MPC horizons and
restore waypoint progress under the existing safety checks and fixed lambda,
then repeat this gate before enriching the restaurant. An offline recovery
probe is exploratory and has not established a validated controller fix.

Artifacts, including camera image and path plot:
`outputs/simple_object/motion-command-tracking/`. Capture SHA-256:
`fdfa8e63de3fc031d906d47169ddb2bb0a60cac97807f4202a242582933abd6c`.
Regression: 1,322 unit tests passed; the final targeted recheck passed 53 tests;
repository Ruff and whitespace checks passed. At this snapshot the implementation
remained on `codex/cra-simple-object-test` pending the navigation correction below.

## Controller correction (2026-10-08)

Replay reproduced a feasible but nearly stopped warm-start horizon. Local SQP
could not discover a coordinated turn-before-move maneuver, and restarting
from an unsafe nominal was expensive. A profiled plan evaluated 1,961
trajectories and repeatedly validated shared field arrays at every sample.

The demo now explicitly enables a bounded online seed search. Up to 435
candidates cover both turn directions, six durations (0.4–3.2 s, capped by the
horizon), three angular fractions, three translation fractions, stopped/slow
turning, heading hold and straight translation. An unsafe reference first gets
a feasibility search before expensive numerical restoration. A safe horizon
whose initial forward speed is below 50% of the positive nominal gets a
progress search, including when it is already rotating. Candidates are ranked
by command cost, a lower bound on the optional nonnegative path cost. A safe
reference is replaced only by a strictly cheaper safe candidate; an infeasible
reference has no admissible objective and is replaced by the cheapest feasible
candidate found in the pool. The same actuator limits, exact DTCBF acceptance
and dense configuration checks apply. No authored sphere pose, radius or
bypass waypoint enters this search. The nominal remains obstacle-unaware.
Other callers keep seeds disabled by default (configurable 25% trigger).
This is a bounded unicycle adaptation, not a global-convergence or exact
Safe-SAGE paper-equivalence claim.

Field sampling validates alignment once per batch and preserves the scalar
interpolation, temporal and yaw-mask rules. Nothing is cached between calls.
Vector geometry queries preserve half-open spatial bounds and both nearest-yaw
slices at a tie. Curved interior samples follow successive Euler execution;
endpoints already unsafe skip redundant interior work. These changes reduce
repeated work without weakening trajectory acceptance.

### Empirical tracking allowance

The seed-only wheel diagnostic stopped at a yaw/cell boundary after 6.55 s,
despite 0.23027 m independent clearance. An initial 0.02 m / 0.08 rad allowance
avoided that failure, but subsequent runs exposed slow heading recovery and
larger command-reversal tracking errors. Those diagnostic runs are not accepted
passes. The older allowance is superseded by the following final settings.

Prefix errors across 1,056 preceding complete execution intervals motivated a
configurable **0.025 m lookahead-position axis / 0.21 rad yaw allowance**.
The first interval uses scale `0.1 + 0.9 * j / 12` at execution prefix `j`;
later predicted intervals use the full allowance. The initial measured state
keeps its original check. Padded samples cover each crossed spatial/yaw cell,
and unsafe or out-of-grid queries reject the horizon. This is an empirical
allowance for the fixture, not a certified tracking bound or robust DTCBF
proof; interpolated DTCBF values retain their nominal-model semantics.

A separate audit of all 420 complete intervals in the final fresh wheel run
covered every execution prefix. Maximum observed lookahead axis error was
0.01571 m and yaw error 0.10133 rad, both inside the configured prefix envelope.
The audit compares simulation-oracle ego poses with executed wheel commands;
it is additional post-run evidence, not an automatic online bound monitor.

### Final matched wheel results

Stationary prerequisites passed with 754 target returns, 0.001440 m position
drift and 0.000654 degrees yaw drift. All trial resets used the same measured
start pose. The saved restaurant asset remained unchanged. Lambda stayed 1.25,
the deadline stayed 30 s, and goal tolerance stayed 0.10 m. Each trial additionally
commanded zero for 0.50 s and recorded monitored stopping clearance.

| Isaac wheel trial | Phi/control score | Goal time | Minimum policy-footprint clearance | Minimum stopping clearance | Mean / maximum MPC time | Result |
|---|---:|---:|---:|---:|---:|---|
| Controlled low | 0.02985 | 26.8 s | 0.19845 m | 0.28176 m | 0.210 / 0.776 s | PASS |
| Constant baseline | 0.40000 | 28.2 s | 0.18306 m | 0.26121 m | 0.199 / 0.769 s | PASS |
| Controlled high | 0.80201 | 29.0 s | 0.18804 m | 0.28342 m | 0.228 / 0.766 s | PASS |

All three completed the actual waypoint, exceeded the 0.02 m monitored-clearance
guard, and retained positive checked MPC/CBF margins without fallback. They had
117 / 124 / 128 MPC interventions and zero analytical-CBF interventions. Path
lengths were 3.004 / 2.971 / 2.991 m. This establishes the bounded functional
bridge and resolves the earlier common stall.

Clearance is not monotonic in phi across these trials. The factor-to-phi-to-flux
mapping is monotonic, but these paths do not establish a performance advantage
or general risk-clearance ordering. All three passed on the positive-Y (left)
side despite the configured right-side guidance. Passing side is soft guidance
in this unicycle adaptation, not an enforced constraint; that discrepancy
must remain visible in later comparisons.

Planning is substantially faster than the earlier failed run, but maximum solve
times still exceed the 0.20 s interval and mean times are near or above it.
Simulation pauses while planning. This does not validate real-time execution,
dynamic actors, or continuous moving-camera factor/field updates. The accepted
profiles remain oracle inputs; measured occupancy, configured priors and
unavailable sensor factors remain separate. Only this target's clearance is
monitored, not every restaurant object. General safety and complete perception
accuracy are not claimed.

Artifacts: `outputs/simple_object/motion-prefix-envelope/report.json`,
`tracking-audit.json`, `camera.png`, and `paths.png`. Capture SHA-256:
`87ed901f2e3ac78d1d097e6a60683865907d3340d8b6ff67c7aaf0c316a4131b`.
Final report SHA-256:
`4cdd081e8bae71ffffa3e30d76c5f427414d3c8d4f78bc06a3242b6c2c3434d7`.
Earlier failed diagnostics remain in the sibling `motion-progress-seeds`,
`motion-padded-progress`, `motion-heading-hold`, and
`motion-translation-trigger` directories.

Regression: **1,344 unit tests passed**, including scalar/batch equivalence,
field replacement validation, vector yaw/cell semantics, padding, unsafe-state
stops, feasibility recovery and three synthetic stalled-horizon reproductions.
Repository Ruff and whitespace checks passed. No packages, checkpoints or saved
scene assets were changed. The next stage is the agreed restaurant-setup and
experiment-design discussion, followed by stationary validation of selected
actors; this gate does not authorize claiming completed sensor-only perception.
