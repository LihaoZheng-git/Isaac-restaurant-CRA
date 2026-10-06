# Simple-object CRA and Safe-SAGE test

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
passing side, and lambda 1.25 match the CRA cases. The legacy class-policy
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

## Results

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
repository Ruff and whitespace checks passed. The implementation remains on
`codex/cra-simple-object-test` for review until the navigation gate is resolved.
