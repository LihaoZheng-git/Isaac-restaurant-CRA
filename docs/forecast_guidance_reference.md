# Forecast-aware planar reference (separate controller ablation)

The parent progress branch completed the native three-waypoint route in a
separate .025-rate run, but its actual walking-window curve still failed the
proactive-bypass gate. Most accepted late plans had no exact-safe auxiliary
reference. This branch improves reference construction; the primary objective,
obstacle-unaware nominal, CRA factors, priors and lambda 1.25 stay unchanged.
See [the parent report](guidance_progress.md) for all retained failures and its
bounded, prior-assisted paced arrival, not full proactive acceptance.

## Explicit mode

`--guidance-reference-mode planar_forecast_velocity` is opt-in. Historical
`bounded_field_heading` and `filtered_goal_velocity` remain available.

For each reference step, evaluate the current paper-space constraint with its
original temporal input. Forecast the local safety value using the same
first-order term and elapsed horizon time as the existing MPC evaluator.
Reject an already negative forecast. Project a goal-directed planar velocity
onto the spatial half-space with angular velocity fixed at zero. The previous
free-angular projection could rely on an angular result then omitted by the
direction-only path builder. This mode cannot borrow that feasibility.

The zero angular value is a **reference-projection assumption**, not a commanded
turn rate or a plant model. The subsequent unicycle rollout may turn; the full
original temporal DTCBF, angular coupling, dense padded geometry and actuator
checks must accept it. A safe instantaneous projection is never a safe horizon.
Current-field revalidation, original source ages, execution leases, per-step
analytical CBF and stopping checks remain mandatory.

A required outward projection may request more than the original slow nominal
speed, bounded by the existing maximum forward actuator speed. Nominal commands
and costs are not rewritten. If conversion/capping cannot satisfy the exact
horizon checks, reject the reference and retain the existing primary fallback.
The .01 m endpoint-displacement quality rule and .4/.8 reference pool are
unchanged. This does not promise global convergence or exact paper equivalence.

## Numeric replay diagnostics

The optional native `--save-planner-replay` writes the latest request/result
per five-second original-source-time bucket to `planner-replays`. It preserves
the actual runtime/controller, complete numerical field, temporal input,
initial state, nominal horizon, warm start and active goal. Metadata records
original frame/time, status, first command and payload SHA-256. The invocation
manifest supplies the clean source revision. Read only these trusted locally
generated pickle files in the matching repository/environment; they are not
runtime inputs or an external interchange format. Existing RGB/LiDAR snapshots
alone do not reconstruct an exact historical planner field.

Serialization is included in worker elapsed time and runner age checks. Export
never renews source/field/command clocks, grants command execution or alters
the original acceptance result. An export failure fails that diagnostic worker
call rather than silently dropping the requested snapshot.
Stored solver success is not runner/execution acceptance; a snapshot can hold
a late or rejected candidate. Offline replay never commands wheels.

## Validation and remaining gate

Focused tests cover the omitted-angular counterexample, increasing outward
demand under a shrinking temporal forecast, future-negative rejection,
unchanged source fields, primary weight-zero identity, exact-horizon rejection
and numeric replay of both accepted and rejected native runtime requests.

All 2,219 unit tests and repository Ruff passed. Both unchanged command-only
fixed-field ORACLE routes arrived in 24.45/21.85 s.
These static fixtures do not exercise natural moving-human temporal changes
and are not sensor/wheel acceptance. Native stationary and crossing outcomes
will be recorded separately. The parent 8 s case remains a failed proactive
arc even though it arrived; a later differently timed encounter is a separate
scenario, never a retroactive pass. Main/long-service remain held.

The next native diagnostic uses the existing 13 s crossing delay, identical
route/physical scene/priors/limits, the explicit new reference mode and replay
export, and .025 pacing. It intentionally ends at 20 simulation seconds after
the same 60-frame stationary gates. It can diagnose crossing-time references
and motion, but cannot establish endpoint acceptance. Preserve the older 13 s
track-loss failure and parent 8 s results separately.
