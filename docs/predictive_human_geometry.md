# Separate predictive human geometry experiment

The user approved this ablation after reference-only parallel-walking changes
failed the prospective smoothing target. The accepted baseline reaches its
goal, bends .222896 m near the walking person and has a 2.383333 s translation
hold. None of the later reference variants removed that hold. Their original
reports remain unchanged; see `parallel_smoothing.md`.

The primary controller is unchanged. Explicit
`--predictive-human-geometry-ablation` adds a planning-only constant-velocity
human-envelope PRIOR to the six-step, .2 s MPC. The physical kinetic estimator,
CRA scores, lambda1.25, human flux-1.7, .40+.25 m body prior, actuator limits,
original current field, exact DTCBF/dense checks, analytical filter and all
independent arrival/clearance/containment/stopping checks stay in place.

## Inputs and bounds

`PredictiveHumanGeometryMemory` fits velocity to the current positive LiDAR
centroid history of each verified human body identity: .8 s window, at least
six samples spanning .5 s, gaps at most .25 s, at least20 current points,
maximum fit residual .1 m and speed2 m/s. Opposing current and fitted motion
above .2 m/s is rejected. These are bounded engineering assumptions. The fit
is inferred planning evidence; future centers are priors, never measurements.
The original RGB identity, LiDAR source window and frame are retained.

The forecast translates the current body circle and adds radius growth
`.25 m/s * elapsed source time`. The .25 m/s uncertainty is a new explicit,
uncalibrated planning prior. It is not substituted into CRA kinetic inputs.
The 1.2 s planning horizon includes the actual field age separately, up to
the unchanged .5 s age bound: maximum propagation from source is1.7 s.
No source clock is renewed. Missing fits stay unavailable and reject a
prediction-enabled plan/revalidation; the original physical field/history
can still commit transactionally so a valid source window can recover.

The exact oriented chassis rectangle, including its original rectangular
safety margin, is checked against each future circle. This avoids replacing
the robot by a larger circumcircle. Additional endpoint constraints are hard
rows without elastic slack in both SQP and nonlinear rescue. Every accepted
horizon also passes future geometry at the original dense integration epochs,
including its current state. Existing current-field constraints remain active
throughout the horizon. This is a conservative intersection, not removal of
the current human body or proof of arbitrary-motion safety/convergence.

Runtime forecasts are immutable request-local data. Revalidation uses the
current field's forecast and actual state epoch; shared MPC state is unchanged.
Insufficient, expired or mismatched source/identity data never fall back to a
zero velocity or an unlabelled legacy plan in this explicit experiment.

## Validation sequence

1. Pure source/fit, exact geometry/Jacobian, between-step collision, hard QP and
   nonlinear-rescue constraints, request-local revalidation and legacy-default
   regression tests; Ruff, compilation and CLI checks.
2. New zero-wheel moving-human component with actual packet frame IDs:
   `--predictive-human-geometry --identity-selection-body-envelope`, preserving
   existing actor/sensor/selection/body gates. Independent actor truth scores
   future coverage only; it never enters the predictor or control.
3. Independent stationary and own60-frame gates before new wheel motion.
4. A separate motion trial adds only the predictive-geometry option to the
   accepted baseline invocation. Preserve all native gates, audit future
   containment again, then score actual hold<=.5 s and>=50% reduction. Failure
   to meet either native acceptance or smoothing remains a failure.

The forecast audit tests source offsets0,.2,...1.6,1.7 s. Sufficient containment
of the independent reference body by either original-current or predicted disk
is required; both constraints remain active in MPC. It reports warmup, missing
fits, source errors, unrecorded future tails and every coverage miss separately.
At least20 source forecasts and20 walking future queries are required. This
bounded audit is not independent RGB timing, a general human-motion bound,
fresh four-factor perception, real-time control or primary CRA superiority.

## Development evidence and status

Two old parked datasets exposed a stopping-transition miss when instantaneous
LiDAR velocity was used: one miss in654 queries in each dataset even with
.5 m/s growth. A short-window fit reduced that development error; the .25 m/s
candidate still requires fresh native validation. The original preliminary
artifacts are under `outputs/restaurant_parallel_smoothing/` as
`raw-prediction-preliminary-v1.json` and
`fitted-prediction-preliminary-v1.json`. These repeated fixtures informed design;
they are not a calibrated/general motion study or independent native acceptance.

Source validation passed3,018 unit tests in108.67 s and60 focused tests in
7.97 s after the final age/horizon guards, Ruff, compilation, both CLI help
paths and whitespace checks. Tests include independent central differences
for geometry/QP/rescue Jacobians, a crossing collision between safe endpoints,
current-forecast revalidation, missing-fit stops and unchanged default commands.
This source validation is not native acceptance.

Native acceptance and smoothing are pending. Keep all previous failures,
perpendicular checked wait-and-go, primary comparison and main/long-service/
PR117 holds unchanged.
