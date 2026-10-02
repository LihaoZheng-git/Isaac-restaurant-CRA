# Restaurant evaluation protocol

Primary comparisons use matched scenario, seed, geometry, nominal waypoint,
initial state, solver settings, and sensor conditions. The baselines are the
geometric velocity-aware CBF, fixed-class Safe-SAGE-style policy, and continuous
CRA Safe-SAGE policy. Primary Safe-SAGE comparisons require lambda 1.25.
Risk-dependent lambda is a separately labeled ablation.

Each trial records evidence mode (`sensor`, `prior`, or `oracle`), completion,
collision, fail-closed state, minimum clearance and CBF margin, interventions,
fallbacks, and elapsed time. Oracle/prior trials cannot be reported as sensor
performance. Baseline summaries are grouped by evidence mode so oracle, prior,
and sensor results cannot be pooled into one completion, collision, or
clearance statistic. Repeated seeds and uncertainty/domain-shift evaluations
remain required before research conclusions.

The protocol and validators are implemented; closed-loop restaurant
experiments remain pending. A local exposed-edge contact checkpoint and visual
mass baseline can be invoked, but their complete factor accuracy is unvalidated.
Actual sensor trials remain blocked on independent restaurant mass references,
positive fall-surface validation, general contact consequence evidence and
verified continuous-plume weights. Wheel motion should only begin after
stationary scenario-specific perception acceptance checks pass. See
[stationary benchmark results](stationary_sensor_gate.md).

The stationary readiness gate enforces that requirement. Sensor trials accept
only measured or inferred inputs; prior demonstrations may additionally use
explicit priors; oracle trials require the explicit `oracle` estimate status.
Fallback and unavailable evidence never pass readiness. Four-factor scope
requires all four diagnostics before aggregate `phi` can be reported.
Factor-specific scope can validate one hazard family, but explicitly leaves
aggregate `phi` unreportable so unknown factors are not silently treated as
zero.
