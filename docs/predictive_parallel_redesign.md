# Earlier preparation and longer predictive planning

The user explicitly prioritized a broader predictive-planning redesign before
preparing the PI demo. The failed short-horizon experiment in
`predictive_human_geometry.md` stays default-off and unmerged. This separate
branch tests earlier intervention; it does not weaken the unknown-fragment
watchdog or claim that its association failures are repaired.

## Bounded design

The new explicit `--predictive-parallel-redesign` requires the existing separate
parallel route-guide/guidance-tracking and predictive-geometry ablations. Its
candidate uses ten .2 s MPC steps (2 s), original field age<=.35 s, and an
explicit .50 m/s forecast-radius-growth PRIOR. Maximum source propagation is
2.35 s. The original current body radius .40+.25 m, rectangle/margin, physical
CRA/kinetic estimates, lambda1.25, human flux-1.7, all actuator and solver
budgets, freshness/execution clocks, current-field DTCBF/dense geometry,
additional hard predictive geometry, analytical CBF and independent physical/
arrival/stopping gates remain unchanged. Primary callers retain old defaults.

`ParallelPreparation` can select a fresh visually associated current human
before motion starts, using a ready positive-LiDAR centroid window, current
robot pose and the obstacle-unaware nominal route direction. It requires one
unambiguous person within3.8 m and .15..1.6 m lateral separation, with quiet
current fitted motion (<=.2 m/s) or opposite-route alignment within30 degrees.
This is a geometry-based preparation POLICY PRIOR, not inference of walking
intention. It supplies only the online away-side field/reference preference.
No pose, authored human route, future actor truth, factor score, occupancy or
free space is invented. The separate current oncoming-motion selector keeps
precedence whenever it has a valid preference.

Initial preparation requires an actual same-frame person cue. Holding requires
current verified body/fit/geometry, unchanged side, source gaps<=.35 s and the
original8 s policy lease. It retains the original visual proof/time. Ambiguity,
missing fit, changed geometry/direction or expiry releases it; reacquisition
needs a new fresh cue on a later source. It never extends physical evidence,
identity, field or command clocks. An explicit audit records whether normal
motion selection or preparation actually supplies each field reference.

The first prospective reference candidate uses activation_h4 and turning
translation preference1 BEFORE unchanged affine projection/exact checks,
with outward preference0 and guide lookahead2 m. This is a declared multi-option
redesign, not a single-variable or primary comparison. The original straight
waypoint nominal remains obstacle-unaware. No minimum executed speed is imposed.

## Evidence and gates

On the trusted first-selected native input, longer prediction alone still
preferred near-zero translation; a stronger reference is not proof of smoother
closed-loop behavior. An outward.35 alternative chose an undesired recovery
turn in this offline snapshot and is not the selected candidate. Artifacts:
`outputs/restaurant_parallel_prediction/reference-horizon2-design-offline-v1.json`.

The proposed2.35 s/.50-growth forecast covered all available queries in
DEVELOPMENT replay: parked108sources/1,261queries/0misses/minmargin.0690022 m;
moving78sources/972queries/0misses/minmargin.1118434 m. Source observations and
epochs were unchanged; only the hypothetical model configuration changed.
These are the same datasets that informed design, not independent calibration
or native acceptance. Preserve the original live onset misses and failed
navigation. Replays are `forecast-horizon2-growth050-*-offline-v1.json`.

Source tests cover preparation geometry/visual/source/ambiguity/expiry policy,
original field-reference proof, longer-horizon exact constraints and legacy
regressions. Before wheels require a NEW zero-wheel moving-human component
with forecast, preparation, original motion-selection, actor/sensor and body
containment acceptance, then independent60 and own60-frame gates at the same
source revision. Future actor truth is evaluation only. During motion retain
all existing gates and audit future coverage again. Actual final arrival and
stopping, >=.20 m curve while walking within2 m, and actual nearby translation
hold<=.5 s plus>=50% reduction from2.383333 s remain mandatory.

The original current field and first-order temporal safety forecast still
constrain the entire MPC horizon. This redesign has not replaced them with a
time-indexed dynamic field. It may remain conservative or stall; do not claim
convergence, real time, general human-motion safety or exact paper equivalence.
No native acceptance or PI demo readiness is established yet. Keep all main,
long-service and PR117 holds, existing priors and earlier failed reports.

Source freeze validation passed3,047 full unit tests in101.43 s, Ruff,
compilation, both CLI help paths and whitespace checks. The parked component
now supplies its configured nominal robot-route direction explicitly; it
never uses the actor's authored motion as a planning input. Native acceptance
is pending.

The first native component at5c56a8c recorded109 forecast sources/1,274 future
queries/zero misses, minmargin.0691675 m, but FAILED execution while exporting
the preparation report: a NumPy Boolean was not strict-JSON serializable.
No overall validation report was saved; do not label this a completed native
pass. Preserve `component-preparation-horizon2-growth050-v1/execution.json`
and its traceback. The serializer normalizes audit checks to Python booleans
and a strict JSON round-trip regression covers it. This changes reporting only;
a fresh zero-wheel repeat is still required before motion.

The preserved observations pass the repaired preparation audit offline
(52 used source updates, one original selection); this companion does not
replace the failed execution or create a native overall pass. The reporting
fix passed3,047 full unit tests in103.64 s, Ruff, compilation and whitespace
checks before the fresh repeat.
