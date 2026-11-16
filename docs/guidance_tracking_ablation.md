# Checked local Laplace-guidance tracking ablation

The user approved explicit guidance tracking as a separately reported controller
ablation. The original Safe-SAGE objective remains the primary; lambda stays
1.25. This changes controller preferences, not CRA scores, physical evidence,
occupancy, safety constraints or the obstacle-unaware waypoint nominal.

## Current method

`LaplaceGuidanceTrackingMPC` builds a short reachable reference from the current
Laplace field and **active configured waypoint**, in the same field frame.
The goal is copied with the asynchronous planning request; the existing
waypoint-index check rejects results for an earlier waypoint. Configured goal
coordinates are nominal-route data, never perceived physical evidence.

The reference heading follows the field direction, bounded about the waypoint
bearing. The bias fades quadratically within .5 m of that waypoint. This fade
affects only reference generation, never physical risk or any safety condition.
Forward preference comes from the maximum forward speed in the unchanged
nominal horizon, clipped to actuator bounds. Heading alignment and a radial
speed cap reduce translation when facing away/near the goal. References stop
inside .04 m; actual waypoint acceptance is unchanged (.10 m in the live demo).

Four default rollouts use moving and turn-first caps .4/.8 rad/s, clipped to
actuator limits. An explicit .1 minimum turn-first cap restores the historical
six-candidate pool. The progress correction in docs/guidance_progress.md also
requires .01 m horizon goal gain (or reaching the existing reference region).
This is auxiliary-reference eligibility, not an executable progress/turn floor.
Every reference needs the original
full temporal DTCBF and dense geometry checks. Safe references are ranked by
goal progress plus look-ahead-distance-scaled heading improvement. No usable
reference means the original controller is called. Reference generation is
local and does not establish convergence or a global path planner.

The selected reference supplies states and commands to an MPC worker. It adds
position weight `e`, yaw weight `e * lookahead_distance**2`, and command weights
`e / actuator_limit**2`, where
`e = beta * clip(1 - h_current / activation_h, 0, 1)**2`.
The original nominal-command cost remains. The added command term appears in
QP, nonlinear objective, gradient and recovery ranking; lower bounds may omit
only nonnegative state costs. Reference arrays are owned and read-only. The
normal final trajectory/command checks still apply after optimization.

The development candidate is beta=100, activation_h=1, heading cap=60 degrees,
fade=.5 m, reference goal radius=.04 m. These are engineering settings for a
separate ablation, not fitted physical parameters or an automatic runtime tuner.
Weight zero delegates directly to the primary, including its result type and
without extra reference sampling. A separate state-path target is rejected in
the enabled variant rather than silently replaced.

Runtime flags: `--guidance-tracking-weight`, `--guidance-activation-h`,
`--guidance-max-heading-bias-degrees`, `--guidance-reference-goal-radius`,
`--guidance-fade-distance`. Defaults keep the primary disabled. Per-plan reports
record the checked reference, effective weights and active goal. The primary
comparison aggregator rejects enabled ablations. Any later attribution of a
CRA benefit needs matched fixed-policy and CRA runs with the same controller.

## Validation contract and retained failures

The fixed-field audit uses authored ORACLE disk geometry/phi, .05 s virtual
integration, no native future-configuration padding and no wheel commands.
Command-only nominal horizons match the live runtime's nominal interface.
`--track-nominal-states` is a **separate stress fixture**: an earlier account
incorrectly called this the native contract. The runtime does not normally pass
an explicit state-path target. Those earlier results cannot stand in for a
command-only/native comparison.

The old heading-only prototype at 9d753bf passed both state-path stress sides
(31.0/34.45 s) but its native 8 s human crossing was interrupted for sustained
intermediate-waypoint nonprogress. Its 295 partial fields have no final report,
arrival, independent clearance, stopping or final human acceptance. See
`outputs/guidance_tracking_ablation/native-beta100-8s-v1/agent-interruption.json`.
Failed command/rate/uncapped-heading and active-goal variants remain in the
output tree; archived source is under `archived-active-heading-speed`.

The checked-path-only and initial checked-command references also circled near
intermediate goals. `checked-command-path-v22` failed both 40 s completion gates
while its geometry/CBF checks passed. Do not erase this negative result. The
goal-approach blend in `checked-command-fade-v23` then completed the command-only
three-waypoint fixture: left 35.6 s / +1.062406 m arc; right 31.7 s / -.792146 m.
Both had exact accepted geometry/CBF margins. These are dirty-tree development
results; source digests are in each manifest. Frozen-source repeat results and
native results must be recorded separately.

Unit checks cover zero-weight identity, active-goal validation, frame rotation,
both steering signs, unchanged nominal inputs, safe-reference/fallback paths,
zero guidance/speed, unsafe initial stops, waypoint approach, owned arrays,
finite-difference objective gradients, QP consistency and runtime epoch
preservation. The primary-comparison exclusion is also tested.

Native collection must first pass the unchanged 60-frame stationary gate. Keep
all mask/identity, provenance, occupancy continuity, field/candidate age,
execution lease, full exact DTCBF/dense geometry and per-step analytical CBF
checks. This ablation does not fix moving-camera plume, close-view human mask
truncation, unresolved occupancy fragments or complete four-factor readiness.
Paced prior-assisted navigation is not a real-time guarantee, exact paper
equivalence, universal safety, or a causal CRA advantage. Main and long-service
holds remain until the intended acceptance is sound.


## Frozen/native results at 7791a5a

The clean command-only three-waypoint repeat reproduced both ablation arrivals
(35.6/31.7 s). Matched primary runs arrived in 30.15/24.75 s; the left primary
run took the opposite side despite the soft field preference. Both controllers
passed the exact virtual checks. This is not a CRA benefit or sensor result.
The full unit suite passed 2,133 tests; repository Ruff passed.

The native checked-path 8 s crossing passed the 60-frame stationary human gate
but **failed**: clearance_guard at 43.9 navigation seconds after 2/3 waypoints.
There were 435 live fields/297 translating, sampled CBF minimum .085788,
reported pre-braking monitor clearance .015525 m, translation stop drift
.005792 m and only .060795 m encounter-specific left arc (required .20 m).
Human identity/selection and behind checks passed, but the human mass-source
check failed: 30 kinetic-supported rows (source 9.0--11.9 s) used the generic
ABO estimate/its retained prior, including a .738458 kg prediction, on the
independently matched human track. Keep that failure; do not relabel those rows
as valid adult-human mass. Planner median/max were .348142/4.335581 wall seconds.
120 accepted visual instances and zero complete four-factor assessments were
recorded; the run remains prior-assisted and paced, not real-time perception.

The close monitor was the configured .40 m human capsule envelope. The raw
report minimum did not include the bare-human braking interval. A separate
post-stop endpoint calculation gives -.002050 m against that padded envelope
(.05 m robot margin included). It is not a full braking-sweep minimum or a
proven physical contact. The raw report/launcher exit are unchanged; see
native-checked-path-8s-v2/review-summary.json and execution-review.json.

The diagnostic now samples all 60 braking frames, including rotation, against
the same independent monitor and reports pre-stop and stopping minima plus yaw
drift separately. A regression has safe initial/final poses and zero translation
but a colliding intermediate rear sweep. This monitor is never a runtime input.
The revised sampling has unit coverage but no new wheel validation yet.

The user approved a **separate labeled human body-envelope occupancy prior**
with current LiDAR support and validated identity for the next bounded branch.
Its implementation/acceptance is separate from this objective ablation. Do not
supply actor truth to runtime, shrink the independent .40 m monitor, extend
factor evidence clocks, or tune CRA priors/lambda to hide this failure. Further
wheel trials require stationary envelope/identity validation and the corrected
braking monitor. The generic mass domain guard also needs correction. PR141
remains a draft; main and long-service are unchanged.
