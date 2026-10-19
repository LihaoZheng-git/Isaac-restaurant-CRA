# Motion-aware tangential passing

This opt-in changes the passing-side policy at the online field boundary. It
does not alter the original service waypoints, physical factors, phi, normal
flux mapping, lambda 1.25, occupancy, source clocks or MPC/CBF acceptance.
Fixed-class and CRA field controllers can use the same opt-in. Geometric CBF
retains its original command path; a shadow field does not steer its wheels.

Current LiDAR tracks supply geometry and inferred world velocity. Three
consistent moving observations, positive point-count, speed, transverse-motion
and near-term relative-closest-approach gates must identify a single candidate.
The selected tangent points opposite the crossing object's lateral motion at
the robot-facing boundary: for a robot moving north, an eastbound crossing
selects the westward tangent. These are bounded constant-relative-velocity
selection hypotheses, not measured future trajectories or a safety theorem.

A same-track policy preference can remain latched for eight original-source
seconds to prevent side chatter through the encounter. It uses fresh current
geometry and releases on source/identity loss, reversal, expiry, passing or a
competing crossing. Its original selection epoch/velocity stay visible. This
is controller policy memory, not retained physical evidence or semantic identity.
No missing obstacle is removed. No factors are borrowed or renewed by this path.
The policy is cloned and committed transactionally only after a complete field
and temporal derivative are accepted.

The separate left-to-right native actor fixture moves its initial scene anchor
from the old east side to the west side and reverses the walking endpoint. This
is explicitly authored rendering/evaluation information; no actor position,
speed or direction parameter enters the perception worker or side policy.
The existing right-to-left default is preserved. Carried groups and crowded
crossings are outside this opt-in.

The new zero-wheel mirrored-human component at source a487d93 passed actor
walking/arrival/stopping, grounded/collider/layout checks and fresh human
perception/motion checks. Its 52 selected field rows passed a separate post-hoc
identity/selection-clock/lambda/flux audit. That audit used observed actor truth
only for scoring; the policy selected from LiDAR tracking. The original report
and separate post-hoc report are preserved in
`outputs/motion_aware_passing/native-left-stationary-v1/`.

The short comparison retains
the original northbound human-service route and uses a separately declared
left-to-right actor crossing, .5 m/s requested native walk and exogenous start
13 navigation seconds. Both policy-off/on trials share all settings except the
passing-policy opt-in. The native actor parameters never enter the estimator.
The existing stationary human gate is repeated before each wheel trial.
No wheel trial passed full acceptance, and no pronounced curved bypass was
established. A side
preference remains soft; an almost straight passage or safe stop is not a
validated large detour. Actual navigation needs endpoint/stopping, clearance,
live kinetic/field/geometry checks and an independently measured trajectory.
The new behavior's gate additionally requires at least .20 m left/trailing-side
excursion during the moving encounter, a trailing projection <= -.10 m at
closest approach, and return within .15 m of the original goal. Independent
arrival, stopping, human kinetics, all occupancy and exact safety gates remain.
These are prospective bounded demonstration criteria, not calibrated social
distances or a theorem. The long service demonstration, prior study failures
and main hold remain.

## Collected wheel attempts: not accepted

All three attempts used committed source `fb8d8af`, the original human-service
waypoints, a native left-to-right crossing beginning 13 navigation seconds after
start, requested walking speed .5 m/s, fixed lambda 1.25 and the existing exact
MPC/CBF, freshness, continuity, mask-quality, truncation and source-expiry gates.
Each passed its 60-frame stationary human gate before wheels were enabled.
The newer 13 s timing is a different scene condition from the previously accepted
8 s right-to-left comparison; those earlier results are not its acceptance.

| Attempt | Camera / connectivity | Arrival | Minimum signed monitor clearance | Full acceptance |
|---|---|---|---|---|
| Policy off, v1 | Stock height, +15 degrees / .18 m | Yes, 24.483 s | .200758 m | Failed: no supported moving-human kinetic or simultaneous rows |
| Policy on, v1 | Same matched configuration | No, track-loss stop at 18.650 s | .366561 m | Failed: perception, continuity and visible bypass |
| Policy on, v2 | Explicit +.60 m height, level / .20 m | No, track-loss stop at 15.250 s | .131582 m | Failed: perception, continuity and visible bypass |

The v1 off/on pair differs only by the passing-policy opt-in. Its invocation
manifest, complete fields, raw reports, commands, independent actor traces,
videos, post-hoc audit and trajectory plot are preserved under
`outputs/motion_aware_passing/wheel-pair-v1/`. The off run's trailing excursion
was .039165 m and its original-goal error .096350 m. The on run selected the
correct current human in four rows (maximum independent centroid error .100296 m),
but its moving-encounter excursion was only .024829 m, below the unchanged .20 m
criterion. Its last nearby-track stop retained an older human-body fragment
instead of declaring that occupancy free. Stopping drift was .004267 m.

At close range high-quality SAM masks touched the image boundary and were
correctly rejected. Earlier visually associated adult-mass priors expired at
their original three-second limit. All three attempts had zero supported
moving-human kinetic rows; missing inputs retained numeric factor priors.
Field/planner latency also caused stale-result rejection and temporal recovery.
The on run selected twice and released twice on source/identity gaps; it did not
establish continuous passing-side coverage. Do not extend evidence expiry,
relax truncation or clear unresolved occupancy to make this test pass.

The v2 configuration was declared separately in `raised-camera-plan-v2.json`.
It first passed a new zero-wheel native/perception/body-prior/passing-policy
gate: 85 fields, nine fresh moving body-prior rows and 52 selected policy rows,
maximum selected centroid error .159517 m. That component is preserved in
`outputs/motion_aware_passing/native-raised-stationary-v2/`. Its later wheel
attempt is a configuration-development diagnostic, not a matched comparison
with v1. It stopped on a new body fragment last seen .8 s earlier, before the
actor completed the crossing; original-goal error was 1.327923 m and stopping
drift .000851 m. Both its stationary success and wheel failure are preserved in
`outputs/motion_aware_passing/wheel-raised-v2/`. Raising the camera did not resolve
the moving perception/continuity gate.

Positive executed CBF margins and signed monitor clearances do not turn these
failures into acceptance or establish absence of physical contact. The launcher
returned zero even when the script raised its explicit failed-wheel-gate error;
the raw failed flags and tracebacks remain. There is no completed fancy bypass,
CRA advantage, fresh moving four-factor acceptance, real-time guarantee or
long-service demonstration from this branch. PR #137 remains draft/unmerged
against `codex/cra-live-four-factor-runtime`; main is unchanged.

## Next bounded work and research decision

First reproduce the near-view mask/source-age and disappearing-body-fragment
failures in saved sensor replays. Any ownership/continuity correction must retain
positive occupancy and the existing identity, ambiguity and freshness checks;
repeat the parked gate before further wheel experiments. Neither a configured
actor role nor a missing mask supplies current physical evidence.

The current MPC minimizes deviation from the original nominal commands/path.
A selected tangent supplies guidance derivatives and safety-field boundary
conditions, not a hard promise to curve behind an actor. Waiting can remain a
valid minimum-intervention solution. A proactive guidance-tracking objective
would be a separate controller extension requiring an explicit research choice,
not a CRA-factor correction. A concrete candidate is an opt-in, bounded penalty
on look-ahead velocity disagreement with the selected current Laplace guidance,
active only for a qualified crossing and zero outside it. Keep the original
waypoints, exact acceptance, all occupancy and lambda unchanged; expose weights
and speed bounds and make the same option available to fixed-class and CRA
controllers. Evaluate it separately from the original primary comparison.
This objective change is not implemented and awaits a user research decision.

Validation completed during implementation: the full unit suite passed 2,058
tests; subsequent focused checks of the selector, validator and actual object-map
integration passed 47 tests after diagnostic refinements. Ruff passed. These
software/component results do not supersede the wheel failures above.
