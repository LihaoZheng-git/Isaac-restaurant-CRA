# Separate prediction-horizon diagnostic

The density-independent body-association experiment reached all three waypoints
without a continuity stop but failed the curved/behind passing gates. Its six
supported human kinetic observations had zero simultaneous robot/human rows.
Keep this successful bounded arrival separate from the failed visible bypass.

Two trusted original late-crossing planner snapshots were tested offline with
their original fields, epochs, commands and physical limits. The default 30
intervals x .2 s predicts six seconds ahead using the existing local first-order
temporal field forecast. In these snapshots, the long horizon produced either
no accepted command or rotation-only recovery. At six intervals (1.2 s), the
same exact validation admitted forward commands: approximately (.1002,-.7996)
at source 29.0 s and (.1999,+.8) at source 29.9 s. The 10/15-step diagnostics
remain separately recorded. No actor truth, source-clock renewal, zero temporal
derivative or removed physical geometry entered these replays.

An additional direct-tangent objective prototype did not change the late-state
commands and is preserved on `codex/cra-selected-tangent-reference`, not this
branch. Do not claim it is integrated or solves the crossing.

## Bounded native comparison

`--mpc-horizon-steps 6` explicitly changes **prediction duration**. Omission
retains the factory default. This does not change the .2 s MPC interval/verified
execution lease, physics-step analytical checks, field/source cadence, gamma,
solver budgets, actuator bounds, nominal waypoints, computed CRA, factor priors,
lambda 1.25 or the human-only -1.7 boundary-flux ablation. Every candidate retains
all exact DTCBF/dense geometry checks over its configured horizon and all current
state/field revalidation and independent clearance/braking guards.

A shorter horizon also reduces anticipation coverage. It is a separate
controller-parameter ablation, not a safety theorem, exact paper equivalence,
new control-frequency claim or default-primary replacement. Any future matched
comparison must use the same horizon in its variants. Preserve the original
six-second failed results even if a shorter trial passes.

The separate baseline-only replay in
`outputs/paper_flux/short-horizon-baseline-replay-v2/results.json` also rechecked
the full returned trajectories with exact DTCBF/dense geometry and the first
commands through the unchanged analytical filter. Both commands remained
unchanged by that filter; exact minimum margins were .003789 and .000375,
analytical margins .226704 and .150323. These are two original-input numerical
diagnostics, not native motion or a matched latency benchmark.

The next trial keeps the same 20 s exogenous crossing, .5 m/s native human,
90 s simulation bound, .025 pacing and own 60-frame parked gate. No new geometry
prior, reverse motion or relaxed acceptance threshold is introduced. Native
arrival, stopping, continuous motion and curved/behind passing acceptance are
pending; offline feasible commands alone do not pass them.
