# Separate avoidance-preference exit hysteresis

The combined adaptive/mass-retention experiment passed native navigation and
evidence gates, but retained an .816667 s nearby hold. During most of that
hold a passing preference was present and exact rescues reached nearly zero
future-geometry margin. Earlier lateral-threshold churn disabled preparation
before the hold; this is an input-specific hypothesis, not a sole-cause claim.
Preserve the original native pass and separate smoothing failure.

The new explicit `--parallel-preparation-exit-lateral 2.2` changes only when
an ALREADY selected avoidance preference releases. Initial selection still
requires the original1.6 m lateral gate, fresh same-frame person cue, current
positive fit/body identity, unique candidate, forward geometry and other
unchanged bounds. Holding may continue to2.2 m lateral separation, but still
requires current fit/body/side, source gaps<=.35 s, original8 s selection lease,
and no competing eligible person. Another person in the larger exit scope
releases the preference. Returning behind the robot, ambiguity or expiry also
releases it. Fresh observations while holding never renew the lease.

The additional .6 m is a bounded planning-preference scope, motivated by the
selected .50 m/s uncertainty ceiling over1.2 s. It is not a body margin,
physical safety theorem, new object association or permission to enlarge
initial selection. Omitted exit keeps the previous policy exactly. Primary,
CRA scores, body radius, source clocks, fixed lambda, all MPC/analytical/physical
and stopping/arrival checks are unchanged. The audit reconstructs the distinct
initial/held limits and rejects a widened initial gate or inconsistent scope.

Offline replay of all279 recorded sources reproduced original default decisions
and preferences exactly. Exit2.2 changed ten source decisions. On the trusted
14.9 s planner input, retaining the original fresh-selected preference changed
the checked first command from(.184217,-.799958) to(.199635,.399797), with future
margin .049217 m versus .138289 m and an outward endpoint. The original physical
pipeline and epochs were unchanged. This is a reference-only counterfactual,
not a new closed-loop result. Its source/selection proof is saved in
`outputs/restaurant_guidance_continuity/default-parity-exit-probe-v1.json`.

Before wheels require source regression checks, fresh stationary component and
independent+own60 gates with original forecast/mass/body/selection acceptance.
The unchanged final criterion remains actual arrival/stopping, >=.20 m nearby
curve, five supported moving-kinetic rows, full current/future safety checks,
and actual hold<=.5 s with>=50% reduction from the original2.383333 s baseline.
No primary/main/long-service/PI/PR117 hold is released; native validation pending.

Source freeze passed3,107 unit tests in103.40 s,55 focused tests in5.98 s,
Ruff, compilation, both CLI help paths and whitespace checks. The component
fixture's original lateral range does not itself establish the wider-exit
behavior; recorded-state replay and the subsequent closed-loop trial remain
separately labeled.
