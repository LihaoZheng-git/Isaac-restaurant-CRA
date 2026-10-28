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
No primary/main/long-service/PI/PR117 hold is released.

Source freeze passed3,107 unit tests in103.40 s,55 focused tests in5.98 s,
Ruff, compilation, both CLI help paths and whitespace checks. The component
fixture's original lateral range does not itself establish the wider-exit
behavior; recorded-state replay and the subsequent closed-loop trial remain
separately labeled.

## Native result at b3eb65f

The fresh component passed all original actor, fresh sensor, identity/body,
preparation and mass-provenance checks with zero wheels. Its future-envelope
audit passed 109 sources / 918 queries, zero misses, minimum union containment
margin 0.048630 m. Preparation used 52 sources from one original selection;
mass retention used 86 sources from 25 original accepted priors. The independent
60-frame parked gate and the live trial's own 60-frame gate both passed.

The live trial **passed native navigation/evidence acceptance**, but **failed
the separate smoothing target**:

| Measurement | Recorded result |
| --- | --- |
| Actual waypoint arrival | 3/3, final goal reached in 21.85 simulation seconds |
| Minimum independent physical clearance | 0.922686 m |
| Stop drift / minimum analytical CBF margin | 0.002754 m / 0.104314 |
| Live fields / fields while translating | 218 / 205 |
| Supported moving-kinetic / simultaneous rows | 39 / 34 |
| Nearby walking bend | 0.683653 m |
| Future-envelope coverage | 218 sources, 1,954 queries, zero misses; minimum union margin 0.026292 m |
| Mass-prior provenance | 81 retained sources, 60 original accepted priors; passed |
| Preparation provenance | 108 used sources, three original selections; passed |
| Actual nearby translation hold | 0.700000 s; target remains <=0.500000 s |
| Reduction from original 2.383333 s hold | 70.63%; reduction gate passed |

The previous adaptive variant held for 0.816667 s. This single new run does not
establish repeatability or a causal benefit of exit hysteresis. In particular,
all seven committed source rows inside the new measured hold still had a
passing preference. Exact progress references were unavailable in that
interval, rescues repeatedly approached zero future-geometry margin, and
current-field revalidation rejected plans. Earlier original-lease expiry and
qualification gaps are additional observations, not a sole-cause explanation.
The companion event window (hold plus 1.2 s before / 0.3 s after) contains ten
new-field invalidations and six stale-field stops; those counts are not counts
restricted to the hold itself. Negative margins within the original solver
tolerance do not represent relaxed acceptance.

The remaining work is feasible reference/horizon continuity around the moving
person. Do not infer that longer preference retention, faster computation,
reduced uncertainty, or removing current/future constraints would fix it.
Preserve the current result before any separately scoped follow-up. No PI-demo
release or primary-comparison success follows from this experiment.

Artifacts are local, ignored experiment outputs:

- `outputs/restaurant_guidance_continuity/component-exit22-adaptive030-cap050-horizon12-v1/`
- `outputs/restaurant_guidance_continuity/stationary-exit22-adaptive030-cap050-horizon12-v1/`
- `outputs/restaurant_guidance_continuity/motion-exit22-adaptive030-cap050-horizon12-v1/`
- `outputs/restaurant_guidance_continuity/comparison-exit22-adaptive030-cap050-horizon12-v1.json`

The motion directory contains the immutable original report, full JSONL fields,
`pause-policy-companion-v1.json`, `trajectory-timing-v4.png`, the unmodified native
`native-nearby-curve.png` frame, and `human-navigation.mp4`. The plotted largest
nearby bend pairs the robot and human at the same source time, 15.416667 s:
their **center separation** is 1.765296 m, not the physical clearance above.
The displayed video frame is within 0.05 s of that pair. Actor truth is used
only for evaluation, plotting and independent abort checks.

Original report SHA-256:
`dc0451e4c199cb9f1d00734df261f55a94cae4bb6214fb42d8eaf54b3b83ed39`.
The wrapper exited zero without a traceback and verified unchanged source
through shutdown; the saved reports, rather than that exit code, establish the
native pass. This remains a paced, prior-assisted, selected oncoming-parallel
scenario: no complete fresh four-factor, calibrated human mass, physical
sharpness/temperature, real-time, general motion-bound or causal CRA claim.
