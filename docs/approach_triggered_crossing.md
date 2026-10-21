# Approach-triggered crossing development fixture

The six-step/1.2 s prediction trial reached B in19.616667 navigation seconds,
before the unchanged20 s exogenous actor start. It therefore failed the intended
dynamic crossing/bypass gate despite a bounded three-waypoint arrival. Preserve
`short-horizon-human17-observed-motion-v1` and its zero supported moving-human
kinetic/simultaneous rows; motion during braking is not navigation acceptance.

## Separately declared fixture

Use the existing actor-only position trigger at **robot world y=8.4 m**, with no
elapsed-time delay. This is the observed approach line reached near the earlier
20 s encounter by the slower controller, and before B by the shorter-horizon
controller. It creates a consistent approach event across different controller
progress rates. The scripted human still walks left-to-right along the same
authored path at .5 m/s. Robot waypoints, start, actor asset, geometry and every
runtime estimator/control input are unchanged.

The native standalone `crossing` mode now permits trigger6.8..8.4, preserving
default7.8. Continuous/carried modes keep their legacy7.8 maximum. This extends
an experiment-event range, not a body uncertainty prior or safety threshold.
The trigger reads the declared ego pose for **actor authoring only**. Actor
position/phase/speed, path, desired risk and the trigger never supply physical
factors, predicted control, LiDAR geometry or passing selection. Estimation still
receives copied RGB/LiDAR/declared ego references. Actual trigger/crossing times
and the independent actor trace remain recorded for evaluation.

The next trial uses the already tested explicit six-step horizon with unchanged
.2 s interval/lease, fixed lambda1.25, computed factors/priors, observed occupancy,
human b=-1.7 ablation, solver budgets, 90 s bound/.025 pacing, exact validation,
independent clearance/braking and its own60-frame parked gate. It must pass
actual arrival, stopping, moving-human perception and curved/behind gates.

This is a different development fixture from the fixed11/20 s trials. Preserve
their failures; do not retroactively pass them or claim causal CRA superiority.
Future matched comparisons must share both horizon and event policy. New trigger
acceptance, current trajectory clearance and controller performance are pending.

Validation before native motion: all2,329 unit tests passed in83.33 s; 34 focused
locomotion/event-policy tests passed in7.41 s; repository Ruff and whitespace
checks passed. The new test accepts the standalone8.4 boundary before any native
asset authoring, rejects out-of-range/invalid values, and preserves the stricter
continuous-mode limit. These are parameter/component checks, not native encounter
or moving-perception acceptance.

## Native result at4481413

`outputs/paper_flux/approach-human17-observed-N6-v1` passed the parked gate,
reached3/3 waypoints in21.566667 s and passed moving-human perception: five
supported kinetic/four simultaneous rows, median steady world-velocity error
.176084 m/s and relative-speed error .010334 m/s. It retained215 fields,
210 while translating, with .061338 m minimum independent clearance, .138637
CBF margin, .002322 m stop drift and .096269 m endpoint error. No nearby-track
loss blocked the run. These are bounded observed-occupancy/prior-assisted results.

Full curved/behind passing still FAILED: left excursion .003206 m (required
>=.20), behind projection +.005886 m (required<=-.10). Preserve this failure;
arrival and a moving perception gate do not establish the requested bypass.

The source policy audit first qualified moving-human track76 at23.0 s, but
fresh SAM/person eligibility was absent; selection waited until24.1 s. This
1.1 s delay is separate from velocity qualification. An exact original-image
replay at23.2 s (RGB23.283333, frame212) with the already-audited person model
detected one untruncated confident(.918585) person mask: all1,786 projected
parent points passed both mask and foreground-depth coverage. The current
identity was independently maintained by the existing LiDAR/domain guard.
See `outputs/paper_flux/approach-fresh-person-frame-audit-v1.json`.

This supports investigating a fresh detector-to-current-track policy cue,
without renewing physical mass/contact/support/plume inputs or pretending a
fresh SAM assessment occurred. It is not yet an integrated replacement or a
granted change to physical uncertainty/geometry policy. Legacy source clocks,
all factor priors and both failed passing metrics remain explicit.
