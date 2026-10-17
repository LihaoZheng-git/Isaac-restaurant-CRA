# Bounded multi-waypoint restaurant service route

`service-route-v1` passed actual arrival at all three waypoints and stopping on
2026-10-17. The 3.9466 m nominal path extends the previous 2.2 m route, remaining
inside the unchanged fixed field and original property-expiry limits. It is
not acceptance of the proposed 30.88 m route, field recentering, human motion,
payload transport, real time, or a causal CRA advantage.

The configured world route is `(28.8,5.6) -> (28.8,7.8) -> (29.0,8.7) ->
(28.8,9.5)`. These are obstacle-unaware nominal waypoints. The new explicit
`--restaurant-route service` selects them; `short` preserves the original
single endpoint. Camera and robot setup, all models, four-factor priors, fixed
lambda 1.25, physical occupancy, temporal clocks, exact MPC/DTCBF/dense geometry
checks and per-physics-step analytical CBF remain unchanged.

CRA route interaction now uses the active waypoint. When a waypoint advances,
the previous command lease and warm start are discarded. A pending planner
result tagged for an earlier waypoint is rejected; its original source time
is never renewed. The existing nominal controller predicts its waypoint
horizon without receiving obstacle information. No observation pauses or
intermediate-waypoint observation stops were scheduled.

## Actual Isaac result

The run repeated and passed its own 18-frame FoodBowl stationary component gate
before wheel motion, using `restaurant_service_v2.usda`. It then recorded:

| Metric | Result |
| --- | --- |
| Arrival / stopping | Passed; all 3 waypoints reached |
| Live fields / translating fields | 204 / 203 |
| Minimum independent clearance | 1.21780 m |
| Minimum analytical CBF margin | .43822 |
| Stop drift | .002273 m |
| Final anchored state | (3.30185, 1.89022, .69925 rad) |
| Waypoint advances | 11.0333 s and 16.0000 s after motion start |
| Motion-stage wall duration | 460.02 s; .05x requested simulation pacing |

The clearance metric uses the existing independent authored mesh raster,
conservative new-prop boxes and bowl disk, actual asymmetric base footprint,
policy margin and pixel allowance. Those bounds never enter perception,
factor estimation, nominal control, MPC or CBF. An offline nominal-path survey
had 1.1906 m minimum extra clearance; that precheck was not motion acceptance.
This is a wide-aisle integration trial, not a forced avoidance comparison.

The complete stream audit passed across 10,471 repeated per-track assessments:
ordered source epochs, four finite scores per object, phi .60561--.68359,
.13333--.16667 s source-to-commit ages, and fixed lambda. It counted 41,502
configured .25 factor-prior uses, 382 evidence-supported factors and 379
retained-property uses. **Only two current visual instances were accepted
during motion; zero complete fresh four-factor assessments occurred.** Those
counts do not establish robust object identity or fresh vision for every track.
Potential/thermal remain priors where unsupported. Existing prop-view failures
in `restaurant_prop_validation.md` remain failed; reaching waypoints does not
repair them. Static occupancy memory remains a positive-geometry prior and
cannot be enabled for moving humans.

## Reproduction and regression

Use the command in `restaurant_service_build.md` with a fresh output directory,
plus `--restaurant-route service --motion --maximum-seconds 90`. All other
accepted options remain: .05 simulation pacing, QP budget 1000, nonlinear rescue
20, planner candidate age .5 s, visual retry 5 s / one track, immediate replans,
vectorized fields and complete JSONL logging. No environment packages, models,
weights or source restaurant files changed.

Raw records, waypoint events, source-age/prior audit and the actual route plot
are under `outputs/restaurant_service/service-route-v1/`. The plot now supports
real-prop monitor bounds and multiple nominal waypoints; it explicitly omits
room geometry from its display. Generate it with
`scripts/evaluation/cra_navigation_plot.py --source <run-directory>`.

Targeted route/nominal/async-inference/prop regressions: 60 tests passed.
Full regression: 1,746 unit tests passed; repository Ruff and code/document
diff checks passed. PR #117's stricter moving-view main-merge hold remains
unchanged.
