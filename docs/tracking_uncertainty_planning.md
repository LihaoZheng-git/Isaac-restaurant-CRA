# Tracking uncertainty as planning geometry

The user approved replacing automatic tracking-expiry stops with explicit
uncertainty supplied to planning. The nominal waypoint command remains
obstacle-unaware. Online MPC, exact dense/DTCBF checks and the analytical CBF
still decide which commands can execute.

## Implemented scope

`--human-tracking-uncertainty` is separate and disabled by default. It requires
the existing human-body prior and bounded recovery mode. A known human whose
track disappears, whose visual identity expires, or whose association becomes
ambiguous produces a separate reachable-region geometry prior. The original
identity is retained, without renewing its camera or LiDAR source time.

The disk remains centered at the last verified LiDAR centroid. Its radius is
the configured body/centroid radius plus the configured1m/s walking-speed bound
times **original source age +1.55s**. The latter covers the1.2s MPC horizon and
.35s field age. Every direction, including reversal, is covered by this prior.
This is an engineering bound for the selected walking-human scenario, not
general human-motion calibration or a new measured velocity.

The occupied part of that disk is added to the current local grid. Its full
extent stays in diagnostics; raster work is bounded by grid size. Actual
current returns remain untouched. Applied risk is at least the original body's
applied risk and the configured unknown-factor floor. Missing evidence is not
zero, and no factor, identity, mass or passing-side observation is refreshed.
Current-body consumers exclude uncertain identities. A truly fresh accepted
association can restore normal processing; the existing unique short-gap
reassociation remains limited to .6s and cannot merge arbitrary old identities.

The nearby-track watchdog permits continuation only when its exact original
positive surface and source are represented by the typed current uncertainty
snapshot and the whole region has sufficient current occupancy/risk/flux.
Incomplete, clipped or weakened coverage does not bypass the watchdog.
Uncertain regions also exclude stationary-furniture enrollment. All memory is
staged with successful field publication, so a failed field cannot commit it.

This complements the already implemented stationary-furniture retention and
fresh whole-surface association. It is not yet general belief tracking for
arbitrary unknown moving objects. Extended sensor loss can make the possible
occupancy block motion; that is a geometric uncertainty constraint, not proof
that a physical obstacle completely blocks the restaurant. No unconditional
arrival or never-stop claim follows.

## Validation plan and current status

The28 new tests cover missing/expired/ambiguous tracks, original clocks,
short-gap ID recovery, mass exclusion, bounded raster work, actual field
integration, transaction rollback, whole-region coverage and CBF behavior
with distant versus overlapping uncertainty. The215 focused tests pass,
including unchanged legacy cases; Ruff passes. The full unit suite passed
3,354 tests in121.46s after final integration review; native results remain
pending.

The separate recipe is `configs/restaurant_tracking_uncertainty_v1.json`.
It adds only the new opt-in to the explicit10m/furniture recipe. The four
prerequisites remain: parked moving-human component, long idle/release,
route-start stationary view and near-human stationary view. All must pass on
matching frozen runtime sources before the full route. The final run must
reach its actual final goal and stop, pass independent geometry checks and
preserve every failed result. The independent uncertainty audit checks current
and available future actor containment; missing final future coverage is
reported explicitly and never fabricated.

At0d3250f, the parked component passed115 sources and all human/furniture
checks. The idle/release gate passed180s with zero idle drift and3.470465m
native travel. Route-start v1 failed before Kit/wheel startup because the new
CLI validation referenced the component runner's flag name instead of the
navigation runner's `predictive_human_geometry_ablation`. The corrected wiring
has28 focused passes, including actual argument validation for all three new
recipes without starting Kit. Fresh frozen-source gates are required again.

No new full native run has yet been accepted. Earlier furniture v6 remains a failed
full mission despite its passed furniture/human encounter. Main/primary/PR117
holds remain unchanged.
