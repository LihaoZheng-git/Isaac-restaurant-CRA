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

## Validation and accepted continuous mission

The28 new tests cover missing/expired/ambiguous tracks, original clocks,
short-gap ID recovery, mass exclusion, bounded raster work, actual field
integration, transaction rollback, whole-region coverage and CBF behavior
with distant versus overlapping uncertainty. The 28 focused tests pass,
including actual CLI validation for the three new recipes. The full unit suite
passed **3,357 tests in 133.33 s** at `abb6e12`; Ruff and `git diff --check`
passed. The earlier 215-test focused integration run also passed.

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
has 28 focused passes, including actual argument validation for all three new
recipes without starting Kit. All four prerequisites were rerun on `abb6e12`.

At `abb6e12`, all four `outputs/tracking_uncertainty/*-v2` prerequisites passed
with unchanged sources: component (115 sources), idle/release (180 s, zero idle
drift, 3.470465 m native travel), startup (60 stationary sources), and near-human
(60 stationary sources). The distant startup's raw mass-retention diagnostic
has zero human samples and is not a passed five-source retention benchmark.
Its explicit distant-start provenance gate passed; the separate near-human
and complete navigation retention gates passed their full requirements.

`long-motion-v2` was interrupted when its execution session disappeared. No
native process remained, final report was absent, and the last records showed
normal forward motion. Its exact termination cause is unknown. Preserve the
partial folder and `interruption-note.json`; it supplies no navigation or
stopping acceptance. Only the mission was restarted with the same four passed,
matching-source prerequisites, using a hidden detached launcher.

**`outputs/tracking_uncertainty/long-motion-v3` passed** on frozen `abb6e12`:

| Measurement | Result |
| --- | --- |
| Route | All 8 route guides, actual final goal and stopping passed |
| Navigation simulation time | 150.816667 s |
| Live fields | 936; 935 committed while translating |
| Independent minimum clearance | 0.391756 m |
| Minimum analytical CBF margin | 0.079385 |
| Stopping drift | 0.002174 m |
| Nearby parallel-walk deviation from nominal polyline | 0.400339 m |
| Supported simultaneous human/robot kinetic rows | 14 |
| Furniture coverage proofs | 1,408 retained-prior and 180 fresh-surface proofs |
| Expired-human uncertainty records | 20; independent containment/source/risk audit passed |

The original human identity expired late in the route. Its uncertainty region
used the last verified LiDAR source without clock renewal: source ages ranged
from 0.2 to 3.9 s, giving radii from 2.4 to 6.1 m. The region initially lay
outside the local grid and later occupied 1,920 cells behind the robot. This
exercises distant identity expiry, **not natural close-human occlusion**. All
20 current/available-future containment checks passed (minimum margin
1.908964 m); 19 had the full future horizon available. The last record had only
86 future samples before the recording ended, explicitly marked incomplete.
The separate injected-gap replay of recorded parked-human data also passed;
it remains a module replay, not an additional wheel trial.

The existing independent smoothness metric found no translation hold below
0.02 m/s during the 240 nearby walking samples; the longest interval below
0.05 m/s was 0.033333 s. This is a single-run descriptive result, not a matched
causal improvement or a claim that the whole mission has no pauses.

The full mission contains 91 accepted generic visual instances but **zero
complete fresh four-factor assessments**. Numeric track policies use explicit
factor/property/body priors where evidence is unavailable. Keep that distinction
in the paper and PI presentation. The moving-human geometry, current kinetic
evidence and separate visual/body-specialist audits do not imply full generic
four-factor perception. Tabletop props do not establish fork-tip discrimination
or moving-camera thermal acceptance; no live plume or temperature claim is made.

The full native video and route plot are:

- `outputs/tracking_uncertainty/restaurant-continuous-v3.mp4`
- `outputs/tracking_uncertainty/long-motion-v3-result.png`
- `outputs/tracking_uncertainty/long-motion-v3/parallel-smoothness-summary.json`

The presentation retains all 1,519 navigation/stop frames at 10 fps; only 69
initial stationary frames are trimmed. This is paced simulation presented at
1x simulation time, not real-time inference. The report SHA-256 is
`aadc706a57ee5aa70462c866625606d39f6abc1c9ecce2c534bf1bbb0d65fbff`.

Earlier furniture v6 remains a failed full mission, and both v1 CLI failure
and interrupted v2 remain preserved. This acceptance is for the separate
guidance/moving-arc, furniture-memory and tracking-uncertainty experiment,
with fixed lambda 1.25 and unchanged safety margins/clocks. It does not promote
the primary controller, establish repeatability/general occlusion safety, or
prove a causal CRA benefit against baselines. Keep PR204 draft into its
development parent and main/primary/PR117 holds unchanged.
