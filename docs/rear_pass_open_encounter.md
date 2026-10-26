# Separate lower-aisle rear-pass experiment

The preceding person-relative experiment reached its goal but failed rear passing.
Its curve-bearing requests started in an occupied human configuration-space cell;
the original failed reports are unchanged. A proposed curve is not execution.

This branch tests a separately declared, more open restaurant encounter. The
native adult walks at the unchanged 0.5 m/s from (25.8, 7.0) to (29.8, 7.0),
then turns into the original upper aisle at (29.8, 12.5). The robot starts at
(27.0, 5.1), facing +Y, with obstacle-unaware guides (27.0, 6.2), (27.0, 7.1),
and (27.0, 7.9). An actor-only robot-y6.4 trigger coordinates the encounter.
These poses/targets belong to rendering and independent evaluation; they never
supply physical factors, identity, actor velocity or safe-space geometry.
Original restaurant assets are unchanged. This is a different fixture, not a
matched improvement of the earlier narrow-aisle result.

`--rear-delayed-commit` is separate and default-disabled. A current qualified
motion observation must place the person at least 1.05 m ahead along their
motion direction relative to the robot before a curve can freeze. This is a
behavior preference, not proof of clearance. The original exact configuration,
dense geometry, DTCBF, forecast analytical and first-command CBF checks remain
mandatory. A frozen compatible curve may continue beyond the legacy 1.2 m
lateral release preference; it cannot outlive its original three-second/8-second
source/selection bounds, survive identity/motion loss, or restart after loss.
No long-horizon human occupancy forecast or new uncertainty prior is introduced.

Robot nominal/MPC speed limits remain 0.20/0.25 m/s and angular limit 0.8 rad/s.
Body radius 0.4 + centroid uncertainty 0.25 m, mass/contact/factor priors,
physical lambda1.25, the separately declared human b=-1.7 control ablation,
source/field/command clocks, containment and braking checks remain unchanged.

Acceptance still requires the unchanged 0.20 m walking arc, actual rear-reference
execution, crossing the observed human plane during transverse walking and
behind motion, actual waypoint arrival, independent clearance and stopping.
Component/unit tests and authored layout surveys are not native acceptance.
Validate the new actor/sensors with zero robot wheels, then an independent
60-frame controller gate and the coupled trial's own gate before motion.
No mirrored wheels until a strict native success. All main/long-service/PR117
holds persist. Native results and final validation are pending.

## First parked component: failed robot-stationary gate

At d36ddf6, the actor and sensor checks passed separately: 9 fresh idle and
27 moving human associations, median motion error 0.09344 m/s, observed actor
speed 0.52225 m/s and minimum furniture clearance 0.28477 m. Zero nonzero wheel
commands were recorded, but the base drifted 0.029382 m over the full trace,
exceeding the unchanged 0.02 m parked limit. Overall acceptance FAILED.
Original report SHA256: 06852610e95f9ed37e9da4b4687a426e57dd11bb524e1c375459b4f7f43081de.
The next retry clears base and joint velocities once after setup relocation,
before settling or sensor capture. This tests a residual-startup-velocity
hypothesis, not an established cause; no pose lock, friction/drive change,
ongoing reset or threshold relaxation is permitted. A new parked pass is still
required before wheels. The delayed-commit mode also uses its own rear-region
eligibility beyond the legacy 1.2 m release preference; original identity,
source/selection expiry and all exact checks remain unchanged.

## Explicitly approved mounted perception benchmark

The user approved a separate fixed robot mount for the stationary perception
benchmark, while retaining the freely moving robot and original physics/stopping
checks for navigation. This supersedes the earlier no-mount setup restriction
only for the stationary benchmark; the original free-standing failure remains.
One-time reset and alternative free-standing pose probes did not pass the
unchanged 0.02 m drift gate (30 s drift: 0.04015, 0.03370, 0.03963 m).
The fixed mount is an explicit session-only world/chassis joint, authored before
physics; it is available only in the zero-wheel actor/sensor program, with no
motion-mode flag. Root asset files, physical risk/geometry inputs and navigation
code do not receive its authored pose. Its isolated 30-second diagnostic passed
with zero drift and zero wheel commands; full human/sensor acceptance is pending.
No freely parked/braking or navigation claim follows from the mounted result.
The first two isolated helper attempts produced no valid result; the second
captured a pose-output parsing error. Both are retained separately from accepted
diagnostics.

## Mounted native component passed

At cf5be51, the full mounted native human/sensor benchmark PASSED with zero
wheel commands and zero robot drift: 9 fresh idle and 24 moving associations,
median world-velocity error 0.087218 m/s, actual actor speed 0.522254 m/s,
minimum furniture clearance 0.284770 m and maximum collider error 2.87e-8 m.
Articulation, crossing, upper departure, grounding and video checks passed.
Original report SHA256: bb840439c5632e7bed92d2380d54f5495b1f0048f4072f718a78424daaadfe67.
This is the separately approved mounted perception benchmark, not a pass of
the freely parked drift test, free braking or coupled rear navigation.

The coupled lower-aisle recipe may explicitly request a 12-second post-goal
zero-command observation. Planning/inference are closed after actual arrival;
this tail merely records the continuing pedestrian's upper-aisle departure.
It is not a scheduled pause during navigation, fresh perception or renewed field
coverage. The original 60-frame stopping-sweep.json remains unchanged. A
separate post-goal report audits all original-plus-tail poses from the ORIGINAL
braking start, requiring clearance >0.02 m, maximum drift <0.05 m and continued
presence in the original goal tolerance. Thus longer observation cannot hide
intermediate contact, drift-return or lost goal. Old modes default to no tail.
Controller/actor/perception code is unchanged from the mounted component;
new independent FREE stationary and own60 gates precede wheel testing.

## First coupled open-aisle result: failed rear passing and containment

At frozen 7a9e7ba, the independent and own60 free stationary gates passed.
The robot reached all three waypoints in 25.016667 simulation seconds, with
0.101086 m minimum independent clearance, 0.011998 m braking-plus-tail drift
and positive reported analytical margins. The post-goal observation passed.
The full frozen rear curve actually executed: 15 accepted walking sources,
131 verified translating rows, 2.183333 s. Nevertheless the strict rear gate
FAILED: only 0.037701 m near-person walking arc, with plane crossing 2.341872 s
after transverse walking ended. Arrival alone is not a rear pass.

One of 249 envelope rows also FAILED containment: center error 0.256388 m
exceeded the labeled 0.25 m uncertainty prior. A transient reduction in observed
support coincides with a centroid/velocity excursion; that is a diagnostic,
not a proven sole cause. Original failed report SHA256:
0ed6ea8db88c51bed70800ee7472d7bb0618e322e96e1c1c54b397024404a007.
There were 76 visual instances and ZERO complete fresh four-factor assessments;
physical factor priors and all original clock/renderer limitations remain.

The old progress subcheck hardcoded 3.5 m despite this configured 2.8 m route.
The prospective correction uses route-directed progress >= configured chord
minus the unchanged 0.15 m goal tolerance, and still requires actual goal
rejoining. The original failed report remains unchanged; the timing/arc and
containment failures remain failures regardless of this metric correction.

## Next separately reported preparation experiment

The user approved testing centroid uncertainty 0.30 m, total body-envelope
radius 0.70 m, in a new trial after stationary validation. Defaults and old
0.25 m reports stay unchanged. No factor score, robot limit or acceptance
threshold changes. The actor-only onset moves to robot y6.0, with identical
human route and speed, to allow earlier preparation.

`--rear-preparation` requires the explicit delayed person-relative experiment.
Before a complete <=3 s passage can freeze, a qualified CURRENT source may
propose a route-anchored 0.35 m motion-trailing-side approach target, with 0.25 m
forward lookahead. This separate maneuver preference uses no retained forecast
or future occupancy. Every horizon remains before the current observed crossing
plane by 0.05 m and must pass the original full exact/dense, future analytical
and first-command checks. Field/command/current-pose gates remain unchanged.
Preparation is rebuilt from each current qualified source, dropped on missing
motion, and cannot restart an invalidated full curve or renew the original
eight-second selection. A complete curve still needs its original 1.05 m
current rear separation and three-second screening/deadline. Preparation plans
are explicitly excluded from full rear-execution acceptance. Legacy modes stay
disabled and unchanged. Native acceptance is pending; no mirrored wheels or
main/long-service/PR117 hold release follows from this change.

Preparation source validation: 2,790 unit tests passed in 108.93 s; 52 final
focused tests passed in 10.49 s. Ruff across robot/scripts/tests, compilation,
CLI help and whitespace checks passed. Trusted original left/right numeric
snapshots reproduce the identical full command sequences with preparation
disabled. These are source checks, not new native rear-pass acceptance.

## Native preparation and timing-only retries: strict rear passing still failed

Both trials ran frozen source **3adb3c6**, the separately approved **0.30 m**
centroid-uncertainty prior (0.70 m total envelope), unchanged physical CRA,
lambda/robot limits and acceptance thresholds. The independent free60 gate
passed in 189.55 wall seconds, with no nonzero wheel commands; one startup epoch
rejection remains recorded. Each coupled trial also passed its own60 gate.
The y6.6 timing-only retry reused the independent gate after an exact frozen-
source/stationary-input comparison: only output and actor trigger differed,
and the actor trigger is disabled while navigation is false. The mounted
actor/sensor component remains separate from free navigation and braking.

| Native trial | Actor trigger y | Goal/stop | Minimum clearance | Total stop/tail drift | Containment | Strict rear result |
| --- | ---: | --- | ---: | ---: | --- | --- |
| v8 preparation | 6.0 m | 3/3, 20.6500 s | 0.368565 m | 0.012040 m | 206/206; max error 0.186551 m | FAILED: plane crossing 3.124941 s after walking; no qualifying near-person walking arc |
| v9 timing only | 6.6 m | 3/3, 29.8167 s | 0.041562 m | 0.011201 m | 297/297; max error 0.200353 m | FAILED: plane crossing 2.075028 s after walking; arc 0.020157 m <0.20 m |

Times above are navigation simulation seconds, not real time. Reported minimum
analytical margins were 0.281102 and 0.171707 respectively. Post-goal checks
passed throughout both full zero-command windows. Launcher exit zero with the
expected gate-failure RuntimeError does **not** mean overall acceptance.
The closer trial's clearance exceeded the unchanged 0.02 m gate by only
0.021562 m; it is not recommended for the final demo. These changed recipes
are not matched causal tests of the prior, preparation or timing separately.

Independent audits reproduced all 53/47 fresh preparation fields and their
original observed source/position/motion/selection, route-anchored targets and
absence of retained forecasts. Accepted preparation plans (11/3) were excluded
from full rear execution. The original execution gate passed separately with
9/13 walking full-reference sources and 76/115 translating rows (1.266667/
1.916667 s). Independent reconstruction conservatively joins 75/114 rows
(1.25/1.9 s); the exact accumulated native start clock reproduces each original
report exactly, including the one floating-boundary row. Companions preserve
the originals. Full source/deadline/identity/forecast and transactional-state
audits pass; 249/334 source-window fits and 75/72 policy records independently
reproduce raw tracking and policy inputs.

Four trusted native numeric snapshots (first preparation and first full curve
from each run) reproduce the complete command sequences and original source
epochs. Their exact DTCBF/dense geometry, future analytical, first-command CBF
and preparation-before-plane checks pass. This validates command dataflow;
it does not repair strict rear timing. Both trials still have **ZERO complete
fresh four-factor assessments**, with 66/93 accepted visual instances among
17,399/24,788 object assessments. Human/body/factor/property priors, oracle ego
and clock references, renderer warning and paced-simulation limitations remain.
No new human mass calibration, plume, physical sharpness or temperature claim.

Original report SHA256:

- v8: `26efb934ff1e618cc791a606bbf9ff1bbb2d7b6288d1c3642791ef0548de843f`
- v9: `9fe1f5a3a91ec5729716ffd4e0e14a01dbed8b3eaa50462d19c3bdfe69866fc6`

Artifacts are under `outputs/paper_flux/native-rear-open-motion-left-v8/` and
`...-v9/`: original `report.json`, `invocation.json`, stationary/containment/
post-goal records, independent audits, `person-reference-numeric-replays.json`,
`execution-clock-boundary-companion.json`, original snapshots and actual
`human-navigation.mp4`. Videos have 406/498 frames at10fps, each readable with
three distinct sampled frames. The adjacent `...-trajectory.png` plots show
executed paths and the walking window, explicitly labeled FAILED acceptance.
`rear-open-report-preservation-final-v1.json` verifies16 selected earlier/new
reports byte unchanged, including old failures and the mounted/free distinction.

The remaining problem is timely productive progress under the current local
forward-only filter/reference, not an absence of a proposed curve. Observed
native walking also curves around the authored horizontal route; the independent
metric uses the median actual crossing plane. Neither that observation nor
successful numeric commands proves one sole cause or global nonconvergence.
Keep the prior accepted left-to-right10s checked wait/go as the perpendicular
fallback and normal guidance for parallel departure. These runs do not validate
a new lower-aisle/mirrored wait/go fixture. Further rear passing would need a
separately designed longer encounter or broader path-reference work, rather
than weaker geometry, clocks or success thresholds. Rear preparation stays
default-disabled/draft; no main/long-service/PR117 hold release or mirrored
person-relative wheels follows from these results.
