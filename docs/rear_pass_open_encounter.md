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
