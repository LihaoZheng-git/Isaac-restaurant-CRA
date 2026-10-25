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
