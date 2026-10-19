# Motion-aware tangential passing

This opt-in changes the passing-side policy at the online field boundary. It
does not alter the original service waypoints, physical factors, phi, normal
flux mapping, lambda 1.25, occupancy, source clocks or MPC/CBF acceptance.
Fixed-class and CRA field controllers can use the same opt-in. Geometric CBF
retains its original command path; a shadow field does not steer its wheels.

Current LiDAR tracks supply geometry and inferred world velocity. Three
consistent moving observations, positive point-count, speed, transverse-motion
and near-term relative-closest-approach gates must identify a single candidate.
The selected tangent points opposite the crossing object's lateral motion at
the robot-facing boundary: for a robot moving north, an eastbound crossing
selects the westward tangent. These are bounded constant-relative-velocity
selection hypotheses, not measured future trajectories or a safety theorem.

A same-track policy preference can remain latched for eight original-source
seconds to prevent side chatter through the encounter. It uses fresh current
geometry and releases on source/identity loss, reversal, expiry, passing or a
competing crossing. Its original selection epoch/velocity stay visible. This
is controller policy memory, not retained physical evidence or semantic identity.
No missing obstacle is removed. No factors are borrowed or renewed by this path.
The policy is cloned and committed transactionally only after a complete field
and temporal derivative are accepted.

The separate left-to-right native actor fixture moves its initial scene anchor
from the old east side to the west side and reverses the walking endpoint. This
is explicitly authored rendering/evaluation information; no actor position,
speed or direction parameter enters the perception worker or side policy.
The existing right-to-left default is preserved. Carried groups and crowded
crossings are outside this opt-in.

Validation is in progress: focused policy/field regressions precede a new
zero-wheel mirrored-human perception/policy gate. No wheel trial or pronounced
curved-bypass acceptance has been established for this feature yet. A side
preference remains soft; an almost straight passage or safe stop is not a
validated large detour. Actual navigation needs endpoint/stopping, clearance,
live kinetic/field/geometry checks and an independently measured trajectory.
The long service demonstration, prior study failures and main hold remain.
