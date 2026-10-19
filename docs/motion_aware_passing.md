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

The new zero-wheel mirrored-human component at source a487d93 passed actor
walking/arrival/stopping, grounded/collider/layout checks and fresh human
perception/motion checks. Its 52 selected field rows passed a separate post-hoc
identity/selection-clock/lambda/flux audit. That audit used observed actor truth
only for scoring; the policy selected from LiDAR tracking. The original report
and separate post-hoc report are preserved in
`outputs/motion_aware_passing/native-left-stationary-v1/`.

Wheel validation is still pending. The prospective short comparison retains
the original northbound human-service route and uses a separately declared
left-to-right actor crossing, .5 m/s requested native walk and exogenous start
13 navigation seconds. Both policy-off/on trials share all settings except the
passing-policy opt-in. The native actor parameters never enter the estimator.
The existing stationary human gate is repeated before each wheel trial.
No wheel trial or pronounced curved-bypass acceptance has been established yet. A side
preference remains soft; an almost straight passage or safe stop is not a
validated large detour. Actual navigation needs endpoint/stopping, clearance,
live kinetic/field/geometry checks and an independently measured trajectory.
The new behavior's gate additionally requires at least .20 m left/trailing-side
excursion during the moving encounter, a trailing projection <= -.10 m at
closest approach, and return within .15 m of the original goal. Independent
arrival, stopping, human kinetics, all occupancy and exact safety gates remain.
These are prospective bounded demonstration criteria, not calibrated social
distances or a theorem. The long service demonstration, prior study failures
and main hold remain.
