# Guidance-reference progress correction

The user requested a fix after the 1b9bd2b native trial failed route completion.
The original failed reports and main/long-service holds remain in place.

## Reproduced defect and bounded correction

The guidance ablation previously ranked safe references by goal gain plus
heading improvement. A six-second reference with **zero translation** could
therefore receive a positive score and a large tracking weight. Replanning
could repeatedly reward this stationary path. This is an auxiliary reference
selection defect, not evidence of extra final-CBF clipping in the audited runs.

A deterministic synthetic yaw-mask field reproduces it without mocked solver
outputs: the primary chooses a checked .05 m/s forward command; the old ablation
chooses about .0000275 m/s and .10 rad/s while tracking a zero-gain turn-only
reference. The regression preserves the field, nominal and all safety checks.

The enabled ablation now requires a checked reference to predict at least
.01 m of waypoint-distance gain over its horizon, or reach its existing .04 m
reference goal region. The threshold is capped by the remaining distance to
that region. This is a **reference-quality gate**, not an executable constraint
on the primary MPC: useful turn-first references remain eligible; when none
qualifies the unchanged primary controller can still turn, wait or take a detour.
No risk score, uncertainty rule, field, occupancy or safety check changes.

`--guidance-minimum-reference-progress .01` is the explicit development default.
Zero reproduces legacy eligibility for audits. Weight zero still delegates
directly to the original controller. Diagnostics record the threshold and
the count of exact-safe references rejected for insufficient progress.

The command-only ORACLE fixed-field audit passed both three-waypoint sides at
35.6/31.7 seconds with unchanged exact geometry/DTCBF/CBF checks. It is virtual
motion, not native perception or wheel acceptance. Unit/regression and native
stationary/crossing outcomes will be recorded after execution.

The previous native moving-kinetic coverage audit also found repeated failed
SAM associations and temporarily unobservable views before the first accepted
moving human instance. The .25 factor priors and original 3-second human mass
expiry remain unchanged. Controller progress does not repair missing visual
evidence or establish full four-factor perception.
