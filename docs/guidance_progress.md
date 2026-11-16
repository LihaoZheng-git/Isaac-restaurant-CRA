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

## First native retry, 38a9b27: retained failure

The progress-only correction passed its stationary gates but failed
`progress_timeout` at 35.016667 s, 1/3 waypoints. Minimum full-sweep independent
clearance was .724725 m, stopping drift .000987 m. Human mass/motion checks now
passed with six supported moving-kinetic/two simultaneous rows, while zero
complete four-factor assessments remained. The first natural identity-prior
selection occurred at LiDAR source 18.8 s with a .1 s original camera age;
14 held-policy rows passed independent identity/provenance checks. Visible arc,
behind and arrival still failed. This is component validation, not bypass success.

Late references now had positive but small horizon gain (about .021 m), yet
continued to request .10 rad/s before translation. The exact-time plan/command
join found no final-CBF changes in 66 accepted rows after 24 s. In the stable
27--33 s interval, requested yaw integrated to .598240 rad while observed yaw
changed only .017182 rad (ratio .02872), with maximum requested forward speed
.0000924 m/s. These are declared USD ego-pose observations in this simulator
stack, not an identified tire-friction, motor-deadband or general actuator law.
They expose the mismatch between an ideal predicted turn and actual feedback;
positive planned progress does not establish realized progress.

## Turn-first pool correction

The experimental ablation's default turn-first pool now uses .4/.8 rad/s caps,
retaining the moving .4/.8 candidates. The explicit
`--guidance-turn-first-minimum-cap .1` restores the older .1/.2/.4/.8 turn-first
pool. The .4 setting is a bounded search-pool choice motivated by this observed
low-command under-response; it is **not a lower bound on executable angular
velocity**, an actuator calibration, a compensation gain or a change to wheel
mapping/drive/physics parameters. Candidates and optimized outputs still need
all original exact checks and actuator limits; no useful reference delegates
to the unchanged primary. Straight aligned commands may still have zero turn.

Diagnostics additionally record current guidance, initial state and time to
the reference's first translating interval. Preserve the progress-only native
failure and its mass/identity component successes separately from the next
frozen-source stationary and crossing retry. Main/long-service stay held.

The revised pool passed all 2,195 unit tests and repository Ruff. Its command-
only virtual ORACLE routes arrived in 33.2/29.5 s with the same exact checks.
Those virtual results do not validate physical turn response. The next native
retry keeps the scene/route, .5 s identity option, all evidence/geometry settings,
90 s total bound, 12 s watchdog and 60-frame stationary gates unchanged.

## Second native retry, ed4fd7a: retained failure

The stronger-pool trial also failed `progress_timeout` at 34.633333 s, 1/3
waypoints, .597057 m full-sweep independent clearance and .000195 m stop drift.
Human checks passed with five supported moving-kinetic rows. Near the final
stall, accepted primary-fallback plans often requested zero commands rather
than a useful maneuver. Current guidance was about (-.846,-.302), while the
active goal was northeast of the robot. The bounded heading/radial cap prevented
following an outward maneuver, while following the raw normal alone is also
not a correct goal-directed navigation rule.

A separate uncommitted direct-outward-heading prototype passed unit checks but
failed **both** 45-second virtual arrivals. It was rejected before wheels. Its
source/patch are archived in outputs/guidance_progress/archived-rejected-outward
and its negative results in virtual-v3-outward; it is not an integrated mode.

## Filtered goal-velocity reference

The new explicit `--guidance-reference-mode filtered_goal_velocity` keeps
`bounded_field_heading` available for historical reproduction. The local code
uses v_sem as the spatial **CBF constraint normal**, not a stand-alone desired
heading. The corrected reference therefore passes a waypoint-directed planar
velocity and zero angular preference through the existing paper-space safety
filter, then tracks the resulting planar direction/speed with a bounded
unicycle rollout. The original waypoint nominal/cost remain unchanged; this is
still the separately reported guidance-tracking ablation.

The instantaneous projection is only an auxiliary reference: converted horizons
and optimized outputs still require all original exact temporal DTCBF/dense
geometry checks, and actual commands still require current-field revalidation
and per-step analytical CBF. The planar projection's angular component is not
directly applied to wheels. Preferred speed is bounded by both the original
forward preference and projected planar speed. The legacy goal-heading cap and
fade do not apply to this mode, since the goal-directed input and CBF projection
already define the direction. No physical factor or missing evidence changes.

Reference quality uses nonzero **endpoint displacement** in this mode, not a
hard requirement to decrease waypoint distance. Candidate ranking retains goal
gain/heading improvement, but accepts a checked translating reference even when
all gains are negative. A turn-only reference remains excluded. This permits
temporary detours without imposing a new executable progress constraint.

Unit checks verify that an opposite-pointing normal does not turn an already
feasible goal velocity away; a binding spatial constraint bends the reference;
and a valid instantaneous projection cannot bypass an infeasible temporal
horizon. Command-only virtual arrivals and the native retry are recorded
separately below after validation. Main and long-service remain held.

## Third native retry, 8fc7af6: progress improved, gate still failed

All 2,201 unit tests and repository Ruff passed. Command-only virtual ORACLE
routes arrived in 24.45/21.85 s. The filtered-reference native trial passed its
60-frame stationary gates and reached two of three waypoints, but failed
`repeated_inference_failure` at 67.933333 s, .170246 m from the final goal. The
unchanged actual-arrival tolerance remains .10 m. Minimum full-sweep independent
clearance was .335345 m and stopping drift .000319 m; neither establishes arrival.

Human validation passed with 12 supported moving-kinetic/five simultaneous rows.
All 640 source-aligned envelope containment rows passed (maximum centre error
.165914 m). There were 76 independently validated identity-prior selection rows,
with .3 s original visual age at selection. The visible encounter arc passed
at .242512 m, while the behind-actor projection failed at +.671023 m. The full
bypass experiment therefore remains unaccepted independently of the final stop.
There were 640 live fields, 505 committed while translating, 147 accepted visual
instances across 43,425 object assessments and zero complete four-factor
assessments. Retained properties and factor priors remain explicit.

At the endpoint a 5.12 s perception computation returned at .40 s source age,
exceeding the unchanged .35 s field limit. The next captured LiDAR source was
75.7 s after 75.4 s, exceeding the envelope's .25 s continuity limit; subsequent
envelope failures blocked field publication. This is not permission to extend
identity, factor, field or command clocks. Continuity errors now identify each
failed gate and report source gap, original identity age and centroid residual
without changing acceptance or committing rejected state. A separate slower-
paced retry uses simulation rate .025 instead of .05, with all physical,
uncertainty and arrival settings unchanged. It is a paced simulation diagnostic,
not a latency fix or real-time claim. Preserve this failure and main holds.

### Independent scoring correction

The original passing report above remains untouched. Its nearest "moving"
encounter at source 74.866667 s was actually `stopped_after`: idle root animation
briefly crossed the .3 m/s speed threshold. The corrected evaluator requires
both native `move` phase and the original speed threshold when complete phase
metadata exists; partial/unknown phase metadata is rejected. Phase-free older
fixtures retain explicitly labeled velocity-only scoring. This evaluation truth
does not enter perception/control, and independent physical clearance still
covers all phases.

A separately saved `phase-corrected-passing-validation.json` re-evaluates the
same unchanged third-trial records. Its real closest walking encounter was
18.1 s, with behind projection -.180254 m (passes). However, its actual walking-
window arc was -.004212 m (fails the unchanged .20 m threshold); the earlier
.242512 m curve happened later. This is **not** a retroactive bypass pass. Both
arrival and the pronounced curve during the crossing remain unmet. New native
reports use the corrected evaluator and preserve its basis/excluded-row count.

## Fourth native retry, c2952d5: route completion, proactive arc still fails

The separate .025-rate retry completed all three waypoints at 76.216667 s and
stopped, with .385703 m minimum full-sweep monitored clearance, .001192 m
braking drift and .082242 minimum sampled CBF margin. The first two waypoint
events were 15.066667/48.933333 s. It retained the .10 m arrival circle, 90 s
bound, 12 s watchdog, identical scene/actor timing, all priors, solver budgets,
lambda 1.25 and all original freshness/exact safety gates. All 60 stationary
frames passed before motion. There were no live inference rejections; the
startup camera/LiDAR epoch rejection remains in the report. This slower paced
run took 3,229.8 wall seconds, so it is not real-time acceptance or a runtime
latency repair.

Human validation passed: 11 supported moving-kinetic/five simultaneous rows,
.166314 m/s median steady world-velocity error and .016312 m/s median relative-
speed error. All 761 source-aligned body-envelope rows passed. The original
.3 s-old person identity selected the side policy across 80 independently
matched rows, retaining its original eight-second expiry. Across 50,517 object
assessments there were 175 accepted visual instances and zero complete
four-factor assessments; 201,183 factor-prior and 675 persisted-property uses
remain explicit. There were 761 live fields, 542 committed while translating.

`completed=true` is actual route completion. Overall `passed=false` remains:
the corrected walking-window arc was only .005813 m against the unchanged
.20 m criterion. Behind projection -.181119 m, original identity/clock checks,
goal rejoin and all other passing checks passed. Its late wide detour is not a
pronounced proactive curve during the crossing. The launcher returned zero
while the demo deliberately raised its failed full-gate exception; inspect the
report, not launcher status. Main/long-service and the draft PR stay held.

Reference diagnostics isolate the remaining weakness: only 4/134 accepted
plans in 20--40 s, 11/126 in 40--60 s and 1/91 after 60 s had an exact-safe
auxiliary reference; most used the primary fallback. A synthetic, explicitly
unintegrated projection audit also shows that the free-angular planar QP can
rely on rotation later omitted by the direction-only path builder. These
findings motivate a separate bounded forecast-aware planar reference, not a
higher tracking weight, relaxed checks, altered physical scores or a new
primary objective. The outputs-only prototype is not runtime integration.
