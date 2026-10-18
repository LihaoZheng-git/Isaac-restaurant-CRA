# Native carried-bowl development diagnostic

The official Isaac adult can pick up and carry the existing audited CC0 Kenney
`bowl-soup` mesh using `pickup_object(..., snap_to_hand=True)`. This is native
hand attachment, not a bowl manually translated alongside a person. The first
four zero-wheel trials passed the attachment/geometry subcomponent but failed
the complete actor arrival/stopping gate. **Carrying navigation is not ready;
keep this branch draft.** The separate standing/slow-crossing navigation step
in PR #130 is already merged into development; the main hold is unchanged.

## What the diagnostic checks

`restaurant_human_carry.py` references the existing local converted asset in the
session layer, checks its original asset audit and records its fingerprint.
The chosen width is .30 m. A kinematic rigid-body/convex-hull collision proxy
supports native pickup; the visible cavity/food mesh remains intact. No mass
API, measured weight, temperature, fall height or risk profile is authored.
The bowl's initial pose is fixture setup only. The native snap bypasses grasp
planning and table pickup; no fluid or spill dynamics are simulated.

The observer reads actual native hand transforms, native held-object state and
PhysX tensor prop poses. It requires a stationary attachment gate before the
actor walks, bounded hand-relative position/orientation, an upright/raised
prop, positive independent authored-layout clearance and actual prop motion.
All robot wheel commands remain zero. These oracle joint/physics references
are evaluation only, never CRA estimation inputs. Source scene and asset bytes
are checked independently. The camera is a diagnostic view, not a new robot
sensor or evidence source.

The original `--carry-bowl` component is limited to the .5 m/s actor path from the existing native
start to `(29.8,10.8,0)`. It is intentionally absent from the wheel-navigation
runner in that original endpoint mode. A carried bowl has **no validated separate sensor ownership** yet.
An attached bowl, its asset name or its authored elevation never supplies
thermal, potential or mass evidence.

## Actual outcomes

| Trial | Native stop command | Complete result |
| --- | --- | --- |
| `native-carry-v1` | Idle after MoveTo success | Failed: .570080 m post-task root drift and final endpoint miss; attachment passed |
| `native-carry-v2` | Zero preferred speed plus Idle | Same failed root trajectory; attachment passed |
| `native-carry-v3` | Zero speed, explicit current-facing Idle, longer observation | Failed: .617287 m root drift; attachment passed; no imported custom actions were available |
| `native-carry-v4` | Keep auto-braked MoveTo terminal state, zero speed, no new Idle | Actual final arrival passed, strict stopping failed (.298263 m root drift versus unchanged .03 m limit); attachment passed |

In v4, maximum hand-relative position error was .000677 m, maximum bowl tilt
14.6547 degrees and minimum independent bowl/layout clearance .423205 m. The
native API reported zero linear velocity and zero preferred speed while actual
root/collider/prop poses still moved. Task success or that API velocity is not
physical stopping acceptance. The original failed reports remain unchanged.
This installed Isaac launcher returned exit code zero even after a script
RuntimeError. Read the saved validation flags and error log; launcher exit
status alone is not a passing simulation result.
No teleport, animation freeze, looser stopping threshold or environment upgrade
was used to manufacture a pass.

The actual image/video under `outputs/restaurant_service/native-carry-v4/`
shows working native carrying and the unresolved endpoint transition. The
12 new unit cases cover session-only source/collision authoring, true prop
motion, attachment failures, invalid/repeated observations, posture and layout
rejections. They do not replace Isaac execution.

## Next bounded gate

The additional `native-carry-sensor-v1` run uses the existing parked-robot
RGB/LiDAR worker and unchanged full-instance gates. Its result is separate
from attachment and stopping. Its human benchmark passed: three stationary
and six steady-moving fresh visually associated adult mass priors, .013513 m/s
median moving velocity error, four numeric policy factors and no property
borrowing. All 127 video frames were distinct, and source scenes/assets stayed
unchanged. However, overall stopping still failed (.395015 m root drift), and
separate bowl ownership remains unvalidated. Maximum result completion age
was .416667 s, exceeding the live control runner's .35 s cap; this is not a
freshness/latency acceptance result for carrying robot navigation.

The current attachment summarizer also rejects nonfinite prop poses and names
its terminal observation explicitly: a passed attachment check never means
the actor passed stopping. Current unit coverage includes 12 new cases.
The full regression passed **1,880 unit tests**; repository Ruff and whitespace
checks passed. No environment, dependencies or model weights were changed.

Before robot carrying interactions, settle the native endpoint behavior or
explicitly select a continuously walking server scenario with its own bounded
trajectory acceptance. Do not relabel a failed stop as a pass. Then validate
fresh body/bowl ownership and retained physical occupancy with zero wheels.
Potential needs justified support/landing evidence; a hand height is not fall
height. Visible plume is a separate cue method, not soup identity/temperature.
The moving-view plume gap, factor-confidence/scale study and matched navigation
baselines remain separate work in `restaurant_final_experiment_plan.md`.

## Selected continuous-server scenario

The user selected continuous crossing on 2026-10-18. The original five failed
endpoint-stop reports stay failed. The new explicit `--continuous-crossing`
parked diagnostic and `--restaurant-human continuous-crossing --human-carry-bowl`
live selector use a separate .5 m/s native actor circuit through
`(28.0,8.9)`, `(28.0,11.4)`, `(29.8,11.4)`, `(29.8,8.9)`.
These are authored actor-scene waypoints, never robot nominal/control inputs.
Native MoveTo uses `auto_brake=False`; actual root proximity within .25 m
requests the next leg. SDK success without actual progress is rejected.

The new independent actor gate requires actual traversal from the right side
to x<=28.3 near y=8.9, articulated feet, bounded observed speed, grounded root,
aligned PhysX collider, positive layout clearance and actual motion over the
last observed second. The .4 m body envelope then clears the x=28.8 robot
centerline. The circuit's turn differs from the old stopped endpoint; it does
not pass or replace that old .03 m stopping test. The bowl retains the same
actual hand/PhysX attachment, posture and geometry checks, using excursion
rather than endpoint displacement for a returning circuit. Robot experiments
still require all actual waypoint arrivals, safe stopping, current inference/
field freshness, positive geometry and unchanged exact MPC/CBF checks.

The independent robot monitor additionally includes a conservative circle
around the actual bowl's rotated world bounds, through the robot's final
stopping observation. These oracle guard/readback values never supply CRA
geometry or factor scores. All observed LiDAR occupancy retains its original
runtime treatment. Human mass remains a visually selected adult prior, and
unsupported bowl/body factors keep labeled priors. Separate bowl ownership,
hand-to-landing fall evidence and moving-view thermal estimation remain open.

The actual `native-carry-continuous-sensor-v1` parked test passed the separate
continuous actor, attachment and human sensor gates: observed median native
speed .493773 m/s, .444205 m displacement in its final second, .436268 m
minimum actor/layout clearance, .398641 m minimum bowl/layout clearance and
.000765 m maximum hand-relative position error. It had eight stationary and
24 steady-moving fresh visually associated adult-prior assessments, with
.037322 m/s median moving velocity error. All 171 recorded video frames
decoded and were distinct; all six scene sources and the bowl source stayed
unchanged. Its .4 s maximum result-completion age is still beyond the live
runner's .35 s cap and does not establish control freshness.

`continuous-validation-current.json` re-audits the unchanged raw actor and
attachment records with explicit continuous-mode names and retains hashes of
the original reports. The original attachment report's old endpoint-observation
check name is not human stopping acceptance. None of the original failed
stopping trials has been rewritten. The full unit regression passed 1,894
tests; Ruff and whitespace checks passed before the coupled wheel trial.

`carried-server-live-v1` was blocked before wheel motion: all three selected
human stationary checks failed over 36 source frames. Small prop-sized SAM
masks were accepted near the server, but full-body proposals failed unchanged
quality/foreign-return gates, so no fresh adult mass prior was selected. One
context-box body proposal had quality .81075 and .3431 foreign-hit fraction;
it was correctly rejected rather than accepted as a full person. The closer
parked-view component pass does not erase this farther-view failure.

A separate explicit initial heading option makes the native carrier face its
first crossing leg (-world X) before acquisition. This changes scene posture
only, preserves the original robot start/3.9 m route/camera and leaves model
quality, ownership, mass and control gates unchanged. It requires another
parked RGB/LiDAR gate before a coupled retry; it is not a general solution to
held-object segmentation or a change to the original failed run.

The farther `native-carry-facing-sensor-v1` observation passed human perception
(eight stationary/18 moving fresh adult priors, .028143 m/s median moving
velocity error) and attachment. Its complete actor test still failed: actual
robot drift was .021443 m over the full observation, beyond the unchanged
.02 m parked limit, with all wheel commands zero. This is a failed parked
trial, not permission to skip that check. Its .45 s maximum completion age
also is not live-control freshness acceptance.

The next separately declared setup uses robot yaw 90 degrees and the same
world start and three world waypoints. `approach-forward` in the parked test
and `--human-start-view approach-forward` in the live runner identify it explicitly. Only
the fixed coordinate transform changes; a unit check reconstructs the same
world waypoint coordinates. All perception, parked drift, navigation and
robot stopping thresholds remain unchanged. Later matched methods must use
this exact heading and actor posture if it passes its own gates.

`native-carry-forward-sensor-v1` did not pass: only one stationary fresh adult
association occurred before the unchanged eight-second stationary timeout;
actor walking and wheel motion remained blocked. The actual closer-view
`native-carry-continuous-sensor-v1` remains the complete passing component.
The next coupled pilot therefore uses that validated robot view (28.8,7.4,
yaw 90 degrees), +15 degree camera pitch and the original carrier posture,
explicitly selected by `--human-start-view crossing-approach`. The three fixed
world waypoints remain unchanged but the robot route is now 2.1 m. This is a
different bounded case, not a retroactive pass or a 3.9 m carrying validation.

That closer `carried-server-near-live-v1` attempt also failed its stationary
human gate with zero wheel motion. Comparing native startup revealed a real
component/runtime mismatch: the live helper started pickup immediately after
idle initialization, whereas the parked diagnostic first let native idle
settle. The live root settled near (29.78,8.94); the passing parked carrying
pose was near (29.55,9.01). The camera change alone did not fix the missing
full-body associations. The live caller now explicitly begins pickup only
after its existing idle/robot settling interval, then checks actual hand/prop
settling before perception. The original 3.9 m route is retried separately as
`carried-server-settled-live-v1`; no old report is overwritten.

The first idle-before-pickup retry still failed the stationary gate (contact
observed, insufficient fresh human mass/current-motion rows); wheels stayed
zero. Its root was near (29.62,9.02). The next `carried-server-aligned-live-v1`
uses the complete parked startup chronology: robot settling, diagnostic camera
settling, native idle settling, then pickup/actual attachment settling. Native
initialization is delayed to the same phase rather than starting pickup during
the initial idle transition. Fixed stepping is explicit, startup metadata is
saved, and the new acquisition window is 60 frames (the existing allowed
maximum). This is a new trial; SAM quality .85 and all other gates stay fixed.

The aligned startup's saved stationary gate passed all three human checks
before any wheel command: fresh visual adult priors, current supported motion
and the observed contact proxy. Pickup was SUCCEEDED and ready at simulation
time 3.416667 s. The subsequent live navigation result remains separate from
this stationary result; it must also pass moving-human assessments, all robot
waypoints, independent body/bowl geometry and actual robot stopping.

The aligned trial **failed navigation**, stopping after its first waypoint
on an unresolved nearby positive track (546). It retained .948958 m minimum
independent clearance, .867332 minimum analytical CBF margin and .004715 m
robot stopping drift across 133 fields (129 translating). Five supported
moving-human kinetic rows and four simultaneous rows passed their independent
perception checks; median world/relative speed errors were .094344/.029862 m/s.
Attachment passed. This is a safe rejected trial, not arrival acceptance.
The continuous-actor final gate also failed because the aborted robot trial
explicitly canceled the actor. No actor stopping result follows.

The lost fragment lay near the human's rear body, not the bowl center. Its
last centroid (29.0883,9.0094) at t=18.9 was .291 m from the next current body
cluster centroid; subsequent velocity queries did not meet the unchanged
.25 m parent-association gate. The watcher correctly retained it and stopped
when the robot came within 1.2 m. No old occupancy was silently declared free.

The next opt-in perception diagnostic increases 3D connected clustering from
.12 to .18 m (`--human-cluster-radius .18`) for this separated articulated
actor. It changes geometric grouping, not sensor returns, mass/score inputs or
safety thresholds. Unit tests preserve every return and the old .12/.20 m
3D/planar defaults. Larger grouping can merge nearby physical objects; masks
and unmatched occupancy retain their existing treatment. Keep all SAM .85,
positive-return, .10/.25 m merge, freshness and exact control checks fixed.
Stationary validation precedes any new wheel trial; no successful wider-radius
navigation or crowd/occlusion claim follows from implementing this option.

The wider-radius `native-carry-cluster18-sensor-v1` component passed every
actor, parked-robot, attachment and sensor check. It recorded five stationary
and 30 steady-moving fresh human priors, .021417 m/s median velocity error,
.35 s maximum completion age, 171 distinct decoded video frames and unchanged
sources. Its physical attachment/geometry trajectory was unchanged from the
original parked case. The subsequent `carried-server-cluster18-live-v1` retry
uses the original 3.9 m route, aligned startup, 60-frame stationary window,
fixed lambda 1.25 and identical control/continuity constraints, plus bounded
copied-sensor snapshots. Its arrival gate is separate from this component pass.

The .18 m live retry still failed on a second disappearing fragment (389),
after 168 fields/167 translating fields and one waypoint. Minimum independent
clearance was .956920 m, minimum CBF margin .730792 and stopping drift .003747 m.
It had 21 supported moving-human/21 simultaneous kinetic rows, with .048243
m/s median world-velocity error and .020048 m/s median relative-speed error.
Attachment and all perception checks passed; arrival did not. Offline copied
scans at t=20.2/22.2 showed identical nearby human grouping at .18 and .20 m,
so they did not justify another radius-only retry. Planar grouping on those
same snapshots also left the nearby groups unchanged; it was not integrated.

The guard's coexistence check compared the parent's continually updated last
timestamp with the missing fragment's original timestamp. At t=21.9 the
parent was within .242 m of the old fragment centroid and its velocity was
stable, but its latest epoch was .2 s beyond the fragment epoch, exceeding
the .15 s coexistence comparison despite actual co-observation at t=21.6.
This shrank the declared .35 s recovery window according to scan cadence.
The copied snapshots do not contain the exact t=21.9 geometry, so this is
timing eligibility evidence, not proof that its positive-return gate would pass.

The correction retains actual simultaneous-observation times for enrolled
track pairs, with their original .35 s expiry. Parent-only updates never renew
that evidence. Recovery still requires the existing stable/current parent
velocity gates, unambiguous current positive returns, .10/.25 m geometry and
one-to-one merge checks. It supplies no body shape, physical factors or free
space. Seven new unit cases cover recovery at .3 s, missing/ambiguous support,
unstable velocity, absent original coexistence and unchanged expiry. Successful
retained-pair events save their exact current positive geometry for an
independent audit. A fresh live retry must pass before merging this branch.

`carried-server-coobserved-live-v1` verified that correction but still failed
overall arrival. Its saved-input independent audit recomputed a .3 s recovery
with 27 unique current supporting returns and .238946 m parent/query distance;
all original source-age, coexistence and velocity/geometry gates passed. The
raw report remains failed: after 214 fields/212 translating fields and two
waypoints, the lower-return circuit brought the server back beside the robot,
and a new nearby fragment triggered the unchanged .15 s loss stop. Minimum
independent clearance was .248459 m, minimum CBF margin .364190 and stopping
drift .004294 m. Thirty supported moving-human/28 simultaneous rows passed
with .069687/.027966 m/s median world/relative errors. This is not completed
navigation or robust repeated-encounter tracking.

The intended next scene is one crossing followed by continued service walking
farther up the aisle, rather than an immediate return to the robot near its
destination. Explicit `--human-upper-continuation` (live) /
`--continuous-upper-continuation` (parked) preserves the original lower-return
variant separately. After `(28,8.9)` it uses `(28,11.4)`, `(29.8,11.4)`,
`(29.8,12.6)`, `(28,12.6)` and repeats those four upper waypoints. The first
crossing is never revisited. An independent authored-layout precheck retained
.102145 m minimum clearance for a conservative .6 m half-extent envelope;
actual stationary body/bowl checks are still required. The actor region is
explicitly extended to y<=13.3 for this variant only. Robot waypoints, factor
policies and every robot safety/arrival/stop constraint stay unchanged. This
new scene cannot retroactively pass the repeated-return trial.

The `native-carry-upper-sensor-v1` parked gate passed actor, attachment,
geometry and human perception checks: three stationary/29 moving fresh adult
priors, .021592 m/s median velocity error, .35 s maximum completion age,
.000696 m maximum hand-relative position error and .387889 m minimum
bowl/layout clearance. All 171 video frames decoded and were distinct, and
source scenes/assets stayed unchanged. The full 1,908-test unit regression
and Ruff passed. The separate `carried-server-upper-live-v1` wheel trial must
still pass all actual robot waypoints/stopping and current perception gates.

## Accepted single-crossing live pilot

`carried-server-upper-live-v1` passed the original 3.9 m robot mission with
the native carrier continuously walking after its one crossing. This is a
different actor trajectory from the failed lower-return circuits.

| Actual gate | Result |
| --- | --- |
| Robot waypoints and stopping | All three reached; .002247 m stopping drift |
| Independent body/bowl/room clearance | Minimum .999485 m |
| Analytical CBF margin | Minimum .494319 |
| Live fields / fields while translating | 225 / 219; maximum committed source age .166667 s |
| Human matched / moving supported / simultaneous rows | 194 / 31 / 30 |
| Median world / relative velocity error | .044209 / .023145 m/s |
| Supported kinetic / visual contact rows | 174 / 37; potential/thermal supported rows both zero |
| Computed human phi | .590074 to .68359375, unchanged prior-aware policy |
| Native attachment | Passed; max relative position error .000697 m, max tilt 11.3374 degrees |
| Independent bowl/layout clearance | Minimum .387847 m; source unchanged |

Human mass uses the visually selected 70 [45,100] kg adult PRIOR. Potential
and thermal remain explicit factor priors; contact is a visible geometry
proxy. Separate bowl physical ownership, justified hand-to-landing fall
evidence, temperature and moving plume are unvalidated. The generic original
report's carried-load limitation refers to those physical-assessment gaps;
the separate native attachment/geometry gate passed. Future report wording
now distinguishes those scopes explicitly, without rewriting the raw report.

Median velocity acceptance is not a uniform error bound: observed world and
relative error maxima were 1.639586 and 1.586481 m/s. No confidence/energy scale
or factor prior was changed to obtain arrival. The run uses fixed lambda 1.25,
oracle ego/extrinsic references, the approximate RGB read clock and paced
simulation (.05 simulation s/wall s); it establishes neither real time,
calibrated injury nor a causal CRA advantage. The renderer history warning
remains. Repeated-return/occluded articulated tracking failures remain failed.

The independent saved-input `continuity-recovery-audit.json` passed. One
retained-coobservation recovery used a current-parent **velocity-advected**
query, original age .3 s, exactly three unique current supporting returns and
.244295 m parent/query distance. It recomputes source/geometry/velocity gates
from the saved live records and exact current-point NPZ. This is distinct from
the earlier lower-return trial's independently checked 27-return unadvected
recovery. Neither audit changes the original trial's completion flag.

All 313 full-video frames decoded and were distinct. `carried-server-motion.mp4`
is a 236-frame, 23.6 s navigation clip at the original 10 Hz simulation-time
speed, with startup trimmed. `navigation-frame-08s.png` and adjacent extracted
frames are unchanged decoded Isaac frames. `video-audit.json` records source
hashes and frame indices. `navigation-validation.png` and
`human-factor-traces.png` show recorded trajectories/scores with priors labeled.
The full 1,908-unit regression, Ruff and whitespace checks passed. Source
scenes/assets and model weights were preserved. Main's separate fresh
moving-view four-factor hold remains unchanged.
