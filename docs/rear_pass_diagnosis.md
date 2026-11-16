# Rear-pass timing diagnosis

This is a separate encounter-policy experiment, stacked on the live rear-pass
implementation. The primary controller and the accepted stop/go fixture remain
available. The original three failed rear reports in `docs/live_rear_pass.md`
remain failed; the unchanged acceptance requires at least 0.20 m of trailing
excursion during actual transverse walking, arrival and safe stopping.

## Observed interruption

The last original native trial (`native-rear-offset12-left8-v3`, report SHA256
`218639ae60fad17fb505acf0a52077d5760fd059a49beaf8c3c2709df732c3d2`)
has a current human centroid disturbance at source 17.2 s. Inferred tracker
velocity changes from `(0.407, -0.051)` to `(0.106, -0.342)` m/s. Its acute
angle to the nominal route falls from 82.7 to 17.2 degrees, despite continued
positive identity and LiDAR support. Classification is uncertain from source
17.2 through 17.7 s. This is a recorded estimation disturbance, not proof that
every centroid error has the same cause.

The executed commands are joined to accepted plans using original optimizer
request times, not field timestamps. At requests 17.75 and 17.9167 s the
ordinary fallback selects angular rates -0.442 and -0.520 rad/s, reversing the
earlier trailing turn. Rear-reference execution resumes at receipt 18.1 s.
These records establish a specific interruption; they do not establish that it
is the only cause of the 0.185079 m failed walking-window excursion. The larger
0.345595 m all-phase bend peaks 1.866667 s after walking ends and cannot repair
that failure.

## Bounded correction

`--rear-encounter-motion-fit` explicitly enables a source-window centroid fit
for **encounter classification only**. It requires the existing checked-rear
experiment. Defaults: 0.4 s window, at least four observations spanning 0.25 s,
at most 0.25 s between source observations, 20 current points, established
matched identity, and maximum 0.06 m position-fit residual. An opposing current
tracker velocity, excessive speed or inconsistent fit rejects the estimate.
Missing IDs and unqualified current support discard their histories.

The estimate is an **inferred average over original observed source epochs**,
with no extrapolation or renewed measurement. Each diagnostic records all
sample times, span, residual, original current velocity and rejection reason.
Rejected fits use the original current tracker velocity in the legacy
classifier; they do not preserve an old crossing label. Existing classification
thresholds and three-observation consistency checks remain unchanged. Fitting
can delay a genuine motion transition, so it is scoped to this separated
crossing experiment rather than general crowded pedestrian prediction.

The fit state is cloned and committed with the existing perception transaction.
CRA kinetic inputs, raw tracking, occupancy, body envelopes/coverage, passing
selection and rear-reference target/checks continue to use original tracking.
The raw current transverse-motion check can still disable a rear reference even
when classification uses the fit. Physical lambda 1.25, the 0.40 + 0.25 m body
prior and every source-age, exact DTCBF/dense geometry, analytical CBF, clearance,
containment, arrival and stopping gate remain unchanged. Actor truth is used
only for evaluation, never for the fit or control.

## Validation state

All 2,671 unit tests passed in 85.71 s with a repository-local temporary test
directory; Ruff, compilation and CLI smoke checks passed. The first default-
temporary-directory run encountered fixture setup errors and was interrupted;
it is not a passing test result. No environment changes were made.

Focused component tests cover rotation/mirroring, source epochs, centroid
disturbances, identity/support loss, gaps, opposing motion, fit residuals and
transaction rollback. A preliminary recorded-position probe reduces the
classification interruption, but is not a counterfactual navigation replay:
the robot's recorded states would change under different commands. Full
regression and a zero-wheel native gate precede any new motion trial. Native
rear-pass acceptance is pending at this source freeze. Keep all main,
long-service and PR117 holds, the declared human/factor priors, oracle ego pose,
RGB read-time approximation, renderer warning and incomplete fresh four-factor
perception limitations.

## Native result at 5ba34c2: timed rear curve still fails

The independent 60-frame stationary run passed with zero robot wheel commands
in 162.84 wall seconds. Its independent source-position audit reproduced all
32 accepted fits and preserved three short-window and two opposing-motion
rejections. The navigation trial then passed its own 60-frame stationary gate.

Original `outputs/paper_flux/native-rear-motionfit-left8-v4/report.json` is
**FAILED**, SHA256
`14cc2683d21ba3d19db0eb820401f2165b8bf72dc9e27f5c3d52a8ea39662217`.
It completed all three guides and reached the goal in 21.65 simulation seconds.
Minimum braking-inclusive clearance was 0.730241 m, stopping drift 0.002659 m
and minimum executed analytical margin 0.260822. All 216 body-containment rows
passed; maximum center error 0.162700 m remains within the unchanged 0.25 m
uncertainty prior. Original source/pair/selection/vector-transform, current
human kinetic, policy, scene/root, arrival and stopping audits passed separately.
The launcher returned zero **with a traceback and failed saved report**.

| Same development fixture | Walking-window arc, required 0.20 m | Goal | Clearance | Overall |
| --- | --- | --- | --- | --- |
| Previous target-offset trial | 0.185079 m | 3/3, 21.433333 s | 0.704079 m | FAIL |
| Source-window classification fit | 0.188876 m | 3/3, 21.65 s | 0.730241 m | FAIL |

There were 30 rear fields, 19 accepted selected rear plans and 18 accepted
walking-window rear-plan sources. The original report joins 121 translating
command rows / 2.016667 s. An independent audit conservatively joins 120 / 2 s.
`execution-clock-boundary-companion.json` reproduces this one-row difference:
recovering the original start `8.199999999999987` from the original joined
command reproduces 121; reconstructing start as `8.2` excludes the exact walking
boundary row. Both execution booleans pass. No original report, source timestamp
or failed curve flag was changed.

The fit improved encounter continuity, but did not fix the entire maneuver:

- The new first rear plan was received at source 17.033333 s, compared with
  16.85 s previously. The fit's onset delay and existing planning cadence matter.
- At source 17.3 s the fitted classification remained crossing, while the
  unchanged **raw** transverse-motion rear-context gate rejected the current
  velocity. This correctly disabled the rear reference under this experiment's
  declared contract; classification alone cannot establish reference readiness.
- At source 17.6 s, searches had 117 exact-feasible candidates but no selected
  complete reference after the future analytical, translation, first-interval
  quality and utility checks. Ordinary fallback remained necessary. New-field
  horizon revalidation also caused eight stops plus seven awaiting-plan rows
  in the diagnostic walking interval. These are safety stops, not intentional
  checked yields and not grounds to weaken acceptance.
- The maximum all-phase bend, 0.364912 m, still occurred 1.983333 s **after**
  transverse walking ended. It cannot replace the failed walking-window arc.

These observations identify several contributors rather than one proven sole
cause. The source-window fit is an optional tested component, **not an accepted
native rear-pass strategy or a new default**. There is no mirrored rear wheel
acceptance. This is a single development run, not a replicated or held-out
comparison and not evidence of a causal CRA advantage.

The navigation record contains 68 accepted visual instances, 14,514 object
assessments and **zero fresh complete four-factor assessments**. The independent
fit audit reproduced 237 accepted fits (29 stationary, 208 live), with original
centroid epochs, raw physical velocities and classifier inputs verified; ten
opposing-motion and three short-window rejections remain explicit. Human/factor/
body priors and the previously documented sensor/clock/rendering limitations
remain.

## Practical choice after this bounded attempt

Use the **previously accepted checked stop-and-go fixture** for perpendicular
crossing: the continuing-away left-to-right pedestrian with 10 s onset,
documented in `docs/live_checked_wait.md` and
`outputs/paper_flux/live-checked-wait-motion-v2/`. That earlier original report
passed actual arrival, intentional checked waiting, clearance and stopping.
It is not a new validation of stop/go on this 8 s guide-mode fixture, every
timing or the mirrored case. Parallel departure keeps normal applied guidance;
do not globally disable lambda. Keep rear development optional, preserve all
four failed native rear attempts and retain the main/long-service/PR117 holds.

Any future rear development should address coherent encounter prediction,
reference feasibility and planning/execution handoff together. Do not keep
increasing offsets, remove geometry, lower the 0.20 m criterion, renew clocks or
relabel late curves to obtain a pass.

Local ignored artifacts include both native folders' invocations and gate
records, `independent-motion-fit-audit.json`, the motion trial's
`independent-rear-audit.json`, the clock-boundary companion,
`rear-motionfit-timing-diagnosis-v4.json`,
`native-rear-motion-fit-comparison-v4.png` and the native
`human-navigation.mp4`. Source code remains frozen at 5ba34c2; subsequent result
documentation does not change the tested runtime.
