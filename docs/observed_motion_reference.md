# Observed-motion advisory reference experiment

The user approved further improvement after the no-preparation Carter run
reached its goal but retained a 2.50 s yield. Quiet-person preparation remains
excluded. This separate, default-off experiment changes the local guidance
reference only. It does not replace the current field with a predicted field,
enable the older failed hard predictive-envelope experiment, or shorten any
checked horizon.

An offline attempt to rank the existing references using predicted clearance
selected the same paths on the recorded first-selection/restoration/failure
inputs. That development result is preserved. The new candidate instead adds
a bounded outward component to the existing selected-human tangent when the
unchanged nominal horizon predicts an approaching encounter. The existing
left/right choice and field activation still apply. The nominal waypoint
commands remain obstacle-unaware.

## Inputs and provenance

- Positive current LiDAR track identity, centroid and point count come from
  the existing tracker; velocity is an inferred, filtered LiDAR estimate.
- The existing current oncoming-human selector must be active, with its
  original visual identity proof and current applied body-envelope prior.
  Missing/ambiguous/quiet observations produce an unavailable advisory input.
  They do not become zero physical risk or invented zero velocity.
- The original body radius0.40 m plus centroid uncertainty0.25 m remain labeled
  priors. A forecast is constant current velocity with an explicit configurable
  radius-growth hypothesis, default0.50 m/s. This is an **advisory engineering
  prior**, not a calibrated acceleration bound or future-containment guarantee.
- Original LiDAR frame/time and visual identity frame/time remain attached.
  The planning-state clock is required; source age must remain in[0,0.35] s.
  No authored human route, future actor truth, language/VLM or new weights enter.

## Bounded reference change

The original obstacle-unaware nominal is rolled out over six0.2 s intervals,
using the original dense integration samples. The original oriented chassis
rectangle and0.05 m physical margin are evaluated against the advisory human
positions over the full1.2 s plus actual source age. Approaching/oncoming motion
is checked again against the obstacle-unaware nominal goal direction. The
actual nominal forecast velocity still supplies the relative-closing test;
temporary avoidance heading does not redefine the route. Crossing and receding
motion cannot activate this parallel preference.

Predicted overlap produces a continuous outward fraction, normalized by the
existing centroid-uncertainty prior and capped by the explicit configurable
maximum (default0.35). It tilts the existing selected-human tangent away from
the current observed person before reference normalization/projection. The
growth prior and this maximum are saved separately in the native report and
per-plan diagnostics. They are neither CRA factor scores nor changed physical
margins. Insufficient data leaves the original reference behavior available.

Every proposed reference and executed horizon still passes the **original**
current-field first-order temporal forecast, exact DTCBF, dense configuration
geometry and per-step analytical CBF. The current human body and all unmatched
occupancy remain. Source/command freshness, candidate revalidation, fragment
continuity, independent physical/containment checks, final arrival and stopping
remain mandatory. No minimum speed is imposed. The extra preference can fail
to find a trajectory and yield; it has no execution authority.

The primary controller and lambda1.25 remain unchanged. This candidate belongs
to the already separate guidance-tracking/human-flux comparison. Mass, contact,
potential and thermal priors are unchanged; it is not new four-factor perception.

## Validation sequence and status

Source tests cover continuous bounded activation, rigid frame invariance,
quiet/receding/crossing rejection, current identity/source binding, unchanged
default/full commands, nominal immutability, original full-horizon revalidation
and the absence of accidental old hard-prediction activation. A source-dataflow
gate requires20 distinct selected current observations,10 quiet/unselected
observations without an advisory context, and no identity/source/body mismatch.
This gate validates current inputs, not predicted human containment.

Run a fresh zero-wheel moving-human component with this dataflow gate and all
old actor/sensor/selection/body/mass-provenance gates. Then require independent
and own60-frame stationary gates before a matched wheel trial. The wheel trial
adds only the new advisory option to the latest no-preparation invocation;
keep the same human route/onset, original1.2 s hard horizon, fixed lambda,
robot limits, priors, margins and solver budgets. Audit the advisory sources
again and retain all old motion gates.

Score actual root motion separately: original hold target<=0.5 s and at least
50% reduction from the accepted original2.383333 s baseline. Also report the
latest derivative-corrected baseline2.50 s, slow speed<0.05 m/s, arrival time,
nearby bend and clearance. No early-preparation policy is allowed. Preserve
failed trials and their original hashes. These were the pre-trial criteria;
the native outcomes are recorded below.

Source validation passed3,167 full unit tests in115.13 s, including20 new
observed-motion/source/controller tests. Whole-repository Ruff, compilation,
both native CLI help paths and diff checks passed. The trusted original
source14.4 numerical replay retains full-command legacy parity with the new
option disabled and the earlier corrected-Jacobian checked result enabled.
Filtered velocity can lag actual starts/stops; the advisory forecast is not
claimed to cover every future actor position.

The first zero-wheel component at4f14a79 **FAILED** the new validator. All old
actor/sensor/selection/body/mass gates passed, with26 distinct advisory sources
and89 quiet/unselected observations. The validator incorrectly treated the
existing `held_current_track_policy_preference` decision as unselected even
though its current source/body values matched. Preserve original report SHA
`c71d7e67f47cd616e6c96942825d47c4aa3bcb70917d3296e333e8c61c7804a7`.
No wheels followed this failure.

The correction audits both existing fresh and held-current selector modes.
Held mode must retain its original verified visual selection and bounded
original lease (at most8 s), current source epoch, selected identity and side;
renewing a held origin is rejected. Current body/velocity/source checks remain.
An offline companion recomputation passes the original26 sources, but it does
not replace the failed native report. A fresh component and stationary gates
remain mandatory. The diagnostic also names inferred centroid/velocity and
configured body/centroid priors separately; the controller method is unchanged.

The correction passed3,170 full unit tests in113.92 s,23 focused tests in5.31 s,
Ruff, compilation and diff checks. On the original failed component data, a
separate descriptive truth audit measured median1.2 s predicted-center error
0.166634 m and maximum0.406306 m (26 sources,208 total offset queries). The
configured advisory growth disk covered those queries; this is reused
development data, not a calibrated bound or permission for hard prediction.
At that point, fresh corrected-source native validation remained pending.

The fresh component atae8d18d **PASSED** all old gates and the corrected new
audit:27 selected current sources,88 quiet/unselected sources and zero audit
mismatches, with zero wheels and unchanged frozen source. The earlier failed
component remains failed. A descriptive fresh-data audit has median1.2 s
center error0.168288 m and maximum0.407628 m; it remains advisory-only evidence.

The first independent stationary launch (`stationary-v2`) then failed before
Isaac import/output acquisition: the live CLI guard referred to the component
demo's `identity_selection_body_envelope` argument, while the live interface
uses `human_body_envelope_radius`. No sensor or wheel acceptance follows.
The guard is corrected to use the actual live argument. Six regression cases
execute the real parser declarations before the guard (valid, missing body,
hard prediction, early redesign, shortened horizon, zero weight). All29 focused
tests pass in4.86 s; the preceding3,170-test full suite covers unchanged
controller/perception code. The complete actual stationary invocation passes
every pre-Isaac CLI guard without creating an output directory.

The passedae8d18d component may be reused only after exact source fingerprints
prove all component/perception/runtime files unchanged, with the sole demo
exception verified as this one CLI expression. A fresh independent stationary
run and own60 gate are still required at the corrected live source. Preserve
the failed launch and every previous report; no wheel trial had run at that point.

## First navigation result and route-classification correction

At9a7d4b3, independent and own60 gates passed. `motion-v3` **PASSED native
navigation/evidence**: final arrival after three route guides in22.483333 s,
0.252364 m minimum physical clearance,0.203984 m nearby walking bend,
0.002440 m stop drift,0.180277 minimum analytical margin and224 live fields.
The new dataflow audit passed28 selected and196 quiet/unselected sources.
Original report SHA:
`3a3ca0efe473f06139f496f6d2db614055f31a1726682ad9c7888c4a715fc721`.

**Smoothing failed**:2.633333 s actual hold, worse than the original2.383333 s
and latest no-advisory2.50 s. Preserve this result; it is not an improvement.
The matched invocation, original1.2 s horizon, MPC configuration, fixed lambda
and preparation exclusions were independently checked. A first positive-advice
request was retained by a read-only export observer; neither the observer nor
offline diagnostics supplied commands. These are single development runs,
not statistical or real-time comparisons.

Five plan records exposed an eligibility defect: advice switched off with
heading0.85--0.93 rad, although the human remained oncoming relative to the
nominal goal (cosine0.980--1.000) and predicted conflict remained. The original
test used the nominal **rollout displacement**, which rotates while Carter
turns back toward its route, to classify human motion. Independent Cauchy
bounds on relative closing remain negative even for any allowed robot velocity
at those states; receding motion cannot explain those five deactivations.
See `motion-v3/route-gating-bound-v2.json`.

The correction uses the configured current nominal goal direction for the
oncoming classification, retaining actual nominal velocity in the closing
test and every other model/physical gate. Three turning-heading regressions
cover this distinction. Source validation passed3,179 full unit tests in114.64 s
and32 focused tests in4.93 s. This is not a new physical uncertainty policy.
All body/centroid/velocity inputs, margins, weights, priors and source clocks
remain unchanged. The component's source reuse requires exact fingerprints
and AST proof that changes touch only unused MPC reference/preference methods
and the already tested live CLI expression.

A trusted same-input replay of the late14.2 s native request reproduces the
original full command sequence and still returns the same checked low-speed
command after the classification correction. Thus the defect is real but is
not established as the sole yield cause. A fresh independent stationary gate
and matched wheel run are still required to test the corrected earlier path.
No smoother-motion claim or hold release follows from source tests alone.

## Corrected route-classification native result

At frozen runtime2e905fdbedf9dfdff71cc6631ae3a439d41c4121, the independent
60-frame stationary run passed in229.29 wall seconds, with zero wheel commands.
The wheel run's own60-frame gate also passed. Reuse of the earlier moving-human
component was proven by exact source fingerprints, the one-expression live CLI
repair, and whole-module AST equality after removing only the unused MPC
`plan` and advisory `preference` methods. No tracked source changed during
either native process, and no separate offline solver ran during motion.

The corrected `motion-v4` **FAILED native acceptance**. Its original report SHA is
`13a6ff2ce7acd46ca43009cde7f9a8502ceb0aff59c80408b898f82226cebb24`.

| Measurement | Result |
| --- | --- |
| Final arrival | Failed;2/3 route guides progressed, final goal not reached |
| Stop reason | `repeated_inference_failure` after18.716667 navigation simulation seconds |
| Actual nearby translation hold |2.833333 s; unchanged0.5 s target failed |
| Nearby walking bend |0.176597 m; unchanged0.20 m curve gate failed |
| Minimum physical clearance / stop drift |0.130204 m /0.000949 m |
| Minimum analytical CBF margin |0.008801 |
| Live fields / translating commits |179 /145 |
| Current advisory source audit | Passed;26 selected,153 quiet/unselected sources |
| Supported moving kinetic rows |24; current motion with adult-mass priors |
| Fresh complete four-factor updates |0 |

All20 recorded advisory plan diagnostics remained active with positive outward
preference; the turning-induced oncoming deactivation was repaired. This
correctness result did **not** produce smoother navigation. The2.833333 s
hold is a descriptive failed-run measurement, not an eligible successful
smoothness comparison. Single development trajectories do not establish the
cause of their differences.

After the pass, current positive LiDAR support for human track0 disappeared.
The original identity/body-continuity gate rejected subsequent0.3--0.5 s source
gaps, and the existing repeated-inference guard stopped the robot. These
rejections were retained; no identity/source clock was renewed and no body
geometry was removed to complete the route. An eventual track-retirement or
reassociation method would require justified uncertainty/association rules and
new stationary plus live validation. This trial supplies no permission to
bypass the current guard.

Independent root-truth evaluation found only0.000466 m maximum lateral motion
before actual human walking at source11.30 s. The first2 cm bend occurred at
source13.05 s. At the largest scored nearby bend, source15.433333 s, the robot
was at(28.622398,8.028129) and the human at(29.854471,8.279545),1.257463 m apart
by centers. Center separation is not body clearance. Authored actor truth
remained evaluation-only and never entered control.

Local artifacts are retained under`outputs/observed_motion_reference/motion-v4`:
the immutable report, full native video including the yield/failure stop,
`comparison-v4.json/png`, `trajectory-timing-v4.json/png`, an untouched native
`native-nearby-curve.png` and its source/frame metadata, and
`final-audit-v1.json`. No success-only navigation trim was created. The launcher
returned0 but the saved report is failed and its explicit gate exception is
recorded; process exit status does not override native acceptance.

Source validation remains3,179 unit tests, including32 focused cases, plus
whole-repository Ruff and compilation. All older report hashes remain unchanged.
Keep this default-off experiment draft/unmerged. The earlier validated yielding
baseline remains the preferred available demo; this candidate is not a replacement.

No primary/main/PI-demo/long-service/PR117 hold is released. A successful selected
trial would not establish real time, general human prediction, causal CRA
superiority, exact Safe-SAGE equivalence or elimination of all safety yields.
