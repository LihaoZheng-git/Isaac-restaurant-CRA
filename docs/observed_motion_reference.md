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
is checked again against the nominal travel direction. Crossing and receding
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
failed trials and their original hashes; native outcome is pending.

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
Fresh corrected-source native validation remains pending.

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
the failed launch and every previous report; no wheel trial has run yet.

No primary/main/PI-demo/long-service/PR117 hold is released. A successful selected
trial would not establish real time, general human prediction, causal CRA
superiority, exact Safe-SAGE equivalence or elimination of all safety yields.
