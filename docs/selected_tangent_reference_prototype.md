# Offline selected-tangent reference prototype

This branch is an **unmerged prototype**, not an integrated controller fix.
The waypoint nominal, CRA factors, fields, actuator limits, exact MPC/DTCBF/
dense geometry and analytical CBF checks remain unchanged. A separate positive
tracking objective may prefer the semantic field's component tangent to the
actual planar safety gradient. The runtime permits this preference only with
the existing current positive passing-selection track IDs; original source
clocks and identity/factor provenance remain unchanged.

The generator limits desired travel speed to the original nominal speed,
projects the actually requested `(v, omega)` through the existing lookahead
control matrix, and still requires full exact horizon acceptance before using
a reference. No primary caller enables the new fraction by default. A
per-request configuration copy disables the prototype when selection is absent;
it does not mutate the shared planner. Geometry, missing-factor handling and
passing-selection gates are unchanged.

Seventeen new component tests passed, including steering before an inactive
halfspace binds, exact full-horizon checks, zero/parallel-gradient behavior,
invalid/current identity and epoch handling, request-local gating and unchanged
source/nominal inputs. The first implementation passed all 2,344 unit tests in
85.79 s. The subsequent angular-coupled generator passed 67 focused tests and
Ruff; its full regression/native validation remains pending.

Trusted SHA-checked original-state replays in
`outputs/paper_flux/selected-tangent-coupled-v1` did **not** improve either late
crossing snapshot at the legacy 30 x .2 s horizon. Both zero/positive fractions
retained the same failure or rotation-only recovery. Keep this negative result.
The separate horizon diagnostic at 6/10/15 steps preserved exact acceptance and
original epochs, and found forward commands at six steps in both snapshots;
it is not a native bypass or real-time result. Shorter horizons change prediction
coverage and must be reported as a separate matched ablation. This prototype
does not justify changing main or promising a curved human bypass.

## Encounter-reference follow-up

The separate `codex/cra-encounter-guidance-reference` development branch brings
this option together with the tested fresh class-only cue and the explicit N6
horizon. This is a bounded native candidate, **not accepted navigation**. The
75f45ce cue-only trial selected the trailing side immediately at23.0 s but
failed the independent .02 m clearance gate at.019088 m; earlier class eligibility
alone did not produce proactive motion. Before selection, accepted planar
references were nearly straight. Later references became infeasible near the
intermediate waypoint. Preserve these observations and the original failures.

The next isolated objective test uses fraction1, .15 m reference fade (legacy
.5 m retained), N6 and the same8.4 m actor-only trigger/.5 m/s human, observed
occupancy, priors, flux/lambda, actuator limits and safety checks. Positive
fraction is still gated by the current published passing selection. It cannot
make an unsafe current configuration or full horizon acceptable. The original
nominal goal and waypoint acceptance are unchanged; .15 m changes only auxiliary
reference fading, never obstacle geometry or clearance requirements.

The first planner request carrying a current positive passing selection is now
saved separately as `first-selected.pickle` with its SHA/source epochs. Later
requests cannot overwrite it; the five-second bucket snapshots remain. These
trusted local diagnostic files are never loaded by control.

All2,365 collected unit tests passed in83.23 s, including the prototype and new
snapshot test. An additional focused19-test run also covers N6 near a .2 m
intermediate waypoint in both tangent directions. Ruff `robot tests scripts`
passed. Own stationary/native validation is pending; no physical evidence,
four-factor readiness, geometry uncertainty or paper-equivalence claim follows.

### Pre-motion orientation rejection and selected-obstacle refinement

The08ea845 native candidate was interrupted at stationary frame27, before its
gate completed or navigation began. A SHA-verified audit of the original cue-only
trial's23.8 s field found that projecting the combined semantic vector onto the
combined planar h-gradient tangent gave **-.083452 along route-left**, despite
the selected left/trailing policy. That state was already unsafe; it is not a
recoverable-trajectory proof. More generally, the combined field projection
need not preserve one selected obstacle's boundary-tangent orientation. Keep
this rejected candidate and do not describe it as a completed stationary or
motion test.

`--guidance-tangent-basis selected_observed_obstacle` is a separate explicit
refinement, preserving `planar_gradient` as the older prototype. A current
published passing selection supplies its original side/source epoch and its
current positive LiDAR centroid. In each reference rollout, let n point from
the predicted robot position to that observed centroid. The preferred tangent
is (-n_y,n_x) for left and its negative for right. It is added softly to the
unchanged waypoint-directed reference, bounded by the configured fraction,
field activation, goal fade and original reference speed. The observed centroid
is not actor truth, a body center measurement or added occupancy. It remains
fixed over this short auxiliary rollout; no future actor trajectory is supplied.

This is **selected motion-policy/boundary-tangent tracking**, not a claim that
the robot follows the entire combined harmonic vector or that the original
Safe-SAGE objective has this extra term. Primary control remains unchanged.
The reference's side and original selection epoch are recorded in its current
field, require current positive identities and <=8 s original selection age,
and are never renewed by the planner. No selection/side metadata disables the
tangent preference on a request-local copy. Ambiguous multi-target inputs reject.
This basis disables the legacy combined-field beam reference search so the
two preference definitions cannot be silently mixed. Exact reference/horizon
and command acceptance, all physical evidence, occupancy and source clocks
remain unchanged. The .02 m independent clearance guard is unchanged.

New component tests require the selected direction even when the combined
field projection points oppositely, check both sides at N6/.2 m waypoint
distance, retain exact acceptance and unchanged nominal, and reject missing,
ambiguous, invalid-side/future/expired source inputs. Full regression/native
results for this refinement will be recorded separately.

Before native retry: all2,383 unit tests passed in81.39 s. After adding the
original side/selection epoch to diagnostic metadata,38 focused tests passed
in5.20 s; Ruff `robot tests scripts` and diff checks passed. Native acceptance
remains independent of these component results.
