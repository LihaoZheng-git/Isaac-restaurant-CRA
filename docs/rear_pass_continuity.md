# Rear-pass policy continuity experiment

This is a separately enabled continuation of `docs/rear_pass_diagnosis.md`.
All four original failed native rear reports remain unchanged. The user has
authorized another bounded attempt; the accepted stop/go fixture remains the
fallback. Native acceptance is pending at this source freeze.

The previous fit was used only for classification. At source 17.3 s the same
human was classified as crossing from fitted motion while the rear target was
disabled by a noisy raw velocity. At 17.6 s the bounded search also found no
preferred translating-first reference. This experiment addresses those two
specific interruptions without changing physical geometry, risk or safety
constraints.

## Consistent policy motion

`--rear-consistent-motion-fit` requires the existing explicit encounter fit.
An immutable `EncounterPolicyMotion` retains its original source times and
observed centroids, raw current velocity, fitted velocity and current identity/
support counts. Construction independently checks the fit, residual, span,
gaps, speed and non-opposing current velocity. Field construction and planning
require the record to match the actual current track, source, centroid, point
count and observation count. Old, sparse, ambiguous or mismatched records
cannot become current rear policy.

Classification and rear-target selection use the **same accepted inferred
window-average velocity** when this new option is enabled. Rejected fits leave
both paths using original current tracking. The original raw transverse check
remains the legacy default. The new option applies the same transverse test to
the fitted policy vector; it is not a current instantaneous measurement or a
motion bound. Its possible transition delay remains explicit.

Raw tracking, CRA kinetic inputs, body geometry/coverage and initial passing
selection remain unchanged. Diagnostics keep `observed_velocity_grid_xy` raw,
record `policy_velocity_grid_xy` separately and retain original fit epochs.
The source-window estimator remains part of the staged perception transaction.
No actor truth, authored direction or desired score enters the fit.

## Productive checked turn-first fallback

`--rear-turn-first-fallback` requires explicit rear mode and a positive preferred
first-translation quality. The existing search pool and 576-candidate bound
remain unchanged. It prefers complete references meeting the configured
first-interval translation quality. Only when none pass does it consider a
turn-first/sub-quality reference, which must still have positive selection
utility, productive complete-horizon translation, exact DTCBF/dense geometry,
future analytical acceptance and an unchanged first command through CBF.

This is a reference-quality preference, not permission to execute an unsafe
rotation, a velocity floor or a relaxation of any safety constraint. Original
command/field age, current-pose revalidation, per-physics-step CBF and stopping
checks remain mandatory. The default is disabled. The primary objective,
obstacle-unaware waypoint nominal, physical lambda 1.25, human 0.40+0.25 m
body prior and factor priors remain unchanged.

The existing explicit planner replay export additionally preserves the first
unavailable rear request and first turn-first fallback request. These are two
bounded trusted-local numeric snapshots; their export time counts against
worker/result age, and their original inputs never enter control.

## Validation protocol

Component tests cover source/identity/geometry mismatches, mirrored fitted
direction, current raw tracking preservation, exact checked turn-first motion
when forward-first acceptance fails, unchanged preferred selection and failed
first-CBF rejection. Full unit/Ruff/compile/CLI checks precede native motion.
At the implementation freeze, all 2,690 unit tests passed in 89.80 s with a
repository-local temporary directory; Ruff, compilation and CLI checks passed.
One trusted original 16.7 s native request (input SHA256
`7f2d5ce03af2adffdb5e5c582100995ac26f0275379d74ee0cc0dd471053476c`)
retains its exact preferred command sequence with the new options and original
source-position fit; the saved physical field hash and original tracking are
unchanged. This is one numeric dataflow check, not counterfactual navigation.

Require a new independent zero-wheel stationary gate and each motion trial's
own 60-frame gate. Start with the unchanged left-to-right 8 s development
fixture and unchanged 0.20 m walking-window arc, actual arrival, stopping,
clearance, containment and source checks. A first-direction failure does not
authorize mirrored wheels. Keep main/long-service/PR117 holds, prior-assisted
scope, incomplete fresh four-factor perception, oracle ego/extrinsic references,
RGB clock approximation and renderer-history warning explicit. This does not
establish real time, exact paper equivalence or global unicycle convergence.
