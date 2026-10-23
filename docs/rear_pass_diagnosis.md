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
