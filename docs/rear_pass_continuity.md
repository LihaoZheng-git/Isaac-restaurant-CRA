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

## Native continuity results at 28e6dec

The independent stationary gate passed 60 frames/zero robot wheel commands in
169.82 wall seconds. Both direction trials passed their own 60-frame gates.

| Direction, 8 s onset | Original overall | Walking arc (required 0.20 m) | Actual goal | Braking-inclusive clearance | Stop drift |
| --- | --- | --- | --- | --- | --- |
| Left to right | PASS | 0.292412 m | 3/3, 21.40 simulation s | 0.696047 m | 2.435 mm |
| Right to left, unchanged lead gate | FAIL | 0.014677 m | 3/3, 21.216667 simulation s | 0.749022 m | 2.323 mm |

Left-to-right recorded 213 live fields/211 translating fields and 213/213 body
containment rows (max center error 0.161136 m). There were 21 accepted rear
plans: 19 preferred and two turn-first fallbacks. Twenty rear-plan sources
occurred during walking, with 124 joined translating rows/2.066667 s. An
independent original-request-clock audit additionally verified 21 actual
turn-first command rows/0.35 s during walking. Eighty policy records and 242
source-window fits independently matched original source times, classification,
worker/planner vectors and unchanged raw physical tracking. The first actual
fallback snapshot at source 17.3 s reproduces exactly `(0, 0.8)`, its full checked
command sequence and 0.079659 m horizon translation, without changing the
original field or first CBF result. Native report SHA256:
`f690764698f908adcd72265371311fd241b5bc1d9f3caf3cae80f90d39a68f8c`.

The mirrored failure had zero rear plans or rear execution. Its first passing
selection occurred at source 17.1 s; all subsequent crossing contexts were
blocked by the 1.4 m lead preference. At the original first planning state,
observed lead was 1.345464 m. Earlier source/commit lead diagnostics were also
below 1.4 m. Classification and current visual selection had separate onset
delays; an observed surface centroid is not the actor's center. Keep its failed
curve and encounter flags despite successful arrival, human-motion and physical
checks. All 210 containment rows passed (max center error 0.149236 m); 252 fits
and 33 policy records independently reproduced their original inputs. Report
SHA256 `c037bedc16b008e711d1471f19c6982c6d81390f47dd7f4ce5302fa2c91074e7`.

These trials recorded 67/83 visual instances and 14,340/13,885 object assessments,
with **zero fresh complete four-factor assessments** in each. The same actor,
factor and body priors, oracle ego/extrinsic reference, RGB read-time clock
approximation and renderer-history warning remain. Exact invocation comparison
confirmed only the two new options relative to the previous fit trial, and
only direction/output differences between these two trials. This is a bounded
development comparison, not repeated/held-out or causal CRA validation.

## Explicit feasibility-led start retry

The next option, `--rear-feasibility-start`, is enabled only with the consistent
motion and checked turn-first experiment. The 1.4 m value remains in the config
and diagnostics as a **non-safety maneuver preference**. For a qualified crossing
currently ahead of the robot, the new option may attempt a full checked rear
reference below that preference. It never permits a new short-lead initiation
for an obstacle at/behind the robot. Motion, identity, source clocks, goal and
release gates remain unchanged.

An admitted **search** does not admit a command: the entire reference still
needs every original exact DTCBF/dense geometry, future analytical, first-CBF,
positive utility/translation and execution-freshness check. No checked candidate
retains the existing primary-MPC fallback. A normal-guidance field is not an
admitted zero-lambda wait field, so this option does not invent intentional
yielding when its reference fails. The original lead-gated and explicit checked
wait/go variants remain available, with their original results.

On the trusted original mirrored 17.1 s numeric snapshot (SHA256
`ae152c12e695249290ad4f57d87e0c883065d1b97efe7537b3b7c7b7982be0a6`),
the new option finds a complete checked `(0.1, -0.8)` first command with the
**same** normal-guidance physical field, raw tracking and source clocks. This
is an offline eligibility counterfactual, not a new wheel result. It motivates
checking actual feasibility instead of reducing geometric uncertainty or tuning
the lead value. A branch audit found no short-lead context in the accepted
left-to-right trace; this does not constitute a new left native run with the
additional option enabled.

The retry source passed 2,696 unit tests in 89.85 s, Ruff, compilation and CLI
checks. Tests retain default lead gating, reject new behind-obstacle starts,
require complete exact admission and reject failed analytical commands. Require
a new independent stationary gate and its own 60-frame motion gate before the
mirrored retry. Native retry acceptance is pending at this freeze. No original
failed flags, geometry, priors, lambda or 0.20 m criterion are changed.
