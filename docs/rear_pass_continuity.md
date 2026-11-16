# Rear-pass policy continuity experiment

This is a separately enabled continuation of `docs/rear_pass_diagnosis.md`.
All earlier failed native rear reports remain unchanged. The user authorized
bounded improvements while retaining the accepted stop/go fallback. The final
results below include a successful left-to-right native trial and a successful
mirrored trial with additional, explicitly labeled feasibility/identity options.
These are separate controller-ablation configurations, not primary comparisons.

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

## Native feasibility-led mirrored result at 1ebdfd4

Its independent stationary gate passed 60 zero-wheel frames in 164.55 wall
seconds, followed by its own passed 60-frame motion gate. The original overall
report remains **FAILED**, SHA256
`b3e5a6624e8cf1db02e41991eae26f88662080ed12a85327f21b796028e13ccd`.
Rear execution now passed, but the walking arc was 0.183845 m < 0.20 m. Actual
arrival was 3/3 in 21.883333 simulation seconds, braking-inclusive clearance
0.690817 m, stop drift 0.002305 m and minimum analytical margin 0.307344.
All 217 containment rows passed (max center error 0.149520 m).

Seventeen accepted rear plans included two turn-first fallbacks; sixteen rear
sources occurred during walking, with 110 joined translating rows/1.833333 s.
Independent audits verified 20 turn-first rows/0.333333 s, 34 current policy
records, 258 original-source fits and all original source/vector/physical
checks separately. Six worker sources and three accepted plans used the
explicit lead-preference override. The actual first selected source 17.1 s
replays `(0.1, -0.8)` and its full exact-checked command sequence from the
unchanged trusted snapshot (SHA256
`41be301746fb61e38e4f93475d00102f08e79c5ed7a213b6b51af0f76f1ff056`).
This is improvement from zero rear plans, not full curved acceptance. The 78
visual instances/14,205 assessments still had zero fresh complete four-factor
assessments. Original priors and sensor/clock/rendering limits remain.

## Reuse of the previously approved identity-selection ablation

The remaining mirrored delay has a directly observed eligibility gap. Sources
16.9 and 17.0 s already qualified geometrically/motion-wise but reported
`unverified_visual_person_target`. The current body adapter had verified 1,048/
1,004 LiDAR returns and retained the original accepted image at 16.883333 s,
with image ages 0.1/0.2 s. Fresh selection occurred only at source 17.1 s.

The next bounded recipe reuses the **existing user-approved**
`--human-passing-identity-prior-max-age .5` option, documented in
`docs/identity_prior_selection.md` (2026-10-20 decision). No runtime code changes
are needed. Only current successfully verified body-envelope continuity may
admit that short person-identity prior; source image/frame, selection time and
current LiDAR epoch remain separate. The eight-second policy clock and physical
factor, field and command clocks never renew through this retention. This is
not a fresh visual selection or an extension of the longer geometry identity
lease. Primary same-frame selection remains unchanged, and this option remains
excluded from primary comparisons.

Require a new stationary gate and own 60-frame gate for this labeled combined
experiment. Keep 1ebdfd4 runtime logic, the feasibility-led option, 1.4 m reported
lead preference, body/factor priors, fixed physical lambda and all exact/physical/
arrival/arc/stopping gates unchanged. Native outcome is pending at this recipe
freeze; neither earlier mirrored failure is relabeled.

## Native short-identity-prior result at 2c4754a

The final recipe changes only `--human-passing-identity-prior-max-age` from zero
to 0.5 relative to the failed 1ebdfd4 mirrored recipe. Commit 2c4754a changes only
guidance/results documentation relative to that runtime source. The exact
invocation audit preserves all other arguments except the unique output path.
No packages, model weights, geometry, physical factors, lambda, lead value,
acceptance thresholds or safety constraints changed for this retry.

The independent stationary gate passed 60 zero-wheel frames in 164.12 wall
seconds; the motion run passed its own 60-frame gate. Its original overall
report **PASSED**, with no launcher traceback. Native report SHA256:
`41a238d2a2685310ab79f6461ac24bd0b59fe58380577cdc3caf338c0cfa0a72`.

| Accepted development configuration | Walking-window trailing arc (required 0.20 m) | Actual arrival | Braking-inclusive clearance | Stop drift |
| --- | --- | --- | --- | --- |
| Left to right, 28e6dec: consistent motion + checked turn-first | 0.292412 m | 3/3 in 21.40 simulation s | 0.696047 m | 2.435 mm |
| Right to left, 2c4754a: additionally feasibility-led start + short identity-selection PRIOR | 0.223436 m | 3/3 in 21.833333 simulation s | 0.682125 m | 2.293 mm |

The mirrored trial recorded 216 live fields, 214 while translating. All 216
independent body-containment rows passed, with maximum center error 0.149661 m
inside the unchanged 0.25 m centroid prior. Minimum analytical margin was
0.258504; the robot crossed the route plane 1.407785 m behind the human. Current
human kinetic/motion, source/freshness, encounter, scene-preservation, actual
arrival and stopping gates all passed separately.

The new selection occurred at LiDAR source 16.9 s, versus 17.1 s in the failed
fresh-only mirrored recipe. Its original image was frame 148 at 16.883333 s;
the selection camera clock was 16.983333 s, so the identity age was **0.1 s**.
There were 1,048 current positive LiDAR returns. The selection was explicitly
**PRIOR**, with no fresh person ID at that source. Thirty-seven held-policy rows
preserved this exact original record and the unchanged eight-second selection
clock. Later camera observations never renewed the original selection. The
independent identity audit verified current body support, unchanged 0.40+0.25 m
extent, original photo/frame/selection epochs and the separate prior label.

Twenty accepted rear plans included two checked turn-first fallbacks; eighteen
rear sources occurred during transverse walking. The original request-clock
join verified 121 translating command rows/2.016667 s. A conservative independent
reconstruction counted 120 rows/2.0 s: its source-derived start was
8.200000000000001 s versus the native accumulated-timeline tick
8.199999999999987 s. Replaying the exact native tick reproduces the **entire
original execution report**; retain both counts and the separate boundary
companion, without altering the original report or any criterion. Independent
turn-first execution verified ten walking-window rows/0.166667 s.

Independent audits reproduced 257 original-source fits and 37 maneuver-policy
records, keeping raw tracking/CRA kinetic inputs separate. Seven worker sources
and three accepted plans used feasibility-led search below the still-reported
1.4 m lead preference. The first accepted source 16.9 s had observed lead
1.372970 m and command `(0.1, -0.6)`. Trusted numeric replay reproduced its full
command sequence, exact/dense checks and unchanged first CBF result (input
SHA256 `fcfa7a99a197e6714b49d2cdbccef2737bcacfb8c8c09102cb68994e4e3a7d80`).
The first turn-first snapshot at source 17.4 s similarly reproduced `(0, -0.6)`
and its complete productive horizon (input SHA256
`35197073c97ae54e9337f49c3d70b392cdc99d47070d0af20b07e0443e591c1c`).

Artifacts are under `outputs/paper_flux/native-rear-priorselection-right8-v7/`:
`report.json`, `native-summary.json`, the five independent audits, both numeric
replays, `execution-clock-boundary-companion.json`, `human-navigation.mp4` and
`crossing-current.png`. The independent stationary run is in
`native-rear-priorselection-stationary-v7/`. The per-direction figure is
`outputs/paper_flux/native-rear-directions-v7.png`; its caption explicitly
distinguishes the two configurations. All seven previous native report hashes,
including six failures, were rechecked unchanged.

## Final scope and remaining limits

The runtime source passed **2,696 unit tests** in 89.85 s, plus 52 focused
identity/passing tests in 5.29 s. Repository Ruff, compilation, CLI and diff
checks passed. The final native retry uses that same runtime source; subsequent
changes only record results and durable guidance.

The mirrored run had 75 accepted visual instances/14,205 object assessments and
**zero fresh complete four-factor assessments**. Configured human mass
70 [45,100] kg, blunt-human contact and unavailable-factor priors remain
explicit, as do the 0.65 m body-envelope prior, oracle ego/extrinsic references,
RGB read-time acquisition approximation and unresolved renderer-history warning.
The 0.5 s identity option supplies no physical factor and remains excluded from
primary comparisons. Physical lambda remains 1.25; human flux -1.7 remains a
separate declared control ablation.

This establishes two successful bounded native rear-pass development trials,
with different explicit options. It is not repeated/held-out robustness,
real-time control, a causal CRA advantage, calibrated human mass/contact/heat,
exact paper equivalence or global convergence. The mirrored trial took about
18.22 wall minutes at the explicit 0.025 simulation pacing. Keep stop/go as the
fallback, original failures intact, and main/long-service/PR117 holds unchanged.
