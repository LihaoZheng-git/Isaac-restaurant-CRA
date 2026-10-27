# Separate parallel-walking smoothing experiment

**Current result: reference-only smoothing is NOT accepted.** The original
baseline remains valid with its2.383333 s yield. Two turning-reference variants
passed navigation but retained longer holds; no safety check was weakened.
The user subsequently approved a separate short-horizon predictive-human-geometry
ablation, using current LiDAR motion and explicitly labeled geometry/motion
priors. That is new work; it is not part of the primary controller or an
already validated prediction model.

The accepted 2 m guide-lookahead trial reached its final goal with a22.3 cm
nearby curve, but held zero velocity for2.383333 s while the person was close.
Its command/planner records contain27 consecutive fresh
`unsafe_configuration_boundary` rejections from navigation elapsed7.25 s;
the next accepted result was at9.65 s. The rejected results' source ages were
about.016667 s. The evidence identifies the configuration-geometry gate as the
principal cause, rather than an intentionally scheduled observation pause or
a long optimizer-result delay. It does not prove that stopping was unavoidable.

Baseline: `outputs/restaurant_parallel_route_guides/motion-lookahead2-v2`,
source3fb7e2e, report SHA256
`4ce3a525b28a12cbb59d9a81a6f0f764165fc09c3d6a57a9ef7cc32d746d9917`.
Its original report, video and prior failures remain unchanged.

## First bounded change

Use the existing explicit `--guidance-activation-h` control to change2 to3,
leaving every other invocation argument unchanged apart from the output path.
These are **field-value units, not metres**. The larger threshold activates the
checked guidance reference over a wider range of field values and increases its
influence earlier after the original current-human selection gates pass. It
does not select an unverified human, extend an original selection lifetime,
manufacture future occupancy or change CRA/phi. There is no new planner, model,
dependency or altered controller default.

Keep the convex selected-obstacle reference, beta100, 2 m pose-only route-guide
lookahead, final-goal requirement, human route/speed/3 s onset, fixed lambda1.25,
human flux-1.7 ablation, body.40+.25 m prior and factor priors. All source,
configuration/geometry, exact DTCBF, analytical filtering, independent physical
clearance/containment, stopping and nearby20 cm curve gates remain mandatory.
This is separate from the primary comparison and perpendicular checked wait/go.
Do not change the body envelope, delete unresolved fragments, loosen association
or freshness, increase robot limits, or tune lambda to obtain a smoother result.

Source tests and fresh independent plus own60-frame zero-wheel gates precede
motion. The already accepted parked actor/perception/selector component can be
reused only with exact source fingerprints proving it unchanged. The first
native candidate is3; any later parameter retry must retain its failed result
and be reported separately. No long mission or main/PR117 hold is released.

## Prospective evaluation

The new offline scorer uses independent root positions and original source
epochs. It measures actual planar speed during the person's walking phase
(speed>=.3 m/s) within2 m, matching the existing encounter window. Rotation or
nonzero commanded velocity cannot hide an actual translation hold. Report both
the longest interval below.02 m/s and the longest slow interval below.05 m/s.
The baseline gives2.383333/2.566667 s respectively. Missing/short/no-encounter,
unordered, nonfinite or temporally gapped records cannot establish smoothness.

The engineering target, fixed before the new wheels, is a maximum.5 s actual
translation hold and at least50% reduction from the baseline, with all original
native acceptance gates still passing. Report slow creeping and navigation time
alongside it; a zero command-label count alone is not success. This smoothness
target is not a new safety theorem or a replacement for the original acceptance.

The comparison checks that only activation/output arguments changed and that
the runtime source stayed unchanged. Reproduce after both reports exist:

```powershell
python scripts/evaluation/compare_parallel_smoothing.py `
  --baseline outputs/restaurant_parallel_route_guides/motion-lookahead2-v2 `
  --candidate outputs/restaurant_parallel_smoothing/motion-h3-v1 `
  --output outputs/restaurant_parallel_smoothing/comparison-h3-v1.json
```

Native smoothing acceptance is pending. Retain the labeled factor/body priors,
zero fresh complete four-factor baseline limitation, paced simulation/oracle
pose/acquisition-clock limits, unresolved fragments and lack of repeatability,
real-time, causal CRA or exact-paper claims from the behavior-validation report.

Before native freeze, source validation passed2,893 units in100.93 s, 85 focused
tests in7.91 s, repository Ruff, compilation, CLI help and whitespace checks.
A trusted locally generated first-selected native snapshot retained exact-safe
plan success at both activation2 and3, with unchanged nominal/goal/field bytes
and original source epoch. The3 reference requests more outward turning within
unchanged limits; this is an offline hypothesis, not a navigation result.

## Activation3 result: preserve failure

At6f87de6, independent and own60-frame zero-wheel gates passed. The matched
`motion-h3-v1` trial reached its goal/3 guides in21.633333 simulation seconds,
with.224664 m nearby curve,.374017 m physical clearance,.002406 m braking drift,
and.230701 minimum accepted analytical margin. Selection and all active body
containment checks passed. It nevertheless FAILED native acceptance: only4
supported moving-kinetic rows were recorded against the unchanged minimum5.
Its actual translation hold was2.55 s (baseline2.383333 s), and its slow interval
was2.566667 s. Both native and smoothing failures remain explicit; goal arrival
alone is not a pass. Original report SHA256:
`d942427d925fa40444de43030264cb61ec9ab7cca12480f4390b725f91bcb2e5`.

The activation-only comparison passed its matched-invocation/runtime-source
audit but FAILED its prospective smoothness target. Preserve this report and
the original accepted baseline unchanged. No further activation increase is
justified by this result alone.

The next already approved wider-reference step returns activation to the
accepted baseline2. It will separately add a small outward current-centroid
normal component to the selected travel tangent, before the existing blending,
normalization, unicycle conversion and exact checks. This is a configurable
travel preference, not new obstacle geometry, a larger safety margin, or
evidence about future free space. Default zero must retain the old behavior;
current selection/identity/source failure must still disable it. Compare it
against the accepted activation2 baseline, keeping all other parameters fixed.
New source tests and independent+own60 gates are required before its wheels.

## Wider current-observed travel reference

The separate default-zero `--guidance-outward-reference-fraction` is limited to
the explicit convex parallel route-guide ablation. For a current observed
robot-to-human unit normal `n` and the selected oriented tangent `t`, the travel
direction becomes `normalize(t - k*n)`, before the existing convex goal blend.
The first bounded candidate uses`k=.35`, activation2 and the accepted2 m guide
lookahead. This is dimensionless reference shaping, not a changed physical
margin, density, hazard score, body envelope or future obstacle position.

All reference rollouts, their converted unicycle commands and actual execution
retain the original exact/dense/analytical checks. Default0 skips the new
operation. Runtime still disables both tangent and outward preference locally
when the current selection is absent, without mutating the shared planner or
renewing evidence/identity/field clocks. Diagnostics distinguish the configured
fraction from its effective request value. Existing primary/crossing defaults
remain unchanged.

The first snapshot hypotheses showed a tradeoff: larger outward fractions slow
initial forward motion while the forward-only robot turns. The.35 candidate is
a bounded compromise (about.052 m/s first command atactivation3 in the trusted
offline selection), rather than proof of improvement. The actual next trial
restoresactivation2 to isolate outward shaping against the accepted baseline.
There is no oracle actor trajectory or extra occupancy in the preference.

Use the same prospective actual-hold/slow-creeping metrics and native acceptance:

```powershell
python scripts/evaluation/compare_parallel_smoothing.py `
  --variation outward `
  --baseline outputs/restaurant_parallel_route_guides/motion-lookahead2-v2 `
  --candidate outputs/restaurant_parallel_smoothing/motion-outward035-v1 `
  --output outputs/restaurant_parallel_smoothing/comparison-outward035-v1.json
```

The invocation audit allows only the added outward fraction/output path. Its
declared source-scope audit permits the tested MPC reference and demo CLI/report
changes, with every actor/perception/selector module still fingerprinted
unchanged for component reuse. This filename audit is not semantic equivalence
or a safety proof; code review and tests remain necessary. Native acceptance
is pending. Preserve activation3's failure and all earlier accepted/failed data.

Wider-reference source freeze passed2,922 units in100.77 s,89 focused tests in
6.66 s, Ruff, compilation, both CLI help paths and whitespace checks. The
trusted baseline first-selected snapshot reproduces its complete command
sequence atdefault0 (1e-10 tolerance); the.35 reference also passes exact
planning with unchanged field bytes/epochs. These are source/offline checks,
not native smoothing or perception acceptance.

## Outward .35 result: preserve failure

Atd66b22c, independent and own60-frame stationary gates passed. The native
`motion-outward035-v1` trial FAILED `progress_timeout` at17.95 navigation
seconds, reaching only1/3 guides. Nearby walking excursion was.106335 m,
below the unchanged.20 m curve requirement. Human motion/kinetic (6 supported
moving rows), current selection and all179 body-containment rows passed, but
those component successes do not establish successful navigation.

Minimum independent physical clearance was.127172 m, stopping drift.00000983 m
and minimum accepted analytical margin.051063. The actual near-human hold
increased to2.95 s (slow interval2.966667 s). The matched-invocation/source-scope
audit passed; both native and smoothing acceptance failed. Report SHA256:
`edcdd489498bd0f3df199bf357617bc46f254f6e3006bf20f4ce3ce17ec8373d`.

Selection records show transient geometric ambiguity atsource13.5 s and no
fresh visual qualification at13.6--13.9, despite a currently supported human
envelope with its original image at13.183333 s. Guidance correctly disabled
the extra preference rather than treating retained identity as fresh. The
next separate configuration may use the **already approved <=.5 s identity
selection PRIOR**, with current envelope/competitor checks and original image,
selection and eight-second policy clocks. This is not a change to the primary
same-frame selector, physical-factor retention, body geometry or acceptance
thresholds. A provenance-aware independent parallel audit must explicitly
distinguish this mode and a new stationary component must precede its wheels.
Retain both failed smoothing trials and the accepted baseline unchanged.

## Separate short-identity-prior retry

The next bounded candidate keeps activation2/outward.35/guide2 and changes only
`--human-passing-identity-prior-max-age 0` to`.5`. This reuses the existing,
previously approved runtime eligibility policy; the passing selector, body
continuity, ambiguity, physical-factor and source-clock algorithms are unchanged.
In particular, a competing geometric candidate still releases the preference.
The hypothesis is that a new, currently supported selection can recover after
that ambiguity clears, before the original visual cue exceeds.5 s. It is not
permission to retain that cue indefinitely or renew any policy/evidence clock.

The independent parallel selector audit now requires an explicit matching
age configuration. Default0 retains the fresh-only criterion. The separate
positive-age mode validates the original typed identity record, actual original
selection-source proof, current positive body-envelope support, unchanged held
record and eight-second clock, plus all existing motion/away-side/field gates.
Reports count prior selections and held rows separately from fresh selections.
An enabled option with no actual prior selection is not natural prior-path
validation. Primary comparisons still reject the enabled variant.

A zero-wheel moving-human component explicitly uses the same.40+.25 m body
prior and`.5` identity option; it must pass actor, perception, selection and
independent envelope containment before independent+own60 stationary gates and
new navigation. No new mass model, perception retention or body radius is used.
The `outward_identity` comparison declares both changes from the accepted
baseline; the failed outward-only run remains an intermediate diagnostic,
never an accepted smoothness baseline. The original.5 s/50% actual-hold target
and every native safety/arrival/curve/kinetic gate remain unchanged.

Atcb45e5f the new zero-wheel component PASSED: one natural prior selection,
28 held-prior rows,115 containment rows (maximum centre error.180435 m),
actor/mass/motion/contact/numeric-policy checks and zero wheels. Independent
and own60-frame gates also passed. The motion trial nevertheless FAILED
`nearby_track_lost` at16.533333 navigation seconds,1/3 guides. Track323's
unresolved positive region was last seen at14.3 s; at24.6 s its distance was
1.196227 m, inside the unchanged1.2 m guard. Do not delete it or call it free.

The trial's nearby curve was.136190 m, hold3.266667 s and slow interval
3.283333 s. Physical clearance.121744 m, stopping drift.004416 m, analytical
margin.170324, five supported moving-kinetic rows and164 passing containment
rows do not establish navigation or smoothing acceptance. Motion used37 held
fresh-origin selections and **zero prior selections**: the parked prior-path
success must not be attributed to this motion trajectory. Original report SHA:
`a06756c82e9e3e32e72007ed9a5d34c94057e327c1db46d85ceb10c029c28b5a`.

All2,960 units,159 focused tests, Ruff, compilation/three CLI paths and the
original native fresh-selection regression audit passed before source freeze.
The accepted baseline and both older failed smoothing trials are unchanged.
The next reference hypothesis returns to strict identity0/outward0 and tests
maintaining moderate forward preference while turning, before the existing
affine projection and full exact checks. The offline original first-selected
snapshot accepts a.75 nominal-speed preference (.150129 m/s versus.118309),
without changing the nominal, field bytes/epochs or limits. This is not an
executed speed floor, a relaxed safety constraint or a native result.

## Turning translation preference

The separate default-zero`--guidance-turn-translation-fraction` is available
only in the explicit convex selected-obstacle parallel guide ablation. The
first candidate uses`.75` with the **accepted baseline activation2, outward0,
identity-selection0 and guide2**. Thus only this new reference/output option
differs from the accepted baseline invocation; the failed identity/outward
variants are not pooled into the comparison.

During a translating reference turn, when both desired-heading error and
goal-heading error are less than90 degrees, request at least the configured
fraction of nominal speed **before** the existing control-space affine CBF
projection. Turn-first references, headings away from the goal, goal arrival
and missing current selected identity keep their original behavior. The
projection may lower the preference to zero. All actuator clipping, predicted
field/temporal checks, exact DTCBF/dense geometry, candidate/execution freshness,
per-step analytical filtering and independent physical/stopping gates remain.
It is a candidate-generation preference, never an enforced minimum velocity.

The integrated trusted first-native-field replay reproduces the complete old
command sequence atdefault0 within1e-10. At`.75` the first planned forward
command increases from.118309 to.150129 m/s with an unchanged.79956 rad/s
turn, original field bytes/epochs and exact-safe planning. Neither this one
offline field nor a visibly larger curve proves smooth navigation. A fresh
strict-selection zero-wheel component, independent+own60 gates and unchanged
native/actual-hold evaluation are required. Use comparison`--variation
translation`; original native failures and all primary/main/long-service/PR117
holds remain in force.

Translation-reference source freeze passed2,983 units in107.50 s,113 focused
tests in7.58 s, Ruff, compilation, CLI help and whitespace checks. A real
box/halfspace projection test lowers the.15 m/s preference below.02 m/s,
confirming the preference cannot become a post-projection minimum command.
Native acceptance is pending.

At61d9c0e, the strict parked component (27 fresh-origin selection rows),
independent and own60 gates passed. The`.75`/activation2 native trial PASSED
goal/3 guides in21.283333 s, nearby curve.230668 m, clearance.405661 m,
drift.002443 m, minimum accepted analytical margin.102331, six supported
moving-kinetic rows and212 containment rows (maximum centre error.167662 m).
Its actual hold2.516667 s and slow interval2.566667 s FAILED the unchanged
smoothing target. Report SHA256:
`e0159b2bde930ba707582d296a2f4cdf1cee9e7fee512913450f41a7a701459e`.

A separately declared **two-option** trial at the same frozen source used
activation3 plus translation1 (strict identity0, outward0, guide2 unchanged).
Independent and own60 gates passed. Native acceptance PASSED goal/3 guides in
21.55 s, curve.228779 m, clearance.162942 m, drift.002432 m, analytical
margin.428904, five supported moving-kinetic and215 containment rows (maximum
centre error.215043 m within the unchanged.25 prior). Smoothing FAILED:
hold3.45 s, slow interval3.50 s. Report SHA256:
`e3633d32188a9b36609d384bb778989f7c63b2c29517a69a4f095962f8238367`.
The two invocation audits explicitly separate the changed activation and
translation arguments; this is not a single-variable or causal comparison.

The later trial's first passing-selection snapshot had no accepted guidance
reference; an offline lateral-reference-ranking probe made no difference at
that snapshot and was **not integrated**. The current MPC samples one current
configuration-space geometry and first-order temporal safety values. It does
not explicitly propagate the human body envelope through the horizon. The
approved next branch will test that additional assumption separately, keeping
original fields, factor policies, source clocks, exact/dense/analytical checks,
physical limits and all prior failures. Forecasts are not future measurements
or a new safety guarantee; stationary forecast validation must precede wheels.
