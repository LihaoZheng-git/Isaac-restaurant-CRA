# Separate parallel-walking smoothing experiment

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
