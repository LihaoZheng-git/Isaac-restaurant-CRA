# Restaurant CRA sensitivity candidate

This is a separate, default-off scoring experiment. The user authorized more
separation between verified physical assessments, unequal factor weights and
a fixed, more expressive color map. Ordinary unassociated obstacles retain
their existing policy. Original V3/V4 recordings, previous failures and
main/primary/PR117 promotion holds remain unchanged.

## Why the old scores looked similar

In the recorded V3, supported low-energy factors were raised toward the same
0.25 fallback by confidence adjustment. Missing potential and thermal factors
also each contributed 0.25 through noisy-OR. For example, retained bin/crate
contact cues differed substantially, but their median final phi differed by
only about 0.016. Most map returns belong to ordinary obstacles; changing
qualified-object scoring cannot turn those into measured low-risk objects.

## Candidate method and assumptions

`--restaurant-calibration kinetic` explicitly enables the candidate in the
existing persistent multi-object CRA path. `equal` uses the same candidate
with equal factor exponents; `off` preserves the original path.

Only objects owning current policy returns qualify, with accepted image-LiDAR
evidence, validated retained properties, or current validated human identity.
No asset name, bin/crate label, simulator object pose or desired ranking enters
scoring. Original geometry, evidence timestamps, ownership, mass/contact
retention and live motion contracts remain unchanged.

For a supported factor, the candidate uses its original upper-bound severity:

`q_i = clip(s_i * (0.75 + 0.25 * interaction_i) * (1 + 0.25 * (1-confidence_i)), 0, 1)`.

Contact uses the square of the upper silhouette/exposed-edge proxy as `s_i`.
This continuously attenuates weak outline cues, not an object-specific override.
Kinetic upper energy uses a 50 J reference instead of 100 J. Potential retains
100 J because the calibration pair contains no measured fall-energy contrast.
These energies are engineering references, not calibrated injury thresholds.
The context floor keeps intrinsic consequences visible beside the route.

A valid, retained visual contact proxy remains usable when only motion/context
is unavailable. In that case interaction uses an explicitly labeled upper prior
of one; kinetic still follows its own unavailable-motion prior. Source age,
status, bounds, approved persistence markers and confidence caps are checked;
no timestamp is refreshed and stale/oracle cues cannot be recovered.

Unavailable factors keep their original numeric policies and labels. Their
maximum forms one shared uncertainty baseline `p`, rather than treating repeated
missing-factor priors as independent hazards:

`physical_phi = p + (1-p) * [1 - product((1-q_i)^w_i)]`

over supported factors. This **correlated-uncertainty assumption** is a new
engineering hypothesis. With no supported factors, the original aggregate is
preserved. Factor exponents are kinetic 1.5 and thermal/contact/potential 1.0;
they are not percentages. A separately verified human receives a protection
floor of 0.80, recorded separately from physical_phi and contact sharpness.
Carried groups retain the maximum-member policy, never summed mass or a
fabricated carrying bonus. The current archive replay rejects carried groups
because that fixture does not contain them.

The shared flux mapping remains `b=-(0.5+1.2*phi)`, with lambda fixed at 1.25.
Waypoint nominal, MPC/CBF objectives, geometry margins, wheel limits and source
admission remain as in the original comparison. The candidate does change the
field through phi, so a score replay is not a new navigation validation.

## Calibration and separate archived V3 response

The short bin/crate runs supply the initial calibration comparison. Archived
V3 human responses also informed selection of the 50 J kinetic reference. The
original V3 mission is therefore a separate archived response check, not a blind
holdout. Correlated frames are not independent trials, and the prop geometries
and clearances differ; these results do not establish causal controller effects
or statistical generalization.

| Qualified retained records | Original median phi | Candidate median phi |
|---|---:|---:|
| Short bin, 120 rows | 0.61462 | 0.34103 |
| Short crate, 79 rows | 0.62338 | 0.52367 |
| Original V3 bin, 83 rows | 0.61567 | 0.35744 |
| Original V3 crate, 311 rows | 0.63136 | 0.49635 |

Candidate settings were compared with equal exponents and kinetic references
20/50/100 J. The 20 J factor-only human diagnostic compressed many scores near
one; 100 J left that diagnostic below the human floor. We selected 50 J for a
broader response in these inspected records, not as an independently validated
optimum or a measured injury threshold. Final ownership admission matters: only
29 human rows have both supported kinetic evidence and current owned returns.
The 258 verified human
records have median 0.80 and maximum 0.95660; two reach at least 0.95.

Fresh views still have outline outliers: the eligible fresh-bin maximum is
0.68729. The change does not guarantee bin/crate ranking in every view or fix
perception. Rough mass remains uncalibrated. Potential/thermal in these prop
records remain 0.25 priors, not measured zero fall-height or observed coldness.

## Source-bound replay and presentation

`scripts/evaluation/export_restaurant_calibration_replay.py` binds candidate
scoring to all 1,182 original exported fields and exact sensor/commit clocks.
It covers 119,260 object rows: 2,406 are eligible; 116,806 ordinary unassessed
rows remain exact. Current observations use candidate policy risk, with
original external risk floors preserved. Hollow prior geometry retains its
original score. This is intentionally not a reconstruction of a new controller's
memory lifecycle or trajectory. The original route, all pauses, masks and
source pixels remain; no new segmentation or obstacle positions are invented.

The `restaurant_v1` color map is a fixed 0..1 scale shared by camera and LiDAR.
It never normalizes to the current frame or object identity. A mask with zero
currently owned policy returns is withheld in the new presentation; a qualified
retained owner's projected points may still be shown. This avoids coloring a
superseded raw fragment with a second, conflicting object score. It does not
turn retained points into a fabricated whole-object mask.

Replay products and full response tables are under
`outputs/restaurant-risk-calibration/`. The composer accepts
`--calibration-replay <candidate_policy_fields_summary.json>` and uses a fresh
output folder. Original archive hashes, geometry hashes, field epochs and
camera source tuples are verified. Every navigation/stopping frame and the
original attribution card are retained.

## Validation status and native recipe

The full unit suite passed **4,083 tests in 235.36 seconds**. After final
presentation/ownership refinements, **100 focused regressions passed**.
Repository Ruff and `git diff --check` passed. These include scoring,
source-clock, ownership, field-integration, carried-group, unknown-policy parity
and visualization checks. After the post-native archive/presentation changes,
50 focused tests and repository Ruff passed. The first native outcome below is
not full acceptance.

The combined presentation and opt-in witness correction then passed **4,171
unit tests in 239.09 seconds**, with 15 CLI/recipe tests also run separately.
Repository Ruff and diff checks passed. Fresh corrected native validation is
still required; these tests do not override the first trial's failed audit.
The independent archived-certificate auditor adds seven passing integration
tests (4.38 seconds), run after that full-suite collection. Its read-only audit
of the first trial passes with zero recorded injective certificates and retains
`native_overall_passed: false`; it cannot grant navigation acceptance.

`configs/restaurant_service_risk_calibration.json` reproduces the exact original
V3 invocations with only the scoring opt-in added to perception recipes. Its
six fresh prerequisites and own 60-source parked gate must pass before new
wheels. Any native failure is retained; original acceptance thresholds and
physically meaningful MPC/CBF stops are not weakened to improve presentation.

## First fresh native trial: arrival with a failed memory audit

At frozen revision `7567c0b`, all six fresh prerequisites and the mission's own
60-source parked gate passed. The first motion run reached all eight route
guides in 156.0667 simulation seconds. Minimum independent clearance was
0.348148 m, minimum CBF margin 0.078256, stopping drift 0.002194 m, and both prop
PhysX contact counts were zero. The nearby-human excursion was 0.384546 m from
the unchanged nominal polyline. These are one-run observations in paced
simulation, not a causal smoothing comparison or real-time performance claim.

**The original native report is nevertheless failed.** Six static-human
exclusion checks at sources 132.7--133.4 s concern one five-point raw surface,
track 2987, last observed at 132.5 s. It was retained as an unclassified static
prior even though its points were near the human. This did not stop navigation;
it failed the final memory audit. Preserve the report, launcher rejection and
all original records under
`F:/CRA-experiments/restaurant-risk-calibration-20261107/long-motion-v1`.

The same-record scoring and source audit passed across 60 parked and 1,087 live
fields, with 115,863 object rows. This isolates scoring response from differences
between separate perception runs:

| Current-owned native evidence | Rows | Legacy median phi | Calibrated median phi |
|---|---:|---:|---:|
| Bin, fresh contact | 16 | 0.59836 | 0.36045 |
| Bin, retained contact | 83 | 0.61187 | 0.30948 |
| Crate, fresh contact only | 2 | 0.60763 | 0.56367 |

The crate has **no retained-contact rows in this run**. Its two qualified scores
span 0.51323--0.61410, and cannot establish reliable continuous acquisition or a
general ranking. A fresh bin outline outlier reaches 0.87836. High silhouette
cues also occur on furniture. Recalibration does not fix those perception
limitations. Among 242 verified human rows, 213 own current returns and 38 have
supported kinetic evidence; final phi spans 0.80--0.98315. Human mass remains a
configured adult prior, and its 0.80 protection floor remains separate from the
physical estimate. Zero-owner human rows do not become fresh calibrated data.

### Narrow geometry-proof repair under validation

Track 2987 had followed furniture before two centroid jumps, inheriting a
1.23631 m/s speed into the five-point near-human fragment. All five returns are
outside the current RGB projection, so existing visual association and the
1 m/s fragment-motion limit correctly reject human association. Those rules
must not be weakened to make this trial pass.

The existing whole-surface replacement does have a unique current owner and
five distinct supporting points within its unchanged 0.10 m distance bound.
Its greedy nearest-neighbor choices collapse to only one or two distinct
witnesses, causing a false negative in the distinct-support proof. The bounded
repair preserves legacy acceptance and adds a fallback requiring a complete
injective match of every distinct old XY point (at least three) to distinct
current XY points belonging to the same uniquely qualifying owner. It retains
the original full geometry, risk, timestamps, expiry and ambiguity checks; it
does not infer a human label or transfer factors. Exact replay, regressions and
a fresh native run are required before claiming this repair succeeds. The
first failed trial remains failed.

The implementation is explicitly default-off through
`--unknown-distinct-surface-witnesses`. The separate
`configs/restaurant_service_risk_calibration_witnesses.json` adds this flag only
to existing unknown-memory modes, including the parked human-sensor component;
the original scoring-only recipe stays unchanged. Sparse matching is bounded
to 256 distinct old points, 65,536 current points and 65,536 candidate edges.
Exhausting those budgets rejects only the additional proof. Exact witness
coordinates/indices, ownership, original/current source clocks and original risk
are recorded and checked independently. No human identity or factor evidence is
transferred by a geometric replacement.

The archived branch replay preserves all 548 original whole-surface accepts.
Among 541 previously rejected eligible regions, the fallback establishes 50
complete certificates and leaves 491 rejected. This replay is a check of the
geometric proof on frozen inputs, not a reconstructed controller trajectory or
new native acceptance. The final replay covers 1,147 parked/live fields. An exact
one-step memory replay of fragment 2987 accepts at ages 0.2/0.3 s and still
rejects after expiry at 0.5 s; its original geometry, risk and source clocks are
preserved. The follow-up native recipe changes both scoring and
the opt-in proof method relative to the older V3, so it must not be described as
a strict risk-only trajectory comparison. Same-record legacy/calibrated score
comparisons remain valid for studying the scoring function itself.
