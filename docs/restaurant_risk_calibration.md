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
recorded below; these tests alone do not override the first trial's failed audit.
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

### Narrow geometry-proof repair

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

## Corrected native trial and exact archive validation

At frozen revision `ae5c7e8`, the separate witness recipe passed all six fresh
prerequisites, its own 60-source parked gate and the complete native mission.
The launcher exited successfully with no traceback and unchanged sources. The
original failed trial at `7567c0b` remains failed and unchanged.

| Native result | Observed value |
|---|---:|
| Route guides reached | 8/8 |
| Navigation simulation time | 155.3833 s |
| Minimum independent clearance | 0.347619 m |
| Minimum analytical CBF margin | 0.194872 |
| Stopping drift | 0.002249 m |
| Bin / crate PhysX contact points | 0 / 0 |
| Live fields / fields committed while translating | 1,084 / 1,082 |
| Nearby-human curve from the nominal polyline | 0.378048 m |
| Minimum human-center distance | 1.415778 m |

The measured nearby-walking translation hold below 0.02 m/s was 0.0333 s; the
longest interval below 0.05 m/s was 0.0667 s. These are one-run measurements,
not a matched smoothing comparison or a pause-free whole-mission claim. The
existing checked-guidance unicycle controller variant, fixed lambda 1.25,
nominal route, geometry allowances and physical limits remain explicit. This
was paced development simulation, not real-time performance or exact-paper
equivalence.

The final source-bound audit compares actual active scores with the saved
legacy equation on the **same observations**, not a second controller run:

| Current-owned evidence | Rows | Legacy median phi | Active median phi | Active range |
|---|---:|---:|---:|---:|
| Bin, retained contact | 75 | 0.610283 | 0.293853 | 0.292942--0.294344 |
| Crate, retained contact | 308 | 0.629747 | 0.449108 | 0.442518--0.498761 |
| Human, verified current ownership | 205 | 0.683594 | 0.800000 | 0.800000--0.973263 |

The broader qualified target sets contain 89 bin and 311 crate rows. Fresh
current-owned contact subsets contain 12 bin and three crate rows; their ranges
still overlap. Two qualified bin rows have unavailable raw contact cues and
remain present with explicit prior handling. The broader visualization coverage
report counts 14 bin and five crate visual observations; those counts are not
equivalent to the current-owned fresh-contact subsets. Crate retention is
intermittent, and its 20 unavailable-speed updates remain visible. A changing
phi does not imply that the stored contact cue changed: motion/context and
confidence still affect the result.

There are 249 verified-human rows, including 44 with zero current policy returns.
Those do not become fresh current-owned assessments. The scoring audit finds
29 current-owned rows with supported kinetic input; the native moving-kinetic
acceptance metric counts 28 under its separate moving-actor criteria. The 0.80
human protection floor is a labeled prior. Potential and thermal remain factor
priors in this scenario, and there are no complete fresh generic four-factor
assessments.

All archive checks passed:

- Exact scoring, source clocks and target point ownership over 60 parked and
  1,084 live fields; all 35 recorded injective-support certificates independently
  bound to original/current arrays, ownership, risk, timestamps and field proof.
- Static property replay: 1,624 retained rows across 16 objects. Geometry replay:
  35,953 references and 738 original archives. Motion replay: 1,201 ambiguous
  proofs. Fragment provenance: 4,710 proofs. These are provenance checks, not
  calibrated motion accuracy or general occlusion guarantees.
- Current dynamic evidence: 213 body joins, zero part joins and 19 uncertain
  human rows. Zero native recovery events; this trial does not validate recovery.
- Camera presentation: 1,194 displayed masks across 67 tracks, 1,144 verified
  source pairs, five accepted zero-owner records withheld, exact pixel/score and
  timing parity. The panel holds the latest assessed RGB between updates.
- LiDAR export: all 1,144 fields, 31,664,061 current points and 23,331,387 retained
  geometry samples; 3,861 source-file hashes checked. Receipt selection withheld
  83 later commits sharing a timestamp. No frame-wise color normalization or
  recalculated display-only risk was used.

The native archive is
`F:/CRA-experiments/restaurant-risk-calibration-witnesses-20261107/long-motion-v1`.
Exact score/certificate outputs are under
`outputs/restaurant-risk-calibration/witness-native-final-audits-v3/results/`;
the map audit is `outputs/restaurant-risk-calibration/native-lidar-audit-v1.json`.
Two earlier audit-helper formatting failures on unavailable contact cues remain
preserved. The corrected summaries retain null cues and explicit unavailable
counts; all required score values were independently checked finite. No runtime
gate or source data was changed to pass those audits.

### Reading retention and report counters correctly

Human mass remains the configured 70 [45, 100] kg prior. Body-identity retention
does not extend its original three-second mass expiry: the native mass audit
records 16 original priors and 57 retained epochs, with retained ages up to
2.9667 s. Static mass/contact has a separate 120-second expiry and 60-second
confidence half-life; maximum observed target ages were 11.3 s for the bin and
61.3667 s for the crate. Current LiDAR ownership still has to qualify.

The legacy `accepted_current_instances=0` report field reads a single-object
diagnostic and is not a valid multi-object count. The corresponding multi-object
counter records 139 accepted visual assessments, of which 134 own current
returns; these are not unique-object counts. Likewise, the report's 0.15-second
occupancy wording concerns the recent-scan queue. Persistent background and
unknown-geometry priors operate separately. These clarifications are bound to
native records and source hashes in
`outputs/restaurant-risk-calibration/native-limitations-clarification.json`.

## Validated native presentation

The new package is `outputs/final demo/v5-risk-calibrated`; share
`restaurant-service-v3-calibrated-share.mp4`. It contains 1,565 consecutive
navigation/stopping frames (original source frames 69--1633) plus 60 asset-credit
frames: 1,625 frames, 162.5 seconds, 1920x1080 at 10 fps. Master and H.264 share
copies both passed complete FFmpeg decoding and exact frame/format checks.
The share file is 83,178,024 bytes. Bin, human and credit frames were visually
inspected after encoding.

The left panel preserves the assessed camera view; the upper-right panel uses
actual controller-applied LiDAR policy scores; the lower-right panel is the
native follow camera. The fixed `restaurant_v1` palette is used throughout.
Camera scores are individual-object values, while map scores include recorded
group/prior policy handling. The map footer's "original scores" means the scores
recorded in this new calibrated native trial, not the diagnostic legacy equation.
No-mask frames can still show current LiDAR risk with qualified retained
properties. Held RGB, scan age, ordinary policies and prior geometry stay labeled.

All 1,100 presentation-input hashes were rechecked, and all 444 previously
protected demo files remain byte-identical. The package includes the native
report, score table, certificate/map audits, source/frame mapping, credits and
`video-validation.json`. Both the first failed native trial and the clearly
labeled earlier archived-score replay remain separate. No main/primary/PR117
promotion is implied; PR247 stays draft.
