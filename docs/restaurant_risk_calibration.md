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

## Archived calibration and held-out response

The short bin/crate runs supply the calibration comparison. The separate
original V3 mission is used as a response check. Correlated frames are not
independent trials, and the prop geometries/clearances differ, so this is not a
causal controller or statistical generalization claim.

| Qualified retained records | Original median phi | Candidate median phi |
|---|---:|---:|
| Short bin, 120 rows | 0.61462 | 0.34103 |
| Short crate, 79 rows | 0.62338 | 0.52367 |
| Original V3 bin, 83 rows | 0.61567 | 0.35744 |
| Original V3 crate, 311 rows | 0.63136 | 0.49635 |

Candidate settings were compared with equal exponents and kinetic references
20/50/100 J. The 20 J factor-only human diagnostic compressed many scores near
one; 100 J left that diagnostic below the human floor. 50 J retains a broader
response. Final ownership admission matters: only 29 human rows have both
supported kinetic evidence and current owned returns. The 258 verified human
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
and visualization checks. Native outcomes will be recorded after completion;
no fresh navigation pass is asserted by this document yet.

`configs/restaurant_service_risk_calibration.json` reproduces the exact original
V3 invocations with only the scoring opt-in added to perception recipes. Its
six fresh prerequisites and own 60-source parked gate must pass before new
wheels. Any native failure is retained; original acceptance thresholds and
physically meaningful MPC/CBF stops are not weakened to improve presentation.
