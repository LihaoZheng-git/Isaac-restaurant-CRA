# Static props viewed by a moving robot

The user approved moving-view association after the short crate navigation pass
exposed intermittent grouped identity. This separate default-OFF
`--moving-view-image-objects` mode requires the existing bounded approach and
its independent/own sixty-field gates. Legacy association remains available.

The mode projects the ORIGINAL positive world points through the current camera
calibration and ego transform. A local affine outline warp is fitted from those
original-to-current projections, with a maximum two-pixel residual and a
well-conditioned two-dimensional support requirement. This is explicitly an
association prior for a static prop, not dense depth or hidden geometry.
The current SAM mask must still meet the original .75 IoU, now against that
projected outline. A clipped outline or camera change beyond .5 m/10 degrees
rejects. The simulator ego/floor reference remains oracle/prior as before.

Every current selected return must stay within the ORIGINAL .06 m support
distance. Partial correspondence additionally requires eight distinct original
nearest-point correspondences, at least 25 percent of the original samples
supported, and at least 25 percent span in each significant original principal
direction. These new configurable engineering assumptions explicitly permit
partial observation; they are not a relaxation of physical clearance. Missing
old samples are recorded as unobserved, not deleted or claimed measured. The
original reference points/mask/camera never move or shrink through chaining.

All original current-fragment coverage, foreign-return, mask-quality, competing
mask, source-gap, qualification and original-expiry gates remain. Multiple
matching histories reject. Any geometric overlap with a rejected original
reference holds new enrollment rather than manufacturing another identity.
Failed groups still publish no group policy and lose property memory; raw
occupancy and ordinary policies remain. Current point geometry and original
raw-track kinetic estimation remain live. Matched static props can reuse the
existing source-tagged physical evidence; no score or object label is supplied.

The current surface centroid and original reference centroid are separately
labeled. Neither becomes a complete object center. Keep the original .20 m
whole-fixture-center audit and its failures. Association continuity must be
reported separately; a stable ID or a navigation pass cannot repair incomplete
geometry or certify the four physical estimators.

All reference comparisons are now logged, including low-IoU rejection paths.
In the opt-in mode every captured RGB/scan and actual grouping candidate packet
is archived for evaluation. Failed/repeated attempts get separate files. The
runtime never reads them; export time is included in build timing.

Validation covers synthetic camera/partial-support positives; wrong masks,
tiny patches, neighboring objects, ambiguous references, excessive viewpoint
change, drift chains, source gap/expiry, unchanged ownership and evidence-clock
negatives; recorded stationary model replays; then fresh independent/own60 and
bounded motion for both original broken crate and rounded bin. All earlier
failures, assets, accepted v2 and main/primary/PR117 holds remain. No v3 promotion
until the scoped results are reviewed. Artifacts: `outputs/moving-view-association/`.

Pre-native results: 3,801 unit tests passed in175.33s, with the final expanded
31 persistence/view/preflight tests passing in37.24s after the last gate change.
The earlier focused suites passed37 and33 tests; Ruff/compile/diff pass. Two
initial negative-test fixtures were corrected (one left the calibrated image;
the other incorrectly prohibited fresh new-ID qualification instead of old-ID
transfer). Their failed log is preserved; no association limit was loosened.

Fresh-model replay of the original stationary bin/crate archives reproduces
every overlap-hold accepted ID. The new mode retains the crate's16 target
updates; bin target updates change15->16 and non-target groups6->11. Those
additional background groups are reported separately, not bin successes. This
is association replay, not native navigation acceptance. The stricter motion
preflight additionally requires current memory for the independently matched
target, so background memory alone cannot authorize this test's wheels.

## Native results at 18b7bce

Both original licensed props ran separate independent sixty-field stationary
gates followed by an own-sixty gate and the original short approach. All four
stationary gates passed, with 58 accepted grouped target updates in each motion
preflight. Each completed navigation and stopping, with frozen source hashes:
0.303904 m actual travel toward the authored 0.40 m goal (original 0.10 m arrival
tolerance), 15 live fields/14 translating intervals, 0.550631 m minimum independent
clearance, and 0.002686 m stop drift. Minimum CBF margins were 0.694667 for the
crate and 0.687801 for the bin. These are short approach dataflow tests, not a
pass-by contrast or evidence that CRA changed the trajectory.

| Result | Broken crate | Rounded bin |
| --- | --- | --- |
| Original grouped ID accepted during motion | 8/15 fields | 7/15 fields |
| Original .20 m complete-fixture-center audit | 1/15 fields | 7/15 fields |
| Accepted current surface-center error | .1930--.3130 m | .1489--.1653 m |
| Runtime phi on accepted original-ID groups | .61260--.61653 | .60738--.65727 |
| Retained mass/contact entries, each | 7 | 5 |
| Continuous association acceptance | **FAIL** | **FAIL** |

All retained entries match their actual positive source values, timestamps and
confidence rules; current kinetic timestamps are current. This is not continuous
reuse of the stationary properties: the crate loses memory after source8.5,
requalifies the SAME ID at9.1, then acquires fresh mass/contact. Its uncalibrated
mass changes from2.5740 to.7177 kg because the current visible extent shrinks.
The bin also reacquires properties; its contact factor uses the.25 prior at one
accepted update. Do not claim stable rough mass, consistent contact contrast,
score freezing, or a complete fresh four-factor result.

The camera correction works on the recorded views. On accepted crate sources,
uncompensated maskIoU falls to.4156 whereas compensated IoU remains>=.8465; the
largest affine residual is.5103 pixels. Bin compensated IoU is>=.9030, residual
<=.8363 pixels. Original reference point/mask/camera hashes remain unchanged in
the exact candidate replays; no cross-ID evidence transfer occurs. The replays
include all60+15 committed fields for each motion run. Their legacy/new counts
refer to qualified groups before final assessment and may include background;
they are not matched native controller comparisons or target-success counts.

The crate's remaining failures have a concrete cause: sources8.6/8.7/8.8 include
2/3/2 actual current points beyond the original.06m support, reaching.08565m.
They trigger rejection and then two requalification fields. Sources9.6/9.7 have
three unsupported points each, reaching.11574m; the last source also falls below
the configured25% principal-span requirement. All these candidate points lie in
the independent fixture box, but this truth is evaluation-only and cannot confer
runtime ownership. Even an OFFLINE union of all stationary observed samples
leaves unsupported returns; simply collecting more of that same initial view
does not resolve it. The diagnostic never updates runtime references.

The bin has competing nonnested masks at8.4/8.7/8.8; candidate support fails at
8.6/9.1/9.2 (.06374/.07335/.09079m). The new camera matcher does not bypass these
original ownership/support checks. A strict additional point-box diagnostic
also records up to.004181m bin box excess; do not describe every bin point as
independently contained. The unchanged.20m center check passes the seven accepted
bin groups. Every rejected group's returns remain ordinary occupied geometry
with the existing policy, rather than receiving an invented object assessment.

Next bounded work must address verified newly exposed surfaces and ambiguous
whole/part masks, followed by source-preserving evidence recovery. Retaining
private properties through a rejected group would change the current explicit
"failed groups lose memory" rule; it is not part of this implementation. No
threshold widening, unsupported-point trimming, hidden geometry, source-clock
renewal or promotion into v3/main/primary/PR117 is justified by these runs.

Native artifacts are `crate-stationary-v1`, `crate-motion-v1`,
`bin-stationary-v1`, and `bin-motion-v1` under the output directory. The compact
`native-summary.json`, exact `candidate-replay-audit.json` and original
`group-persistence-audit.json` keep the separate acceptance claims. The figure
`moving-view-results.png` shows all live gaps; camera panels show accepted source
frames only. `crate-surface-loss.png` shows actual unsupported samples. All2,190
monitored earlier assets, accepted-v2 files and prior test archives are unchanged.

Final regression at the unchanged runtime source: **3,805 unit tests passed in
185.06 s**. Ruff over `robot scripts tests`, compilation of the changed runtime
and scripts, and `git diff --check` passed. The native manifest hashes1,697 new
archive files. Earlier failed fixtures and all native perception failures remain
available; no failing row was removed to improve the report.
