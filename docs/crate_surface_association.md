# Broken crate: observed surfaces and association continuity

This bounded branch follows the rounded-bin comparison, preserving all twelve
original native results and the accepted v2 demo. No wheel motion, main/primary
promotion or PR117 change is part of this work.

## What the archived sensors show

The broken crate's failed 1.6 m view does not lose its missing slats at the
image-association gate. The current above-floor scan itself has only 20--26
returns in the independent target region. For all eleven assessed sources,
all target-region returns survive projection, visibility, mask and depth-layer
selection. They form two fragments near an edge. The previously successful
1.85 m view has 64--67 returns across three fragments. The independent authored
box is used only for this point-loss audit, never to recover ownership.

Thus association cannot manufacture a complete center or extent for the sparse
view. The existing 0.20 m fixture-center check is retained, along with its failure.
New diagnostics explicitly identify the reported center/extent as those of the
current observed returns, not a verified complete physical object.

A separate continuity defect appears when small sample changes violate the
original 0.06 m whole-support gate. The legacy state enrolls another hypothesis
with a nearly identical mask. Later observations can satisfy both histories and
are rejected as ambiguous. The original 1.85 m bin has exactly this pattern.

## Bounded correction

`--image-object-overlap-hold` is default OFF and requires the existing stationary
image-object experiment, which excludes humans and wheel motion. It adds one
enrollment rejection: when no original hypothesis fully matches, but a prior
mask meets the existing 0.75 IoU and at least one current point is within the
existing 0.06 m support distance, do not enroll another overlapping hypothesis.
Log the rejected source and original references instead.

This rejection is not an identity merge, geometry repair or partial-support
acceptance. A later accepted observation must still pass the FULL original
bidirectional support distance, mask, three-source/span/gap and other gates.
Rejected observations do not update original source/proof/support/factor clocks.
The original ten-second reference expiry stays; spatially separate objects can
still qualify through the existing three-source path. Multiple existing matches
remain ambiguous. Current raw occupancy and ordinary unknown policies remain.

Contact estimation, mask refinement setting (OFF), model weights, confidence,
uncertainty/factor priors, fixed lambda and the independent target gate do not
change. The bin may recover valid source assessments that were previously
blocked by its duplicate; their scores are computed from those current images.

## Validation plan and artifacts

- `scripts/evaluation/replay_image_object_association.py` reads every original
  RGB PNG and scan with the installed audited SAM weights. Legacy and opt-in
  states receive identical same-source candidates. It records exact mask/point
  hashes and verifies source files remain unchanged. It does not run final CRA
  field commit and must not be described as native acceptance.
- Unit negatives cover rejected sample persistence, no clock renewal, expiry,
  separated objects, existing ambiguous identities, support gaps and unchanged
  raw occupancy. The legacy path remains tested.
- `configs/restaurant_crate_surface_association.json` repeats the original
  1.85 m bin/intact/broken views plus the 1.6 m broken-crate stress case, with no
  new pose search. Each fresh native trial requires eighteen stationary sources
  and zero wheel commands. Keep failures and all startup rows.
- Diagnostics and reports are under `outputs/crate-surface-association/`.

The four full archived sensor replays reproduce every legacy accepted object ID.
At 1.85 m the hold changes bin acceptance from 1 to 15 of 18 sources; the one
bad-support source stays rejected. Both crate sequences retain their original
16 assessments. At 1.6 m the broken crate has only 7 instead of 11 group
assessments: the sources that violate the original support do not gain new
identities and cannot qualify. This is deliberately stricter, not a repair of
its missing physical samples or fixture-center failure. See `replay-v1/summary.json`.

## Fresh native results at a81fa05

All four fresh runs complete eighteen sources with zero wheel commands and
unchanged runtime/scene hashes. Three original-view components pass; the closer
broken-crate stress case fails. No acceptance limit or score changed.

| View | Object | Component | Target assessments | Median contact | Median phi |
| --- | --- | --- | ---: | ---: | ---: |
| 1.85 m | Rounded bin | PASS | 15/18 | .090913 | .584530 |
| 1.85 m | Intact crate | PASS | 16/18 | .101168 | .589222 |
| 1.85 m | Broken crate | PASS | 16/18 | .109458 | .593010 |
| 1.6 m | Broken crate | FAIL | 0/18 | -- | -- |

The bin's source84 violates complete original support by .061964 m versus the
unchanged .06 m limit. It remains rejected; source90 again satisfies the original
hypothesis and recovers without a competing bin identity. Both crates pass the
same original-view setup. The bin run also has six non-target grouped assessments,
retained separately in `native-summary.json`; they are not bin successes.

At 1.85 m the bin/intact/broken raw cue medians are .142786/.228227/.317193.
No exposed-edge specialist joins occur on these matched targets. Exact current
selected-return ranges are 366--377, 115--121 and 63--66 respectively. The three
contact ranges overlap: .085811--.148186, .093597--.151737 and .096030--.155194.
All matched sources, including startup, are included. Each pair has the intended
ordering in 14/15 common recorded frame indices, but these are separate runs,
not synchronized samples or a statistical calibration. The contrast is modest.

The closer broken crate still lacks sufficient spatial coverage: five runtime
group assessments have 21--23 returns and 3D surface-centroid distances
.3138--.3176 m from the independently authored center. None passes the original
.20 m fixture test. Eight candidate sources are held rather than enrolling
overlapping histories. This stricter association does not recover unsampled
slats; preserve `native/broken_crate_mid-v1` as a failure. The replay's seven
groups and the fresh run's five are different captures, not interchangeable.

Median field builds are 1.87--1.97 s. These are paced stationary diagnostics,
partly overlapping the unit suite, not real-time acceptance. All original
synchronization-rejection logs are preserved, including those in passing runs.

Validation: **3,774 unit tests passed in 177.75 s**, **31 focused tests passed in
36.69 s**, Ruff across `robot scripts tests` and the complete branch diff check
pass. The first integration test used a point-tight synthetic mask that changed
IoU with its perturbed return; its failed log is preserved. The corrected fixture
uses a fixed visual silhouette to test the intended same-mask support defect;
no runtime threshold was loosened.

All **1,322** files in the preservation snapshot remain byte-identical: the full
accepted-v2 tree, prior assets and all twelve original rounded-trash native
archives. See `preservation-final.json`, `native-summary.json`,
`ranking-diagnostic.json` and `comparison.png` under the output directory.

We now have a passing shared-view stationary bin/crate comparison. The improved
native acceptance is the bin's recovery; the broken crate already passed this
original view and remains a valid candidate here. Its sparse closer view is
NOT repaired. No component result establishes strong contact separation,
calibrated sharpness, complete four-factor sensing or navigation readiness.
Potential/thermal .25 priors, uncalibrated mass and simulator ego reference remain
explicit. Keep the draft/default-OFF scope and all main/primary/PR117 holds.
