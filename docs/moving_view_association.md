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

Validation in progress: synthetic camera/partial-support positives; wrong masks,
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
