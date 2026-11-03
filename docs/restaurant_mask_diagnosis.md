# Real-prop mask diagnosis

This separate stationary experiment investigates the partial recycling-bin mask
and failed wooden-crate associations from `ba13bd5`. It does not change accepted
v2, primary navigation, ownership thresholds, factor priors or any wheel policy.

## Cause established from archived sensor inputs

Replaying all original scans reproduces their track centers exactly. On the last
bin image, the median owned LiDAR point lands on its recycling panel. SAM scores
the small panel at 0.906 but the whole-bin point-prompt candidate at only 0.280.
The panel covers 203/829 projected owned returns (24.49%), exceeding the existing
20% part-coverage threshold. Acceptance therefore did not mean whole-bin extent.

An observed-points box with 25% symmetric padding produces a whole-bin-looking
candidate at quality 0.872, covering 825/829 returns with no foreign hits. This
is an offline observation, not native validation or annotated segmentation IoU.

The crates have a different failure. SAM can outline them, but the slats form
separate LiDAR clusters. The original whole-crate masks contain 30.8% foreign
returns for the intact crate and 78.5% for the nearest broken-crate fragment,
exceeding the unchanged 15% threshold. A tighter box can accept only an intact
crate rim; that is not a whole-crate solution. No IDs are merged or foreign
returns relabeled. See `outputs/asset-mask-diagnosis/prompt-replay.{json,png}`.

## Bounded proposal option

Default-OFF `--sam-track-extent-retry` requires `--sam-observed-context-box`.
Only an already accepted first point-prompt mask with less than 80% of its
current track's projected returns triggers one additional box proposal. The box
uses that same source's owned points and 25% padding. Replacement must pass
every original mask/quality/foreign-hit/depth gate, cover at least 80% of the
track, and contain at least 98% of the original mask. Otherwise the original
valid part remains, with a logged unsuccessful retry. Explicit alternate-point
and box retries never recurse. This adds inference work, not a confidence boost.

Both fractions describe the current track, which may itself be only a part.
They do not certify whole physical objects, merge clusters, invent points or
transfer factor properties. Diagnostic records preserve original and retry
candidates, coverage, containment, quality failures and the selection decision.

## New downloaded props

The user's Kitchen Cleaver (KAMAAIL) and Broken Plate (Akimkhan, original Poly by
Google credit) retain their supplied CC-BY-4.0 licenses, original source hashes
and all three meshes each. Installed Isaac conversion produced USD without
package changes. The cleaver is 0.32 m long and about 0.011 m thick, flat on a
separate 0.45 m-high preparation surface. The plate is 0.28 m across, about
0.026 m high, on the floor. These are evaluation placements, never runtime
class, density, detector-box, sharpness or height evidence. Thin geometry is
not enlarged for LiDAR acceptance.

`configs/restaurant_mask_diagnosis.json` defines bin, cleaver and plate cases.
Build fresh layers using `scripts/scene_creation/build_restaurant_real_asset_bench.py`
and the original `restaurant_service_demo_v2d.usda` base. Run each with
`scripts/evaluation/run_restaurant_demo.py --config configs/restaurant_mask_diagnosis.json
--case <bin_extent|cleaver|broken_plate> --stage stationary --output <fresh-folder>`.
Each capture requires clean committed sources and independent source fingerprints.
No wheel trial is part of this experiment. Native results are pending at this
implementation revision; do not promote a new prop from scenery alone.
