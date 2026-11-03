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
No wheel trial is part of this experiment.

## Fresh native results

Frozen runtime `ae599d3`, 18 stationary sources per case:

| Case | Verified target observations | Native component | Result |
|---|---:|---|---|
| Bin with extent retry | 18/18 | Pass | 13 broader masks; 5 panel fallbacks remain |
| Cleaver, flat on low support | 0/18 | Fail | No separately matched/accepted cleaver instance |
| Broken plate on floor | 0/18 | Fail | No matched/accepted plate instance |

All three report zero nonzero wheel commands and unchanged runtime/scene source
fingerprints. Both failures retain their native error, gate and execution files.
The launcher exiting zero does not turn either failed component into acceptance.
Local folders are `outputs/asset-mask-diagnosis/{bin,cleaver,plate}-native-v1`.

The bin's last selected mask grows from the same-source 430-pixel panel covering
203/832 returns to 1,970 pixels covering 827/832 (99.40%), quality 0.875, with
zero foreign hits and complete containment of the original panel. The broader
outline is visually much closer to the bin body; it is not annotated whole-object
IoU or general segmentation acceptance. Five frames still use the existing
small-panel fallback. Consequently inferred mass changes between about 0.48--0.59
kg on those parts and 3.39--4.34 kg on broader masks. This is an unresolved
part/whole consistency limitation, not physical mass change or calibration.

There are 17 supported contact rows, median silhouette cue 0.763993, contact
factor 0.165907 and prior-assisted phi 0.618809. The first row instead has phi
0.968457: before motion initialization, the existing interaction fallback is
1.0 and the contact prior policy preserves the higher 0.925231 proxy risk. Keep
that row; do not report only the settled range as the whole run. Potential and
thermal remain 0.25 priors here. A larger outline cue does not establish that
this ordinary bin is physically sharp. The earlier no-plume false-cue review
also remains unresolved; this prompt experiment is not thermal validation.

In the cleaver scene, the nearest track belongs to the front support surface;
its final centroid is about 0.263 m from the cleaver center. Its SAM proposals
describe support geometry and peak below the existing 0.85 quality threshold.
No authored target box or category is supplied to help the estimator. An
independent bounding-region audit finds zero scan returns more than 2 mm above
the support and within the thin cleaver region in all 18 sources. This audit
scores data only and does not establish exact mesh membership.

The plate is visible in RGB. The same audit finds zero above-floor-region scan
returns in all 18 sources, before and after obstacle filtering. Its entire
0.026 m authored height is also below the existing 0.05 m obstacle filter, so
changing a SAM prompt alone cannot resolve this fixture. We did not increase
size/thickness, lower the floor threshold or assign a sharpness score manually.

`diagnosis-overview.png` shows original and new bin masks, rejected crate masks,
and the actual cleaver/plate views. `bin-sequence-native.json`,
`bin-sequence-replay.json`, `thin-prop-return-audit.json` and `comparison.json`
retain detailed diagnostics. All 107 accepted-v2 archive files and every original
licensed asset source hash remain unchanged (`preservation.json`).

Validation: **46 focused tests**, **3,659 full unit tests in 116.20 s**, Ruff for
`robot scripts tests`, and diff whitespace checks pass. The first focused launch
could not create pytest's sandbox temporary directory; rerunning with a fresh
workspace-local basetemp passed without environment changes. Native gate hashes:

- Bin: `e38d3dcf0fc2fed60095032e1da1e7ac22cd41d9536d8bec12ce4693fe6821c3`
- Cleaver: `15a82df292962889e0e631b872f5789b79b3f91a3ee6900cf25f7b2ec2464550`
- Plate: `0e40c52ab2b44bd87afe76d0d62dbc2a39b65370c61d303beb76ff73196cfd0a`

The next useful perception work is consistent part/whole mask selection and
carefully validated image-guided ownership across fragments of one object.
Simply permitting arbitrary foreign clusters would also merge adjacent objects.
For a first contact demonstration, the visible broken crate remains a better
candidate for that association work than these thin, poorly sampled utensils.
No sharp-contact contrast, complete fresh four-factor, moving-camera or full
restaurant-demo acceptance is established here. Keep the option default-OFF and
the change on a draft development branch; primary/main/PR117 holds remain.
