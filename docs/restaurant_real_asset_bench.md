# Downloaded restaurant prop tests

The real recycling bin passes the existing stationary component gate, but its
accepted mask mainly covers the front blue panel. This is **not whole-bin
segmentation acceptance**. Both wooden crates fail visual ownership, so there is
no accepted intact-versus-broken contact comparison and neither is promoted to
the final demo as an assessed sharp hazard.

## Native results

Frozen runtime `ba13bd5`, recipe `configs/restaurant_real_asset_bench.json`:

| Isolated prop | Accepted target visual observations | Component gate | Median supported contact factor | Median phi on contact-supported rows |
|---|---:|---|---:|---:|
| Recycling bin | 18/18 | Pass, partial front-panel mask | 0.124677 | 0.544387 |
| Intact crate | 0/18 | Fail | unavailable | not a measured comparison |
| Broken crate | 0/18 | Fail | unavailable | not a measured comparison |

All three trials issued zero nonzero wheel commands, and source fingerprints
remained unchanged. The bin has 17 contact-supported rows after motion evidence
initialization; median replayed outline cue is 0.452555. Its last mask has 433
pixels and owns 488 LiDAR returns. The existing gate can accept a visible part:
the rendered bin's whole outline is larger than the accepted front-panel mask.
Do not describe its contact score or mass as full-object geometric validation.

The intact crate has a geometrically matched track in all 18 sources, but its
whole-object SAM proposal includes returns belonging to other LiDAR clusters.
At the final source, the 0.925-quality candidate includes 81 owned and 36 foreign
returns (foreign fraction 0.308). Cleaner fragment masks have quality below the
unchanged 0.85 threshold. The context-box retry also fails that quality gate.

The broken crate splits into several nearby tracks. Whole-object candidates
likewise include other clusters' returns; for the nearest final track, quality
0.924 accompanies 14 owned and 51 foreign returns (0.785 foreign fraction).
Its nearest track center is 0.231 m from the authored center, outside the original
0.20 m independent match gate. There are zero unique target matches. No thresholds
were widened and no manual group/identity/score was supplied to make it pass.
This is a segmentation-to-cluster ownership limitation, not proof that SAM cannot
outline a crate or that LiDAR sees nothing.

The intact crate's numeric prior-assisted phi is 0.68359375. It is not an
estimated broken/intact contact contrast. Failed runs retain their native error
and gate files; the launcher exiting zero is not counted as acceptance.

## Scope and limitations

The unchanged ABO RGB/LiDAR mass model remains an uncalibrated rough estimate.
The bin's potential factor remains a 0.25 prior; authored ground placement is
not silently supplied as observed fall height. Kinetic uses current tracked
motion and low-confidence mass, with a factor near 0.1875. Thermal starts with
its 0.25 prior and later reports a positive visual cue even though this scene
authors no plume (final cue about 0.401, factor about 0.146). That response needs
a separate negative-example review; this is not thermal validation or evidence
that the bin is hot. No complete fresh four-factor or navigation claim is made.

The actual camera pose is the prior bench's front Owl +0.60 m level mount at
robot `(26.3, 6.1)`. Each prop is centered at `(28.15, 6.1)` and grounded in a
separate additive scene. The bin is authored at 0.65 m height. Both crates use
the same 0.65 uniform source scale, resulting in heights about 0.224/0.229 m.
The distant server is disabled only in these isolated stationary layers.
The simulator ego pose remains explicitly oracle; fixture truth scores outputs
only. Accepted v2's 107 archived files were fingerprinted and remain unchanged.

## Assets and next candidates

The user's ZIP files retain their supplied CC BY 4.0 licenses and attribution:

- [Blue recycle bin, SUSUSUBE](https://sketchfab.com/3d-models/blue-recycle-bin-5516523f31f343e68da70629555291de).
- [Wooden Dirty Crates - Broken Whole 2 Piece Set, Sunbox Games](https://sketchfab.com/3d-models/wooden-dirty-crates-broken-whole-2-piece-set-1ebaf13ac44748baa845ab7f5cb3fc9f).

Original geometry/textures remain byte-identical; the installed Isaac converter
generated USD. The bin download contains overlapping low-detail assembled,
component and high-detail baking meshes; the scene selects the original textured
803-triangle assembled mesh without changing its geometry. The crate selection
preserves broken boards. All transforms/selections are described in
`assets/restaurant/sketchfab_contact_props/README.md`; no asset category, density
or contact labels enter the estimator.

Preferred next sharp-object candidate is
[Kitchen Cleaver by KAMAAIL](https://sketchfab.com/3d-models/kitchen-cleaver-93d395fa7cf04811a7cc56f5169c30d5),
listed as downloadable, CC Attribution, about 1.3k triangles. A realistic broad
blade on a low prep trolley, presented face-on, is a more promising observation
than tiny fork tips. This is an engineering hypothesis, not tested acceptance;
the edge remains thin and this proxy cannot measure physical blade sharpness.

[Broken Plate (low poly) by Akimkhan](https://sketchfab.com/3d-models/broken-plate-low-poly-1cb31d29ec0d4289b43f1794fa05abe1)
is another downloadable CC Attribution candidate (608 triangles, with original
Poly by Google credit). A dropped dish is a natural restaurant accident, but
low height and disconnected shards likely make it a harder ownership test.
Neither additional candidate has been downloaded or tested. Do not exaggerate
scale, manually increase CRA or replace priors to manufacture a contrast.

## Reproduction and evidence

Create fresh scenes with `scripts/scene_creation/build_restaurant_real_asset_bench.py`
using `--base scenes/generated/restaurant_service_demo_v2d.usda`, a fresh output,
and `--kind recycle_bin|intact_crate|broken_crate`. Then use
`scripts/evaluation/run_restaurant_demo.py --config configs/restaurant_real_asset_bench.json
--case <kind> --stage stationary --output <fresh-folder>` from the installed environment.
The runner fingerprints the referenced layers, original/converted assets,
textures and runtime sources. No packages were installed or upgraded.

Local records are `outputs/restaurant-online-assets/{recycle-bin,intact-crate,broken-crate}-v1`.
`comparison.png` displays native RGB and accepted masks; unsuccessful cases use
explicitly labeled display crops without masks. `comparison.json`,
`association-diagnosis.json`, `asset-import-audit.json` and the v2 preservation
reports retain the evidence. Gate SHA256 hashes, in table order:

- `5112c419bccf31397785d8e8a864fe34cee393c400d64fafd852d3e8b998761b`
- `f4cb249ec9af4911bccbdf91ae6a43ee4c81ea527224f75ab2db0ba4045afc56`
- `c333e044ab523a8b19b786e4d610c2745a967887927f821452ca7b929b7f9e32`

Focused tests: 11 passed. Full unit regression: **3644 passed in 124.60 s**.
Ruff passes for `robot scripts tests`. Keep this on a draft development branch;
accepted v2, primary controller, main and PR117 holds remain unchanged.
