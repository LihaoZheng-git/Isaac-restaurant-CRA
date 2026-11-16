# Downloaded vase stationary comparison

The downloaded vase set is usable for stationary image--LiDAR perception, but
the tested intact/broken pair does **not** establish a sharpness contrast with
the current silhouette contact proxy. All four component runs passed; the
broken version scored slightly lower in both matched comparisons. Preserve
this negative comparison rather than assigning risk from the asset name.

## Asset and setup

User download: **Vase (Normal and Broken)** by **Fridge**, Sketchfab, CC BY 4.0:
https://sketchfab.com/3d-models/vase-normal-and-broken-aa618cea451d445087925607b4051281

Original ZIP SHA256:
`43e88d700276bab4a4c5156b304cdf2213e7d2cb5f7e980ad30b2eddae356d23`.
The source license, glTF, binary and texture remain unchanged under
`assets/restaurant/sketchfab_contact_props/vases/`, with extracted hashes in
`manifest.json`. The installed Isaac converter produced the referenced USD;
there was no environment upgrade or new model download. Share the CC BY credit
with derived images/video.

The bench selects the original intact `Vase_ModelMain_0` and upright broken
`VaseBroken2_ModelMain_2` before scoring. Both keep their original inner and
outer ceramic meshes, topology and materials. One uniform scale gives the
intact vase 0.40 m height and both versions about 0.255 m width. Broken height
is about 0.348 m; it is not independently enlarged. The other two broken
variants remain in the source asset, untested. The separately rejected
"Broken Vase" asset was not used.

Separate generated scene layers place each version on the floor at
(28.15, 6.10) m, disable the background server for the prop-only bench, and
preserve the base restaurant and accepted v2 demo. Robot pose, Owl camera
raised 0.60 m, lens, original LiDAR and 5 cm obstacle floor filter are matched.
No asset identity, authored bounds, density, mass or contact label enters CRA.
Authored fixture bounds are independent evaluation metadata only. Runtime
placement/motion still use the declared Isaac ego-pose oracle.

## Native results

Each run recorded 18 distinct sources and zero wheel commands. The first two
sources qualify repeated support; subsequent 16 sources have accepted grouped
image--LiDAR assessments, current inferred motion, rough mass and supported
contact. Each accepted vase comprises one raw LiDAR fragment in these views.
All original quality, ownership, temporal and unique-target gates remain.

| Native resolution | Vase | Accepted assessments | Selected returns | Median contact factor | Median CRA phi |
| --- | --- | --- | --- | --- | --- |
| 640 x 400 | Intact | 16/18 | 146--151 | 0.124841 | 0.600034 |
| 640 x 400 | Broken | 16/18 | 111--118 | 0.114514 | 0.595315 |
| 1920 x 1200 | Intact | 16/18 | 146--152 | 0.124337 | 0.599804 |
| 1920 x 1200 | Broken | 16/18 | 112--118 | 0.110948 | 0.593686 |

The 640 captures froze revision `9fc86b1`; 1920 captures froze `4d510d8`,
which adds only matched diagnostic recipes. All four execution records verify
unchanged source and scene hashes. The higher-resolution option is the existing
stationary-only sampling option, not image upscaling or a changed runtime
camera default. Mask sizes increase from roughly 531--601 to 4580--5485 pixels.
Median field build wall time is about 1.94--1.95 s at 640 and 2.72--2.79 s at
1920; these paced stationary tests do not establish real-time performance.

## What the comparison means

The current contact estimator takes the maximum of circularity deficit,
elongation, concavity and a polygon-corner cue. Replaying all 64 accepted masks
through the unchanged estimator exactly reproduces their recorded contact
severity upper bounds. This is not an object-ownership or CRA wiring failure.

The intact vase's long neck drives an elongation cue around 0.45. The broken
version has a shorter, broader outline and the visible break is largely inside
that silhouette. Its median proxy cue is 0.362 at 640 and 0.329 at 1920.
Higher resolution makes the texture and opening clearer, but does not produce
a stronger outline proxy. Geometry/texture visibility is not physical edge
radius, material hardness or calibrated injury severity. No thresholds were
tuned, masks filled, or scores overridden to force the expected ordering.

Contact factors also include the separately inferred interaction likelihood
and uncertainty prior; they are not the raw silhouette cue. Potential and
thermal remain explicit 0.25 priors in these tests. Ground placement is not
passed in as proof of ground support, and object height is not fall height.
The category-free mass network supplies uncalibrated estimates with confidence
0.25 and broad bounds; visible LiDAR extents under-observe the full vase.
Neither mass ordering nor full four-factor sensor readiness is established.
Numeric phi is available under the approved prior-assisted policy, mapped to
Safe-SAGE flux with fixed lambda 1.25; there is no new navigation trial.

The bounded next improvement is to evaluate local exposed-rim/tip evidence
separately from whole-object elongation, with intact rims, smooth elongated
objects, patterned surfaces and multiple views as negative controls. Keep the
current results as a baseline. Do not promote this asset as the final
high-contact demonstration until that distinction is supported by observations.

## Reproduction and validation

Build fresh scenes with `scripts/scene_creation/build_restaurant_real_asset_bench.py`,
base `scenes/generated/restaurant_service_demo_v2d.usda`, and kinds
`intact_vase` / `broken_vase`. The recipe scene names are recorded in
`configs/restaurant_vase_contact_bench.json`. Then, from a clean committed tree:

```text
python scripts/evaluation/run_restaurant_demo.py --config configs/restaurant_vase_contact_bench.json --case intact_vase --stage stationary --output <fresh-output>
```

Use `broken_vase`, `intact_vase_rgb1920` or `broken_vase_rgb1920` for the other
matched cases. Preserve all old output paths. Local captures, source snapshots,
native reports, figures and exact mask replay are in `outputs/vase-contact-bench/`.
The comparison figures are `native-comparison-v1.png` and
`native-comparison-rgb1920.png`; `contact-replay.json` records severity parity.

Focused geometry/association/SAM tests: **61 passed**. Full unit regression:
**3711 passed in 132.62 s**. Ruff across `robot`, `scripts`, `tests` and Git diff
checks passed.
All 106 indexed accepted-v2 files and every original licensed asset file match
their preserved hashes. No new wheels, final-demo, main or primary acceptance;
all existing main/primary/PR117 holds remain.
