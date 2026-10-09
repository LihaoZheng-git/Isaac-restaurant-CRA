# Image2Mass versus the existing visual mass model

The 2026-10-09 offline comparison favors retaining the existing frozen ABO
MobileNetV3 regressor for the current bounded thesis scope. It orders relative
heaviness better on both evaluated sets and is faster on the tested GPU.
Image2Mass remains an evaluated candidate, **not an integrated robot backend**.
No estimator confidence, runtime class gate, CRA aggregation, controller, or
wheel experiment was changed. Neither candidate has general-object mass
accuracy established by this comparison.

## Released model and compatibility

This tests the original [2017 Image2Mass release](https://github.com/tstandley/image2mass),
not the VLM-based 2026 physically guided model. Audited upstream commit:
`b61506cd4f5915339ae6a69fc0bb5b04478de197`.
The shape-aware model requires RGB, a predicted thickness map, 35 geometry
features, and a density-percentile lookup. No category or language input is
required. External metric dimensions are still needed; a single RGB image
does not establish metric scale. The public wrapper accepts inches and returns
pounds; the comparison converts SI dimensions to inches and output to kilograms.

The README targets Keras 2.1.1; the mass H5 records Keras 2.0.4 and the
thickness H5 records 2.1.1. Neither TensorFlow nor Keras is installed in the
current robot environment. Instead, an independent restricted PyTorch numeric
graph evaluates the released weights offline. HDF5/JSON are read as data;
serialized Lambda byte strings are hash matched to reviewed arithmetic and
never executed. The compressed density file contains 999 ordered numeric
constants; a constrained pickle-opcode parser extracts them without unpickling.
Unknown layers, unreviewed Lambdas, degenerate geometry, and invalid predictions
are rejected rather than silently approximated.

The mass model's eight-transform median and the thickness model's eight
inverse-transform mean are retained. A geometrically equivalent axis-aligned
OpenCV rectangle tie is restored to the release's side/angle convention. With
TF32 disabled, **20 published example predictions** agreed within **0.00047%**
relative difference (gate: 0.01%). This supports the bounded inference port;
it does not prove equivalence for every input or independently validate mass.

Audited SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| `models/shape_aware.h5` | `a49f740b7de8bc27f7e88a12deb6862f504dc44bdf851c50987f48e9278b82a3` |
| `thickness_model.h5` | `1b4ea1ac10503f227ff9cea57345fc8c6f94c3eff8ecc0e38f2de5f6f2910b79` |
| `score_to_percentile.pklz` | `f7bb0de68ca1860479dbc83e15ede4ebeb272c2562b8214fa57de5d9f62b3e74` |
| Current ABO `best.pt` | `92d437580efb283071e0ac0cd7c16de77f76ef3017a243329e4bf826e0e5a31e` |

No explicit repository license was found in the inspected Image2Mass release.
The weights, table, upstream source, and example images stay in ignored local
outputs and are not included in this PR. Deployment/redistribution terms must
be clarified before adopting or distributing this candidate. No packages were
installed or upgraded: the existing base Python reads HDF5/lz4, and the existing
Torch environment runs inference.

## Matched-input results

Both neural networks receive the same RGB object content and width/depth/height;
class names appear only in the report, not in model inputs. ABO uses the original
catalog images. Isaac uses the same rendered object mask/bounds, with the
existing model's gray outside-mask fill and Image2Mass's white background.
Each model retains its native resizing/normalization. Authored/catalog
reference masses are read for scoring and never supplied to either predictor.

The primary ordering metric counts correctly ordered object pairs whose
reference masses differ by at least 2x. Repeated views are collapsed by their
median prediction before primary metrics. All 172 inputs produced predictions;
there were no Image2Mass input rejections on these sets.

| Evaluation / metric | Current MobileNet | Image2Mass |
| --- | ---: | ---: |
| ABO: 152 held-out products, 8,303 eligible pairs | | |
| Heaviness pair ordering | **95.2%** | 78.9% |
| Spearman mass rank correlation | **0.891** | 0.593 |
| MAE / median absolute error | **3.65 / 1.46 kg** | 7.13 / 3.30 kg |
| Isaac: 10 primitive proxies, 2 views each, 14 eligible object pairs | | |
| Heaviness pair ordering | **100% (14/14)** | 64.3% (9/14) |
| Spearman mass rank correlation | **0.915** | 0.236 |
| MAE / median absolute error | 0.523 / **0.159 kg** | **0.352** / 0.302 kg |
| Two-view max/min ratio: median / worst | 1.46 / 3.87 | **1.38 / 1.65** |
| Warm GPU pipeline median: ABO / Isaac | **6.3 / 6.5 ms** | 41.6 / 41.6 ms |

Latency includes model-native preprocessing and CPU/GPU transfers, with CUDA
synchronization, excluding image loading, checkpoint loading, segmentation,
LiDAR, CRA, and control. Hardware: RTX 5070 Ti; Torch 2.11.0+cu130,
NumPy 2.3.1, OpenCV 4.13.0. These are workstation measurements, not an
end-to-end robot real-time guarantee. The models' different native augmentation
counts are part of the comparison, not normalized away.

ABO dimensions and masses are catalog metadata, potentially affected by
packaging/sets. The 152 products are held out for our model; overlap with
Image2Mass's older Amazon training data cannot be excluded. Isaac RGB images
are captured stationary, but its masks, metric dimensions, and reference mass
are **oracle authored labels**. The fixtures are simplified primitive proxies,
not realistic restaurant objects. Four-factor or navigation readiness is not
inferred from these results.

Image2Mass's lower Isaac MAE does not imply better CRA ordering: its estimates
are more compressed across these proxies. The current model's view sensitivity
(one object changes by 3.87x) also remains a material limitation. Both are
uncalibrated for realistic robot images. The current runtime retains its
confidence cap, broad intervals, domain/class gate, and unknown/fallback policy;
this raw-network comparison does not remove those safeguards. Although its
runtime gate uses classes, the existing network itself is already category-free
(RGB plus dimensions). Grounded mass references and usable full-object geometry
remain necessary for a stronger restaurant-domain claim.

## Reproduction and validation

Entry points:

- `scripts/evaluation/export_image2mass.py`: audit hashes and export local H5
  numeric arrays/configuration and the numeric density table.
- `scripts/evaluation/image2mass_compat.py`: restricted offline numeric bridge.
- `scripts/evaluation/compare_image2mass.py`: published-output gate, both full
  test sets, paired ordering, independent-object metrics, view variation,
  timings, and input/export fingerprints.
- `tests/unit/test_image2mass_comparison.py`: units, augmentation alignment,
  padding, rectangle ties, rejected geometry/serialization, cropping, and
  independent-object evaluation boundaries without requiring public weights.

Local audit, exports, prediction rows, conformance examples, and report are in
`outputs/image2mass_comparison/`. The comparison consumes the existing
`outputs/visual_mass_data/abo/subset/manifest.jsonl` and
`outputs/visual_mass_isaac_fixture_v3/manifest.jsonl`; it filters test splits only.

Validation on this branch: 43 focused mass/comparison tests passed; the full
unit suite passed **1,496 tests**; repository-wide `ruff check .` passed.
The actual offline comparison completed both full test sets after its 20-example
numerical conformance gate. No new simulator or wheel-motion run was launched.

```powershell
& C:/Users/tayli/anaconda3/python.exe scripts/evaluation/export_image2mass.py `
  --source outputs/image2mass_comparison/upstream `
  --output outputs/image2mass_comparison/exports

& C:/Users/tayli/anaconda3/envs/env_60_isaacsim_yolo/python.exe `
  scripts/evaluation/compare_image2mass.py `
  --exports outputs/image2mass_comparison/exports `
  --upstream outputs/image2mass_comparison/upstream `
  --parity-manifest outputs/image2mass_comparison/parity-examples.json `
  --checkpoint outputs/visual_mass_data/abo/baseline_frozen_final_v2/best.pt `
  --abo-manifest outputs/visual_mass_data/abo/subset/manifest.jsonl `
  --fixture-manifest outputs/visual_mass_isaac_fixture_v3/manifest.jsonl `
  --output outputs/image2mass_comparison/results --device cuda
```

The next perception step remains category-free object segmentation/association
on stationary camera--LiDAR captures, followed by a small selected restaurant
object set. SAM 2 has not been integrated by this mass comparison. Keep any
mass density prior for unsupported objects explicit; do not advertise either
visual regressor as validated for every segmented object.
