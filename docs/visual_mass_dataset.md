# Visual mass dataset decision

The general-object path collects RGB and object masks in controlled Isaac Sim
scenes with mass labels from an authored USD mass, a documented density/volume
calculation, or a physical scale. These are oracle training/evaluation labels;
they must never enter runtime CRA inference.

Splits use physical `object_id`, preventing multiple views of one asset from
leaking across train, validation, and test. The manifest records scene, camera,
seed, class, dimensions, mass, and mass source. Collection stores RGB and masks
losslessly. Before training, inventory multiple independent instances per
class—not merely views of one mesh—and randomize viewpoint, lighting, texture,
background, and container fill state.

No external mass model is integrated. Image2Mass has legacy Keras requirements
and no verified repository license. CORSMAL is container-specific; some related
components are noncommercial. Architecture selection waits until the dataset
inventory demonstrates enough independent objects. Real-image evaluation is
required before simulator-trained output can be called runtime evidence.

## ABO restaurant subset

The bounded real-image path uses the official Amazon Berkeley Objects (ABO)
product listings and catalog images. The current ABO download page and archive
README identify the release as CC BY 4.0. Preserve the archive license and give
the required Amazon.com and dataset-author attribution in publications and
derived-dataset documentation.

Run `scripts/data/prepare_abo_visual_mass_subset.py` against the 16 official
`listings_*.json.gz` shards and `images/metadata/images.csv.gz`. The selector:

- uses exact product-type mappings for restaurant-relevant classes;
- requires a positive catalog weight, all three catalog dimensions, and an
  indexed main image;
- converts normalized pounds and inches to kilograms and metres;
- caps each class before training and assigns splits by product object ID; and
- records the source license and provenance in every manifest row.

ABO values are external catalog metadata claims. They are offline supervision,
not scale measurements made by this project, and they remain unavailable to
the CRA runtime. The selector does not turn object height into fall height.
Raw archives, selected images, and generated manifests live under `outputs/`
and are intentionally excluded from Git.

Example:

```powershell
python scripts/data/prepare_abo_visual_mass_subset.py `
  --listings-dir outputs/visual_mass_data/abo/raw/listings/metadata `
  --image-metadata outputs/visual_mass_data/abo/raw/images.csv.gz `
  --output-dir outputs/visual_mass_data/abo/subset `
  --max-per-class 200
```

The data-preparation step does not itself provide trained weights or claim
general-object mass inference. Architecture selection follows the inventory,
image-availability, pretrained-weight, and validation checks below.

## Lightweight baseline

The first bounded model is MobileNetV3 Small image features plus three positive
SI dimensions, with a head that predicts log mass. The dataset size affects
offline download and training; runtime cost depends on the fixed 2.54-million-
parameter image encoder and small regression head. The training script compares
the RGB-plus-dimensions model with a global training-median baseline and a
log-dimensions least-squares baseline.

The official Torchvision model page identifies the ImageNet-1K weights as
`MobileNet_V3_Small_Weights.IMAGENET1K_V1` (9.8 MB). Torchvision source is
BSD-3-Clause, but the installed weight enum has no explicit license metadata.
Training reports therefore record the exact URL, SHA-256, size, upstream
training dataset, and this unresolved weight-license field. Do not redistribute
the checkpoint or call it deployment-ready without resolving that point.

The local frozen-backbone diagnostic on the deduplicated 1,003-object subset
selected epoch 12 of 12. On 152 held-out catalog objects it obtained 3.650 kg
MAE and 102.0% MAPE, compared with 4.398 kg and 132.9% for log dimensions alone
and 7.322 kg and 5305.6% for a global median. RGB therefore adds catalog-domain
signal, while the large relative error and sparse bottle/knife samples prevent
general-object or runtime claims. Fine-tuning all layers produced similar MAE
(3.641 kg) but much worse MAPE (329.3%), so it is retained only as an ablation.
Catalog item weights may also describe packaging or multi-item sets rather than
the single visible object, and this label-quality risk needs a manual audit.
The stationary single-image model-forward diagnostic measured median latency of
4.00 ms on the local RTX 5070 Ti and 4.49 ms on the local CPU. These timings
exclude detection, cropping, geometry estimation, and camera transfer.

Checkpoints and evaluation reports stay under `outputs/`. Run
`scripts/demos/visual_mass_model_diagnostic.py` for a stationary, offline
inference and latency check. Isaac camera crops, live LiDAR dimensions, domain
shift, calibration, and uncertainty rejection remain required before a model
output can become inferred runtime evidence.

## Bounded fused-track runtime adapter

`VisualMassRuntimeEstimator` now accepts a fused instance mask, its RGB crop,
and live LiDAR width/depth/height. It supports only the nine classes represented
in the ABO subset. Unsupported classes and missing dimensions remain
unavailable. The crop background is masked before the existing MobileNet
forward pass; no catalog metadata or language reasoning enters inference.

The adapter uses a configurable absolute-error bound (default 4.542 kg, the
held-out catalog validation MAE) and caps confidence at 0.25 for unvalidated
ABO-to-Isaac domain transfer. These are conservative engineering settings, not
calibrated uncertainty. The checkpoint hash, catalog error, sensor inputs, and
domain limitation are retained in evidence. Configure
`VISUAL_MASS_CHECKPOINT` when running the stationary camera/LiDAR fusion demo;
if it is absent, mass stays unavailable. A successful Isaac forward pass is
still a sensor-pipeline diagnostic, not proof of mass accuracy.

The first stationary Isaac run used checkpoint SHA-256
`92d437580efb283071e0ac0cd7c16de77f76ef3017a243329e4bf826e0e5a31e`.
The live camera detected a chair, calibrated projection associated its mask to
LiDAR track T3, and LiDAR inferred dimensions 0.229 x 0.292 x 0.426 m. The
adapter returned 8.522 kg with configured bounds [3.980, 13.064] kg and
confidence 0.25. Robot drift was 0.000136 m and 0.0004 degrees. No simulator
mass or prim label entered the estimate, and no wheel command was issued. Since
the scene object's reference mass was not read, this result validates data flow
and provenance only; it provides no Isaac-domain accuracy measurement.

## Isaac-domain offline evaluation

`scripts/evaluation/evaluate_visual_mass_isaac.py` runs the existing runtime
adapter on persisted test RGB arrays, instance masks, and captured dimensions.
It constructs every prediction before reading the sample's authored mass for
offline scoring. Oracle mass therefore cannot enter the runtime estimate.

Repeated views are averaged per physical `object_id` before MAE, median error,
MAPE, and interval coverage are computed. This prevents many camera views of
one asset from inflating the independent-object count. Per-class results have a
separate minimum object gate, and one report cannot mix checkpoint hashes.
The default minimums of ten independent test objects and two per reported class
are functional evaluation gates, not a publication-scale sufficiency claim.
Until a persisted Isaac test inventory satisfies them, visual mass remains an
uncalibrated inferred input with its existing confidence cap.
