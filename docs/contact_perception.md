# Contact perception status

The implemented adapter consumes a fixed-label dangerous-part mask and reports
the fraction of object pixels assigned to the dangerous part. This is inferred
visual exposure. It is not sharpness, hardness, physical exposed area, contact
probability, or severity.

Sharpness, material hardness, and orientation toward the route require separate
specialists with their own confidence and provenance. Missing specialists stay
unavailable. No pretrained contact model or weights are integrated yet; model,
dataset, license, and restaurant-domain suitability still require verification.

## PACO specialist audit

PACO is the strongest identified fixed-label source for this step. Its official
release covers 75 objects and 456 object-part categories, including `knife` and
`knife:blade`, and publishes part masks across COCO/LVIS images. The official
PACO code is MIT-licensed and the PACO-LVIS annotation archive has publisher
SHA-256 `02ac4edb22c251e07853e6231d69aec3fad0a180f03de2f8c880650322debc80`.

The repository does not explicitly state a license for the added PACO
annotations. Underlying COCO images retain per-image licenses. This distinction
blocks a claim that PACO-trained weights are cleared for redistribution until
the annotation terms are resolved.

The official PACO model zoo reports part-mask AP, but its R50/R101/ViT FPN
models require an older Python 3.9, PyTorch 1.10, Torchvision 0.11, and
Detectron2 stack. They are candidate reference teachers/evaluation baselines,
not compatible lightweight runtime integrations for the current environment.

`scripts/data/inventory_paco_contact_labels.py` reads only the small category
arrays from the large JSON files. It records actual knife-blade image/mask
counts before any COCO image download or training. The verified PACO-LVIS v1
archive contains 1,653 train images/2,528 knife-blade masks, 82 validation
images/120 masks, and 332 test images/540 masks. Even a successful blade mask
would support dangerous-part exposure only. It would not measure edge radius,
sharpness, material hardness, or contact severity, so those fields remain
unavailable unless independently supported.

## General contact geometry scope

Contact risk is not restricted to restaurant utensils. The target visual
geometry taxonomy is `exposed_edge`, `pointed_tip`, `protrusion`, and
`jagged_surface`, which can cover hazards such as nails, screws, spikes, broken
glass, exposed metal, and sharp structural corners. Geometry class and mask are
still only inferred visual evidence; a class name does not establish physical
sharpness, material hardness, route orientation, contact probability, or injury
severity.

`robot/perception/contact_geometry_features.py` now extracts three continuous
descriptors from any accepted mask in this taxonomy: dangerous-part exposure,
principal-axis elongation, and isoperimetric boundary complexity. These make
the future contact specialist less dependent on utensil names and can represent
long edges, pointed or protruding parts, and irregular broken surfaces. They do
not populate CRA sharpness, hardness, or route orientation. Those physical
fields remain unavailable until separately calibrated 3D/material specialists
exist. Current runtime enablement still defaults to `exposed_edge`; synthetic
tests of the other geometry classes are oracle plumbing evidence only.

`scripts/data/build_paco_hazard_manifest.py` streams the large annotation file
with bounded memory and maps `knife:blade` and `scissors:blade` to
`exposed_edge`, plus `screwdriver:tip` to `pointed_tip`. The generated manifest
retains source labels, masks, parent-object masks, image metadata, dataset hash,
and license status. PACO does not provide the required general protrusion or
jagged-surface coverage, so those classes remain unavailable pending controlled
synthetic or real supervision. PACO geometry labels also do not fill sharpness
or hardness fields in the CRA factor.

The verified manifests contain 3,528 geometry masks in 2,226 train images, 174
masks in 108 validation images, and 794 masks in 448 test images. Only one
validation screwdriver-tip mask is present, so `pointed_tip` performance cannot
be treated as established; the current data chiefly supports an exposed-edge
baseline.

`scripts/data/prepare_paco_contact_segmentation.py` converts polygon and
compressed COCO RLE masks into normalized YOLO segmentation labels without
requiring `pycocotools`. On the verified manifests it writes 3,881 train
polygons from 2,224 images, 204 validation polygons from 108 images, and 877
test polygons from 446 images. Seventeen records with no usable exterior
contour are reported as skipped rather than silently used. The image downloader
retains only the selected COCO images and writes a failure report.

`scripts/training/train_contact_segmentation.py` fine-tunes a local lightweight
Ultralytics segmentation checkpoint with deterministic settings. The resulting
checkpoint is a PACO geometry baseline, stays under ignored local outputs, and
must be evaluated before any runtime-integration claim. Its unsupervised output
classes are not evidence that protrusion or jagged-surface perception exists.

The local baseline used Ultralytics 8.4.104 (package metadata: AGPL-3.0) and the
existing `yolo11n-seg.pt` checkpoint with SHA-256
`55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`.
No package or environment version was changed. The 30-epoch checkpoint has
SHA-256 `4c5b7b6b60067c0339d21694c861db5dac6f41457f9075bd937dfefe61dc8949`
and remains in ignored local output because PACO annotation redistribution terms
are not explicit and the Ultralytics license must be respected.

On the held-out PACO test split, exposed-edge mask mAP50 is 0.358 and mask
mAP50-95 is 0.194 across 845 instances. The model scores zero on 32
screwdriver-tip instances, so pointed-tip perception is not supported. A
30-repeat fixed-frame diagnostic produced identical masks and confidences each
time, with 11.61 ms median and 12.05 ms p95 wall time on the RTX 5070 Ti. These
results establish a reproducible blade-edge baseline only; they do not justify
runtime CRA integration or a general-contact claim.

## Track association boundary

`robot/perception/specialist_track_evidence.py` transfers a fixed-label contact
segmentation to an existing LiDAR track only when its detection index was
matched by the same timestamped camera association frame. Stale and unmatched
results remain unavailable. This plumbing does not improve the trained
baseline, infer physical sharpness or hardness, or turn its authored diagnostic
fixtures into sensor-derived evidence.

`robot/perception/contact_detection_adapter.py` adds the fixed-label runtime
boundary needed before that transfer. It requires the part-model frame and the
parent-object segmentation frame to identify the same camera capture, then
associates a part only when most of its pixels lie inside one unambiguous
object mask. The supported runtime vocabulary defaults to `exposed_edge`
because that is the only class with nonzero held-out performance in the local
baseline. A missing model detection creates no segmentation record and is not
converted to zero exposure. Pointed tips, protrusions, jagged surfaces,
sharpness, and hardness remain unsupported.

## Stationary live invocation

The camera--LiDAR fusion demo can load the evaluated local exposed-edge
checkpoint through `CONTACT_SEGMENTATION_CHECKPOINT`. It invokes the specialist
on the same stationary Hawk frame as parent-object segmentation, associates
only contained part masks to accepted object detections, transfers them only to
accepted LiDAR tracks, and writes `contact_diagnostic.json`. A model
non-response stays `unavailable`; it is not recorded as zero exposure.

In the first restaurant run, the specialist completed in 63.6 ms and returned
no exposed-edge detections in the chair view. The dataflow diagnostic passed,
while contact evidence and the complete contact CRA factor stayed unavailable.
The robot drifted 0.001188 m and 0.0228 degrees, published no nonzero motion,
and received no wheel command. This negative stationary invocation does not
validate knife, nail, protrusion, or general outdoor hazard performance. A
positive live track-level Isaac example remains required, as do independent
sharpness, hardness, and route-orientation methods.

## Positive held-out replay

`scripts/demos/contact_positive_replay_demo.py` provides a reproducible
positive test without relabeling it as live Isaac evidence. It runs the generic
object segmenter and exposed-edge specialist on the same held-out PACO image,
applies the conservative mask-containment association, records all three input
hashes, and saves object and part overlays plus `diagnostic.json`.

For held-out image `000000040248.jpg`, the generic checkpoint produced 22
object instances and the contact checkpoint produced 13 exposed-edge masks.
Four masks were accepted for three parent instances, all classified as knives;
six ambiguous masks and three unmatched masks were rejected. This validates a
positive visual part-to-object path on existing held-out imagery. It is not a
live Hawk/LiDAR observation, does not add an Isaac-domain result, and does not
infer sharpness, hardness, contact probability, nails, tips, protrusions, or
jagged surfaces.
