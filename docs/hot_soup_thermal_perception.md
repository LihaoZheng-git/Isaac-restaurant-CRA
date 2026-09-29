# Contextual thermal cues and the hot-soup scenario

Hot soup is one restaurant evaluation scenario, not a runtime class. The
general runtime contract requires three fresh visual inputs: plume likelihood,
source-context likelihood, and plume-to-source spatial association. The
restaurant adapter supplies an open-container cue as the source context and the
likelihood that the plume lies above it. The contract uses their minimum as a
continuous inferred heat-source proxy, so a strong generic plume detection
cannot hide missing or weak source context.

The output is not a hot/not-hot decision, temperature, burn severity, fluid
temperature, or proof that the plume is steam. A detector labeled `smoke_like`
or `visible_plume` remains unavailable in the steam field. Only a fixed-label
steam detector can populate that field, and even that remains a visual cue
rather than a thermometer.
Thermal interaction with the robot route is a separate input to the physical
factor adapter. Without it, the CRA thermal factor remains unknown.

## Candidate-model audit

[D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset) supplies fire
and smoke boxes, not steam, soup, open-container, spatial, temperature, or spill
labels. [SAINet](https://github.com/AlterMundi/sai-sainet-latest) publishes an
AGPL-3.0 wildfire checkpoint, but its data preparation explicitly treats steam
as a hard negative. It is therefore not a justified direct hot-soup steam
detector. Neither model has been downloaded or claimed as integrated on this
branch.

[Kitchen Safety System](https://universe.roboflow.com/diploma-2023/kitchen-safety-system)
is a closer visual-domain candidate. Its public Roboflow page reports a CC BY
4.0 object-detection dataset with 3,998 images and classes that include
`Steam`, `open_pot`, and `open_pot_boiling`. The hosted version-4 model reports
aggregate mAP50, precision, and recall, but no per-class hot-soup result. These
aggregate numbers are not evidence for the project's two-label contract.

The public browser also exposes a split-validity problem. Frames named
`höyry1_mp4-*.jpg` from one apparent source video occur in train, validation,
and test (for example, frames 227, 224, and 247 respectively). A checked Steam
frame (`höyry1_mp4-334.jpg`) contains one `Steam` annotation and no
open-container annotation. The published splits therefore cannot serve as the
project's held-out evaluation, and the labels do not directly satisfy the
required plume-plus-container association. If this dataset is used later, it
may only seed training after source-group reconstruction, annotation audit,
container relabeling, and confirmation that the uploader had rights to license
the source imagery. Project-owned capture groups must remain the independent
validation and test evidence.

Dataset download requires a Roboflow account and acceptance of Roboflow's
terms. Hosted inference likewise requires an API key. No account was created,
no terms were accepted, no data or weights were downloaded, and no hosted
service was added to the runtime. This avoids making network availability or a
third-party credential part of the robot safety path.

[The Isaac Sim 5.0 Replicator Incident documentation](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/action_and_event_data_generation/tutorial_replicator_incident.html)
describes fire and smoke event generation. The installed Isaac Sim 6.0
environment does not contain the `isaacsim.replicator.incident` extension, so
the stationary diagnostic authors a small USD visual proxy: translucent plume
geometry above a bowl. The capture step renders RGB and Replicator tight 2D
boxes. Boxes from the authored plume and container prims are unioned per class
and recorded as `isaac_authored_oracle`. This tests the synthetic data path,
not aerosol physics or detector accuracy.

## Validation boundary

`hot_soup_cues_demo.py` uses synthetic prior fixtures and verifies the context
gate plus the separate CRA interaction requirement. The Isaac demo exports a
stationary visual plume scene and captures RGB plus oracle boxes without wheel
commands. A successful run writes the image and manifest under the ignored
`outputs/hot_soup_steam_isaac/dataset` directory. By default it renders the
deterministic `baseline`, `wispy-right`, and `broad-left` appearance variants;
set `HOT_SOUP_VARIANTS` to a comma-separated subset when debugging one scene.
All variants share a capture group so dataset splitting cannot leak near-duplicate
synthetic views across train and evaluation. The run fails rather than
fabricating either required class when Replicator returns no visible box. On
the current Isaac Sim 6.0 host, the visible renderer produced a 640 by 480
frame with both classes, while three headless attempts returned an empty RGB
buffer. The script therefore defaults to the validated visible stationary mode;
`ISAAC_SIM_HEADLESS=1` remains available for retesting this environment issue.
Actual perception still needs
a steam/cooking-vapor dataset, restaurant-domain negative examples, trained
weights with a compatible license, an open-container detector, and spatial
association validation on real and Isaac images.

## Project-owned dataset path

`collect_hot_soup_data.py` ingests project-owned stationary images or samples
frames from a project-owned video. Every capture records its source reference,
usage rights, domain, content hash, and capture group. Collected frames remain
`pending`; an unlabeled frame is never treated as a negative example.

The annotation contract in `robot/perception/hot_soup_dataset.py` requires
explicit present/absent decisions for both `visible_hot_soup_plume` and
`open_container`. Real labels are human visual annotations. Isaac labels remain
`isaac_authored_oracle`. Hard-negative tags cover smoke, glare, haze, white
backgrounds, and empty containers. They supervise visual discrimination only;
they do not say that an object is thermally safe.

Capture groups receive deterministic train/validation/test splits so adjacent
video frames cannot cross splits. Validation also rejects the same image hash
across splits. `prepare_hot_soup_detection_dataset.py` exports validated boxes
and explicit empty negative labels to an Ultralytics-compatible detection
layout. No checkpoint is trained or integrated by this dataset branch. Before
training, the project still needs owned real hot-soup footage, completed human
annotations, a documented train/evaluation protocol, and a final license check
for the selected pretrained backbone and released derivative weights.

## Fixed-label runtime boundary

`robot/perception/hot_soup_detector.py` accepts detector boxes only for
`visible_hot_soup_plume` and `open_container`. It computes a configurable
continuous spatial association from horizontal overlap, relative vertical
position, and vertical gap, then invokes the general thermal cue contract.
Partial overlap or weak vertical evidence receives a correspondingly weak score
instead of failing a geometry threshold. Missing detections remain
unavailable because detector non-response is not proof that a hazard is absent.
An observed plume/container pair with incompatible geometry can provide a zero
association score. Detector confidence is recorded as an uncalibrated model
score, not temperature or a physical probability.

The optional YOLO boundary does not bundle or download a checkpoint. The local
`yolo11n-seg.pt` file has SHA-256
`55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`,
but its acquisition provenance was not recorded and its COCO classes do not
include the two project labels. It is therefore not a hot-soup model and should
not be redistributed from this repository. Ultralytics states that its
open-source software and models use AGPL-3.0, with a separate enterprise
license for proprietary use ([official licensing documentation](https://docs.ultralytics.com/)).
This thesis repository can evaluate that technique under its applicable terms,
but trained project weights still require documented base-weight provenance,
owned labeled footage, and a release decision consistent with those terms.

## Track association boundary

The selected open-container box can be assigned to an existing LiDAR track
through the track's projected camera median. Exactly one projected track must
fall inside the box. Zero matches stay unassigned, and multiple matches are
reported as ambiguous. Frame ID, timestamp, and image dimensions must agree;
freshness is checked again when the per-track thermal provider is queried.

## Training and held-out evaluation

`assess_hot_soup_training_readiness` now blocks training unless the dataset is
internally valid and contains group-separated, project-owned real positives
and negatives in train, validation, and test splits, plus real evaluation hard
negatives. Isaac synthetic samples are counted and may supplement training,
but they do not satisfy the real-data minimums. These defaults are a functional
pipeline gate, not a claim that the dataset is publication-scale.

`scripts/training/train_hot_soup_detector.py` exports the validated two-class
dataset and trains a local Ultralytics checkpoint without downloading or
bundling weights. The resulting provenance report records manifest, base
checkpoint, trained checkpoint, package version, reported package license, and
training arguments with hashes. A blocked dataset cannot produce a trained
provenance report.

Held-out evaluation reports plume and open-container precision, recall, and F1
separately at configurable confidence and IoU thresholds. It also counts real
hard-negative images that produce plume false positives. These metrics assess
fixed-label visual detection only; they do not measure temperature, prove
steam, or validate CRA thermal severity.
