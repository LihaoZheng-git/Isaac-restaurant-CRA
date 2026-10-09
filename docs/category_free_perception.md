# Category-free camera–LiDAR CRA track

## SAM instance masks (first bounded step)

`robot/perception/sam_instance_segmentation.py` refines an existing LiDAR track
with an RGB mask. The positive prompt is an observed projected return nearest
the track's image median. No class, text, color, authored mask, density or object
dimensions are supplied to SAM. Discovery and occupancy remain LiDAR-based.
The output class is `unknown`; this means category unknown, not association
unavailable. Rejected proposals remain unassessed occupancy.

The adapter reuses installed Ultralytics 8.4.104 (`SAM2Predictor`) with Meta's
official SAM 2.1 Hiera Tiny weights. No package installation or environment
upgrade was performed. Download:
<https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt>.
The local filename `sam2.1_t.pt` selects the corresponding installed builder.
SHA-256: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`;
156,008,466 bytes; 471 tensor state entries, checked before model loading.
Official [SAM 2 repository](https://github.com/facebookresearch/sam2) and
[license](https://github.com/facebookresearch/sam2/blob/main/LICENSE) identify
Apache-2.0 for the weights. The reused Ultralytics implementation retains its
own AGPL-3.0 licensing. Neither weights nor upstream source are committed here.

Configurable gates reject too few projected points, low model quality, masks
touching the image boundary, unreasonable mask area, poor LiDAR coverage,
competing-track contamination, and similarly scored non-equivalent masks.
Equivalent decoder alternatives are allowed. The default 0.75 evidence
confidence cap is a heuristic, **not** calibrated physical confidence; SAM's
predicted mask quality is recorded separately.

The simple-object demo requires explicit `--instance-mask-geometry
--sam-instance-proposal` and an existing `SAM2_CHECKPOINT`. SAM is applied to
each distinct camera image. Exact-mask return selection keeps the complement
as conservative occupancy; no mask removes an obstacle from safety processing.
The default mass class gate is unchanged in this first step.

### Stationary validation

Fresh headless Isaac capture in
`outputs/sam2_integration/segmentation_stationary/` passed stationary sensor and
CRA mapping checks, without wheel motion. SAM accepted 186 exact-mask LiDAR
returns, with model quality 0.988, encoder time 0.100 s and decoder time 0.093 s
for the recorded frame. The output remains category unknown. Raw YOLO actually
recognized this glossy ground sphere as `sports_ball` (0.867); this capture is
**not a demonstrated YOLO miss**. Its mass remained unavailable and aggregate
phi remained 1, as expected before the category-free mass step.

The visible LiDAR envelope was approximately 0.247 × 0.085 × 0.093 m for a
configured 0.36 m diameter fixture: sparse visible surface extents are not a
complete reconstruction. Fixture dimensions never enter the estimator. Target
track selection and simulator calibration/extrinsics remain explicitly bounded
demo assumptions. Timing is a recorded diagnostic, not a real-time guarantee.

Validation: 39 focused segmentation/association tests; **1,508 full unit tests**;
repository-wide Ruff passed. Unit tests cover ambiguity, boundary/area/quality
gates, competing tracks, coverage, source-image shape and checkpoint hash.

## Explicit category-free mass path

`VisualMassRuntimeEstimator.estimate_instance` is a separate opt-in API. It
requires an accepted association for the requested track, an audited SAM-source
frame within the configured age limit, a sufficiently resolved non-truncated
mask, and current measured/inferred LiDAR dimensions. The legacy `estimate`
class guard is retained even when the new path is enabled. Unaccepted instances,
stale/future evidence, non-sensor dimensions and invalid network outputs remain
unavailable. The live-mass diagnostic accepts category unknown only with the
separate accepted instance association and matching provenance.

The unchanged ABO MobileNet network consumes RGB plus metric dimensions, never
category or density. New instances are explicitly labeled
`category_free_instance_domain_extrapolation`, with the original 0.25 confidence
cap and ±4.542 kg catalog-MAE engineering allowance. This is rough heaviness
inference outside the catalog domain, not calibrated mass measurement. Complete
object dimensions may be underestimated by the visible LiDAR envelope.

The demo opt-in is `--category-free-visual-mass`; it requires SAM and excludes
the effective-density fallback. The first fresh combined stationary capture
(`outputs/sam2_integration/mass_small_stationary/`) produced **0.445 kg** with
inferred provenance and passed live RGB/LiDAR mass dataflow. All four factors
were available in prior-assisted mode: the floor continuity and harmful-fall
interaction remain priors; contact is a visible-silhouette proxy; thermal is a
validated fixed-camera low-plume proxy. Conservative phi was **0.98057**, versus
nominal diagnostic phi 0.24651, with no unknown-factor fallback. Its high phi is
an uncertainty-policy result and was not reduced for this test. No wheel motion.

The 0.30 m radius comparison (`mass_large_stationary/`) produced **1.820 kg**,
above the 0.18 m radius object's 0.445 kg, without category or density input.
This is sensible size/heaviness ordering for two rendered fixtures, not measured
mass accuracy. Its mass dataflow passed, but combined readiness **failed**:
fitted-bottom plus floor-plane uncertainty exceeded the unchanged ground
consistency gate. Fall height stayed unknown, phi stayed 1, and motion remained
blocked. A simulator process exit code alone does not establish readiness;
inspect the written acceptance report.

Mass-path validation: 30 focused tests, **1,526 full unit tests**, and
repository-wide Ruff passed. No new training, density input, package install or
dependency upgrade was needed. This does not establish general-object accuracy.

## Next scenario step

The bounded category-free track is complete through physical pass-by validation.
Next, select and enrich restaurant props/actors, validate each stationary, then
run matched restaurant baseline experiments. Moving-camera plume estimation and
continuous field updates remain separate work; these endpoint trials freeze
stationary perception during motion.

This track does not claim universal segmentation, general mass accuracy,
physical sharpness, temperature measurement or universal navigation safety.

## Stationary combined benchmark

The session-only `--object-appearance matte` fixture control is not an estimator
input. SAM records overlap with the preceding raw YOLO masks, so recognized
objects and actual mask misses can be distinguished rather than inferred from
the output category `unknown`.

Three fresh matte captures passed all four factors in **prior-assisted** mode:

| Fixture | Raw YOLO target mask | Rough neural mass | Fall height | Computed phi |
| --- | --- | ---: | --- | ---: |
| Ground, radius 0.18 m | recognized sports ball | 0.366 kg, inferred | 0 m, ground-continuity prior | 0.98071 |
| Raised, same radius | missed; only chair masks | 0.555 kg, inferred | 0.301 m, inferred support-to-landing drop | 0.98385 |
| Ground plus persistent rendered plume | recognized sphere | 0.348 kg, inferred | 0 m, ground-continuity prior | 0.98955 |

The raised capture uses an explicit **2.5 m landing search limit**; 2.0 m failed
to localize a lower landing band and left potential unknown. The farther band
is candidate landing evidence, not an exact impact site. Harmful-fall interaction
still uses a worst-case prior, so even this inferred drop is not a fully
sensor-only four-factor trial. Contact/thermal remain named visual proxies.
Mass and other factors vary with the different observed views, so this is not
a potential-only causal comparison or mass calibration. All unmatched support
and environmental returns remain conservative unknown-risk occupancy.

The positive plume capture produced continuous visible-plume cue **0.451**.
Its nominal thermal factor rose from 0.0131 to **0.4708** and conservative
thermal risk from 0.4572 to **0.7089**. The rendered puffs are stylized RGB
fixtures, not temperature or simulated steam physics. No thermal score or
plume-control label enters estimation. This checks positive cue sensitivity
with SAM associations, while retaining the fixed-camera/reference limitation.

The 0.25 m matte and 0.30 m glossy ground captures produced mass estimates but
failed complete readiness at the unchanged fitted-bottom uncertainty gate.
These rejection records remain in the diagnostic set; no zero fall height was
substituted. The accepted scope is the tested fixtures, not every object size.

`scripts/evaluation/cra_category_free_report.py` reads the actual capture,
live-mass and SAM reports, fingerprints them, checks unchanged phi/fixed lambda,
reports missing factors and produces a factor-specific chart. Unknown factors
have no nominal bar; their separate conservative fallback remains visible.
It does not supply or equalize any factor score. Actual factor-input provenance
now selects runtime readiness mode, fixing the earlier ground-flag-only label.

Local accepted artifacts: `unknown_matte_ground_stationary/` and
`unknown_matte_raised_valid_stationary/`, plus `unknown_matte_plume_stationary/`
under `outputs/sam2_integration/`;
comparison `four-factor-summary.json` and `.png`; rejected/accepted diagnostic
set `all-stationary-diagnostics.json`. No wheel motion in these captures.
Validation: **80 focused tests**, **1,545 full unit tests**, repository-wide Ruff.

## Physical wheel acceptance

Each wheel trial recaptured RGB/LiDAR and passed four-factor readiness before
motion. It used the computed phi unchanged, fixed lambda **1.25**, the straight
obstacle-unaware waypoint nominal, online unicycle MPC and analytical CBF
rechecks, and physical Nova Carter wheel commands. No controller objective,
uncertainty policy, mask gate, fall-height gate or tolerance was weakened.

| Fresh trial | Computed phi | Waypoint time | Minimum independent footprint clearance | Stop |
| --- | ---: | ---: | ---: | --- |
| Ground matte sphere | 0.98070 | 26.2 s | 0.15301 m | passed |
| Raised matte sphere, actual YOLO miss | 0.98389 | 34.0 s | 0.12399 m, includes platform | passed |
| Ground plus visible plume | 0.98965 | 25.2 s | 0.15422 m | passed |

All three reached the waypoint, retained positive MPC/CBF margins and preserved
the saved restaurant scene. MPC intervened 116/161/111 times respectively;
CBF remained active for rechecks but needed zero additional interventions.
Mean/max MPC wall times were 0.196/0.577, 0.228/1.007 and 0.207/0.608 s against
a 0.20 s planning interval. These are **not real-time deadline results**.

The ground mass estimates were 0.354 and 0.343 kg; raised mass was 0.527 kg and
inferred support-to-landing drop 0.30125 m. Frame/view variation and broad
engineering mass bounds remain explicit. Ground continuity and harmful-fall
opportunity are priors. No true mass, physical sharpness or temperature labels
were available. The high phi values mainly reflect conservative uncertainty;
the raw evidence and factor-specific scores still vary continuously.

Support/environment occupancy remains conservative phi 1 and can dominate the
raised planar field. Different fixture geometry, fresh evidence and solver
trajectories confound causal risk-to-clearance claims. These runs establish the
bounded perception-to-policy-to-controller connection, not a CRA advantage,
monotonic clearance ordering, universal safety or exact paper equivalence.
The control uses the existing planar/unicycle adaptation, with no moving-camera
perception or dynamic field updates during execution.

Artifacts: `outputs/sam2_integration/unknown_matte_{ground,raised,plume}_motion/`
contain complete `report.json`, masks, camera/fusion overlays and `paths.png`.
`physical-summary.json` fingerprints actual reports and requires physical route
completion, nonzero command/finite trajectory records, safety margins and safe
stopping. `--require-motion` rejects a stationary-only report with exit status 1;
the three actual wheel reports passed with exit status 0. Final validation:
**47 focused tests**, **1,556 full unit tests**, repository-wide Ruff.

### Reproduction

Use the existing Isaac/vision import bridge and existing local weights. Example
from the repository root (process-only environment; no installation):

```powershell
$env:ISAAC_SIM_HEADLESS = '1'
$env:CRA_VISION_SITE_PACKAGES = 'C:\Users\tayli\anaconda3\envs\env_60_isaacsim_yolo\Lib\site-packages'
$env:YOLO_MODEL_PATH = 'C:\myworkplace\models\ultralytics\yolo11n-seg.pt'
$env:VISUAL_MASS_CHECKPOINT = 'outputs/visual_mass_data/abo/baseline_frozen_final_v2/best.pt'
$env:CONTACT_SEGMENTATION_CHECKPOINT = 'runs/segment/outputs/contact_perception/training/yolo11n_seg_paco_geometry/weights/best.pt'
$env:SAM2_CHECKPOINT = 'outputs/sam2_integration/sam2.1_t.pt'
& C:/isaacsim/python.bat -u -c "import sys,os,runpy; sys.path.insert(0, os.environ['CRA_VISION_SITE_PACKAGES']); target='scripts/demos/cra_simple_object_isaac_demo.py'; sys.argv=[target,*sys.argv[1:]]; runpy.run_path(target,run_name='__main__')" `
  --output outputs/sam2_integration/reproduction-ground-stationary `
  --object-radius 0.18 --object-appearance matte --instance-mask-geometry `
  --sam-instance-proposal --category-free-visual-mass --object-plume-policy `
  --ground-continuity-prior --worst-case-fall-interaction-prior `
  --require-four-factor-readiness
```

For raised, remove `--ground-continuity-prior` and add `--support-drop-m 0.30
--fall-landing-maximum-offset-m 2.5`. For plume, add `--plume-control persistent`.
Use fresh output directories. Validate stationary first; then add `--motion
--motion-maximum-seconds 40` in another directory, which repeats the gate before
commands. Check report acceptance, because Isaac shutdown can mask exceptions.

```powershell
& C:/Users/tayli/anaconda3/envs/env_60_isaacsim_yolo/python.exe `
  scripts/evaluation/cra_category_free_report.py --require-motion `
  --case ground=outputs/sam2_integration/unknown_matte_ground_motion `
  --case raised=outputs/sam2_integration/unknown_matte_raised_motion `
  --case plume=outputs/sam2_integration/unknown_matte_plume_motion `
  --output outputs/sam2_integration/physical-summary.json
```
