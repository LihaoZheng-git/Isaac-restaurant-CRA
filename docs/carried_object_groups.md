# Human and carried-object CRA grouping

The user subsequently approved a separate low nonzero human-body contact prior.
See [human_body_contact.md](human_body_contact.md) for its explicit opt-in and
new component results. The outline-based results below, including group phi 1.0,
remain the original legacy-model experiment and are not relabeled.

This opt-in component estimates a person and a compact carried object separately,
then applies `max(phi_person, phi_object)` over both members' observed occupancy
when their carrying relationship passes the visual/metric association gates.
Other obstacles keep their own policies. A group does not add masses, combine a
person's mass with an object's elevation, or represent a cumulative probability.
The waypoint nominal, online MPC, analytical CBF and fixed lambda 1.25 are
unchanged. New robot-wheel acceptance is separate from this parked component.

## Inputs and provenance

`robot/perception/human_pose.py` uses an audited local YOLO11n-pose checkpoint to
obtain COCO wrist/elbow/body keypoints from the actual RGB frame. The matching
person mask comes from the existing COCO segmentation model. One unambiguous
person and source-aligned RGB/LiDAR inputs are required. Simulator actor names,
joint states, hand transforms, authored prop masks and temperatures are never
perception inputs. The current ego pose/extrinsics remain oracle references;
RGB acquisition retains the documented simulation read-time approximation.

`robot/perception/carried_object.py` crops observed person context at native
resolution and prompts the existing SAM 2.1 Tiny with actual LiDAR returns. A
lower-body prompt avoids treating the bowl as a torso when the scanner cannot
see shoulders. An explicit two-wrist occluding-subinstance mode handles the
COCO person mask including the carried object. It requires compact geometry
near both wrists and a bounded foreground depth partition; person-mask overlap
is recorded rather than represented as passing the legacy disjoint-mask gate.
The separate one-wrist mode retains its person-overlap rejection. SAM quality,
boundary, coverage, foreign-return and ambiguity checks remain in force.
Selecting two-wrist mode disables implicit fallback to the broader one-wrist
scene-component method. A rejected two-wrist proposal remains unresolved.
Rejected or unowned points remain physical occupancy.

Independent trackers use each accepted member's positive LiDAR points. A
relationship requires at least three fresh observations spanning .4 s, gaps at
most .35 s, at least .12 m translation by each member, at most .15 m change in
member separation and .3 m/s net relative centroid displacement divided by the
observation span. Separation is rotation invariant, so a turn need not keep a
fixed world-frame offset vector. The full vector drift is also recorded. The
wrist-depth proxy supplies proximity context, not a stable 3D joint or motion
reference; that attempted reference was rejected after actual sensor jitter.
This is an inferred carrying relation, not a measured grasp. Stationary
proximity alone cannot confirm it. Identity/history may survive a gap up to
.35 s, but a missing object publishes no member assessment or group overlay;
fresh reacquisition must pass the existing .35 m tracker association. Source
gaps and failed association never renew old evidence. Component thresholds are engineering assumptions,
not calibrated probabilities or general crowded-scene guarantees.

`robot/evaluation/carried_object_runtime.py` runs each member through the existing
physical adapters. Human mass remains the explicitly configured adult prior
(default 70 kg, bounds 45–100); object mass uses its own category-free RGB/LiDAR
rough-mass inference. Motion is estimated separately. Contact remains the
visible geometry proxy where accepted. A carried object's center or hand height
is not fall height. Missing support, fall interaction or plume evidence uses
the existing labeled factor priors, not zero. Four numeric scores do not imply
four measured factors. There is no new temperature estimator or heat claim.

The maximum-member policy is an overlay in the Safe-SAGE field builder;
individual scores and their provenance stay intact. A complete field transaction
must succeed before member trackers and relation histories advance. Occupancy
diagnostics label which member supplied the group policy. Current primary-result
phi remains the individual primary assessment; the actual shared policy is in
`carried_object_groups` and `occupancy_policies`.

## Reused model audit

The newly reused checkpoint is the official
[YOLO11n-pose v8.3.0 asset](https://github.com/ultralytics/assets/releases/tag/v8.3.0),
6,255,593 bytes, SHA256
`869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`.
It supplies [COCO 17-keypoint pose outputs](https://docs.ultralytics.com/tasks/pose/)
and is [COCO pretrained](https://docs.ultralytics.com/models/yolo11/).
[Ultralytics licensing](https://www.ultralytics.com/license) offers AGPL-3.0 or
Enterprise terms; this is not a license-free model. The adapter requires the
audited existing local file and never downloads weights automatically. No new
dataset, retraining, package installation or environment upgrade was performed.
Compatibility was exercised in the existing Ultralytics 8.4.104 / Torch
2.11.0+cu130 / Torchvision 0.26.0+cu130 environment. Pose keypoint confidence and
SAM model quality are separate from physical evidence confidence.

Audit outputs: `outputs/carried_object_model_audit/probe.json` and each diagnostic's
`carried-pose-model-audit.json`. Existing person, SAM, contact and ABO mass
checkpoints retain their earlier audits and limitations.

## Validation and reproduction

The zero-wheel command is:

```powershell
$env:PYTHONEXE='"C:/Users/tayli/anaconda3/envs/env_60_isaacsim_yolo/python.exe"'
& C:/isaacsim/python.bat -u scripts/demos/restaurant_human_locomotion_isaac.py `
  --scene scenes/generated/restaurant_service_isaac_server_v2.usda `
  --output outputs/restaurant_service/NEW-UNUSED-DIRECTORY `
  --layout-survey outputs/restaurant_service/layout-monitor-v1/geometry-survey.json `
  --sensor-diagnostic --carry-bowl --carried-object-groups `
  --continuous-crossing --continuous-upper-continuation --carry-facing-crossing --speed .5 `
  --human-cluster-radius .18 --robot-view carried-diagonal `
  --camera-upward-degrees 15 --sensor-simulation-rate .05
```

`carried-object-validation.json` independently checks fresh separate ownership,
distinct human/object mass sources, four numeric factors, actual bowl geometry,
moving relation confirmation, the maximum policy on both geometries and .35 s
completion age. Authored bowl truth is used only by this validator. Actor motion,
native attachment and this new perception gate must all pass. The Isaac launcher
can return zero after a script failure; inspect the saved flags and error log.
The diagonal fixture parks at world `(26.5,7.4)`, yaw `atan2(1.5,2.8)`, and
explicitly starts the actor facing its crossing direction. These select a
view and authored actor behavior only; neither supplies estimator geometry,
ownership or risk. The earlier side view is separately reproducible with
`--robot-view crossing-approach` and without `--carry-facing-crossing`.

Historical failures are retained:

- `carried-groups-parked-v1`: 168 fields, only two separate-object rows and no
  confirmed groups. Its original aggregate report predates the group gate;
  `carried-object-validation-posthoc.json` fails the new gate. Do not call its
  actor/body/attachment pass a successful grouping experiment.
- `carried-groups-parked-v2`: 29 separate rows but no confirmed groups; centroid
  relative motion rejected the turn, and one incorrect object partition failed
  independent bowl geometry. Its aggregate validation correctly fails.
- `carried-groups-parked-v3` (source `e94c115`): 167 fields and 27 separate
  member rows; all separate bowl centroids were within .144886 m of independent
  truth. Native actor, attachment and human-sensor gates passed, but the noisy
  wrist-depth motion reference and per-miss identity reset yielded zero confirmed
  groups. The aggregate remains failed. A relation-only offline replay motivated
  the bounded identity and rotation-invariant corrections; it is not sensor field
  acceptance or a new navigation trial.
- `carried-groups-parked-v4` (source `3b96377`): 167 fields, 24 separate rows
  and no confirmed groups. The saved source image at 7.1 s shows a background
  mask accepted by the old broader one-wrist fallback, with .394727 m independent
  bowl error. This caused removal of implicit mode fallback, not relaxation of
  the geometry gate. Human-sensor and native attachment checks still passed.
- `carried-groups-frontal-v5` (source `99d34f6`): the direct west-side pose
  `(26.2,9.3)` had body clearance but a blocked RGB view. The stationary human
  gate aborted before actor motion; no successful component or navigation
  acceptance is claimed. Keep its black frames, startup error and failed sensor
  report. A free robot footprint does not establish camera line of sight.

## Accepted bounded parked component

`outputs/restaurant_service/carried-groups-diagonal-v6/`, functional source
`2a80dee`, passed the actor, attachment, human-sensor and carried-group gates.
The separate authored diagonal pose and crossing-facing initialization above
were used. All robot wheel commands were zero; the original scene sources
remained unchanged. The full run took about 408 wall seconds and is simulation
paced, not a real-time result.

- 167 current fields; 26 fresh separate person/object assessments, each retaining
  its own four numeric policy factors and mass provenance.
- Three confirmed moving groups at source times 7.6, 7.8 and 7.9 s, with the
  maximum individual phi applied to both member occupancy policies.
- Maximum independent bowl-centroid error over all 26 separate rows: .158226 m
  against the unchanged .25 m component gate. A visible-return centroid is not
  the object's full center of mass.
- All three group completions were .183333 simulation seconds old, below the
  existing .35 s gate. This does not measure wall-clock control deadlines.
- Human mass remained the 70 kg adult prior; bowl mass came from its own rough
  RGB/LiDAR network. No authored mass or fall height entered estimation.

All three group phis were 1.0: human phi was 1.0 and bowl phi was respectively
.817847, .823367 and .814419. The person's visible-silhouette contact proxy
saturated; potential and thermal each remained explicit .25 uncertainty priors.
The source-7.8-s figure `carried-groups-preview.png` shows the factor-specific
finding. This is not measured physical sharpness, maximum real-world danger,
temperature, calibrated risk, or evidence that the bowl was cold. No confidence,
factor scale or prior was adjusted to lower these results. Maximum-member unit
cases separately cover either member dominating and ties.

The group was confirmed only during a brief clear-view interval. Later turns,
mask rejection, wrist ambiguity and occlusion still lose the association.
Standing proximity alone never establishes carrying. This passes a bounded
stationary-robot dataflow gate, not continuous carrying coverage, grouped robot
navigation, a causal CRA benefit or the PR #117/main acceptance gate. Separate
carried-object fall/release/landing and plume evidence remain unvalidated. A
dedicated grouped-navigation adapter/gate and improved association coverage are
next; no new robot-wheel experiment was enabled in this branch.

Unit/regression validation passed **1,984 tests in 76.74 s**
(`outputs/carried_object_model_audit/full-unit-v3.log`). Repository Ruff and
whitespace checks passed. Coverage includes mode fallback rejection, source
epochs, disjoint positive ownership, independent motion/mass, stationary and
non-carried negatives, bounded turning, missing-instance identity without
evidence renewal, full field transaction rollback, both maximum-policy cases,
and independent report failures. The initial failed source/view trials remain
available alongside the accepted one. Unit tests and copied-scan probes do not
replace these sensor diagnostics or establish new wheel acceptance.
