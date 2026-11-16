# Parked-robot perception of the native walking human

The selected official Isaac adult-human scene now passes a separate RGB/LiDAR
component benchmark during native walking. The worker produces four numeric
CRA scores and Safe-SAGE fields for current geometric tracks. The robot remains
parked; this is **not robot/human crossing, full four-factor sensing, calibrated
human mass, real-time operation or PR #117 moving-view acceptance**.

## Evidence and models

An optional fresh visual person mask selects an explicit configured adult-human
mass **PRIOR** only when it unambiguously overlaps the independently accepted
same-frame SAM/LiDAR instance. The default engineering assumption is 70 kg,
interval 45--100 kg, confidence at most .5. It is configurable with
`--human-mass-prior KG LOWER UPPER`. It is not a population-calibrated interval,
measured weight, carried-load estimate or trained human-mass network. An asset
name, authored position, height, density or role never selects this prior.
The SAM class remains unknown and the legacy category-free mass API/class
guard is preserved. The original catalog-network result is retained separately
in `geometry_evidence.catalog_mass_before_human_prior`.

The additional model is the official COCO-pretrained YOLO11n segmentation
checkpoint, used through the existing adapter. See the
[official model documentation](https://docs.ultralytics.com/models/yolo11/)
and [release](https://github.com/ultralytics/assets/releases/tag/v8.3.0).
The external download is 6,182,636 bytes, SHA-256
`55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`.
It loads with the existing Ultralytics **8.4.104** environment, reports 80 COCO
classes/person class 0 and task `segment`. It detected the separate native
motion preview person at confidence .8563; this is a compatibility check, not
a detector accuracy study. Ultralytics documents the model under AGPL-3.0 /
Enterprise licensing. No weights are vendored, no dataset is downloaded or
retrained, and no package is installed/upgraded. SAM, ABO mass and contact
weights remain the existing separately audited models.

The person/instance adapter rejects unmatched tracks, wrong frame/epoch/shape,
stale/future evidence, weak class cues, boundary truncation and competing
overlapping person masks. A .5 mask-IoU threshold and .15 competitor-IoU threshold
are explicit bounded association assumptions. Camera read time remains an
acquisition-clock approximation; robot pose/extrinsic references remain oracle.
No language/VLM reasoning is used.

## Articulation and acquisition

This opt-in benchmark disables intrinsic-property retention and translated
instance bounds. Accepted current masks/geometry are required again for a fresh
human prior. A current visual person cue reserves one of two visual slots;
the other retains the existing fair schedule. The .2 s visual reassessment
interval remains enforced. This prioritizes attention only; SAM quality,
boundary, foreign-return and final ownership gates are unchanged. Other tracks
and unmatched physical occupancy keep their numeric policies. The legacy
static-session mode and callers without this opt-in remain unchanged.

Static occupancy memory is disabled. Positive scan occupancy expires after
.15 s, and current LiDAR tracking supplies planar relative/world velocity.
Support, fall height and heat are never borrowed or filled from the actor's
authored body height. Potential and thermal stay explicit .25 factor priors.
Contact uses a fresh visible-silhouette/exposed-edge proxy when accepted;
physical sharpness/hardness remain unavailable. A missing contact cue uses its
configured prior rather than establishing harmless anatomy.

The native RTX scan can expose its final declared return approximately one
physics step ahead of available pose history. The diagnostic copies the scan,
metadata, RGB and original epochs, then waits at most .05 simulation seconds
for recorded poses. The unchanged deskew/epoch checks run only afterward;
there is no extrapolation or source-time rewrite. Rejected and duplicate
acquisitions retry on the next physics tick; cadence limits successful packets
so a stale phase in the 10 Hz scan cycle cannot lock out all later observations.
Per-return ego correction does not correct articulated/moving actors.

An asynchronous worker receives only copied sensor packets, robot references
and fixed waypoint context `(30.6,11.8)`. Native actor state is stored separately
for evaluation. Independent planar matching uses .45 m, compares inferred
centroid velocity with the native root-motion reference and excludes transition
speeds below .4 m/s from the steady moving error metric. This is a bounded
proxy comparison; articulated centroid motion is not identical to root motion.

## Actual benchmark and failures

`outputs/restaurant_service/human-sensor-v4/` passed. Requested human speed was
.7 m/s; the goal was `(29.8,10.5,0)`, a shorter component route than the earlier
actor-only validation. Simulation pacing was explicitly .1. The robot received
zero wheel commands throughout.

| Check | Result |
|---|---:|
| Completed source fields / object assessments | 55 / 8,809 |
| Fresh visual human-mass-prior fields | 27; one human track (134) |
| Independently matched stationary / steady moving rows | 10 / 8 |
| Median stationary inferred speed | .012023 m/s |
| Median steady world-velocity error | .084934 m/s (gate: at most .25) |
| Actual native moving speed | .707089 m/s |
| Actor minimum capsule/raster clearance / stop drift | .410867 m / .003534 m |
| RGB video decoded / distinct frames | 75 / 68 |
| Maximum completed-result source age | .416667 simulation seconds |

All actor motion, grounding, collision-envelope, source-preservation and parked
robot checks passed. The current independent validator re-read the saved raw
records and confirmed fresh mass epochs, current inferred motion, positive
ownership, finite four-score/phi values and no property borrowing. The complete
records SHA-256 is
`c76eb669e448295a899afa6d3aac64056605478118137916d06ea9453fc6a6b1`.
See `compact-audit.json` and `human-sensor-validation-current.json` there.

Of the 27 fresh-human-prior fields, kinetic was evidence-supported in 26,
contact in 17, and potential/thermal in none. The other cases retained labeled
priors. The original catalog network ranged 1.0503--2.4042 kg on these human
views, which is not credible human mass. Human phi ranged .631761--1.0.
At source time 2.8 s, contact had nominal silhouette risk 1 and zero combined
confidence during motion initialization; the unchanged conservative assessor
retained that high nominal value despite the .25 fallback prior. This explains
the initial saturation, rather than hiding unknown factors as zero. Later
silhouette scores were also high for some articulated outlines. These are
proxy/uncertainty findings, not measured sharpness or injury severity; no score,
confidence or aggregation was lowered to improve this trial.

The earlier v1/v2 trials failed before walking because pose history did not
cover declared return epochs. V3 retained source times with the bounded wait
but phase-locked acquisition retries; only four fields completed and moving
perception failed. Preserve those records. V4 also preserves six empty-scan,
one out-of-history and ten stale/future-epoch acquisition rejections. No failed
packet was accepted. Renderer-history, authored fisheye and GMO startup warnings
remain. The .416667 s result is not a control-freshness pass; a navigation caller
must reject late results using its own unchanged caps. Large JSON audit writes
and paced simulation also preclude interpreting this run as real-time timing.

Full regression: **1,813 unit tests passed**, including rejection, freshness,
no-retention and copied-scan epoch tests; repository Ruff passed.

```text
C:/isaacsim/python.bat -u scripts/demos/restaurant_human_locomotion_isaac.py
--scene scenes/generated/restaurant_service_isaac_server_v2.usda
--layout-survey outputs/restaurant_service/layout-monitor-v1/geometry-survey.json
--output outputs/restaurant_service/human-sensor-new
--speed .7 --goal-y 10.5 --sensor-diagnostic --sensor-simulation-rate .1
```

This accepts the selected one-adult, parked-robot component case. It does not
validate children, carried loads, crowds, crossings, general human mass,
biological contact consequence or moving-view plume. Next is a separate control
integration gate: carry only justified intrinsic mass priors through visual
cadence gaps, keep geometry/motion current, preserve track-loss stops, reject
late fields, and test a controlled crossing before extending the service route.
PR #117's main-merge hold remains separate and unchanged.

The subsequent bounded control integration and its failed/retried standing
reference are documented in [restaurant_human_navigation.md](restaurant_human_navigation.md).
It keeps explicit priors and current geometry; this parked benchmark is not
retroactively a crossing or real-time validation.
