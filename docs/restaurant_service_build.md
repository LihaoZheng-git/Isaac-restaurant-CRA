# First restaurant service build

The first build adds 11 static food/tableware instances to the existing
restaurant, preserves the original restaurant and robot files, and accepts a
separate 2.2 m live CRA/Safe-SAGE route. This is a **prior-assisted, paced
development result**. It does not validate the proposed 30.88 m service route,
human interactions, payload transport, fresh four-factor perception during
motion, or a causal navigation advantage from CRA.

## Assets and saved scene

The selected [Kenney Food Kit 2.0](https://kenney.nl/assets/food-kit) subset is
CC0-1.0. The original license, 13 OBJ/MTL models, shared palette texture, member
hashes and archive hash are under `assets/restaurant/kenney_food_kit/`.
No downloaded executable was run, and no package was installed or upgraded.
The existing restaurant/NVIDIA assets retain their separate terms.

`robot/scene/restaurant_service_build.py` converts the audited meshes with the
existing trimesh/USD environment and authors a separate additions layer.
Static triangle collision retains bowl cavities and thin utensils. The
arrangements are ordinary mixed scenes, rather than authored factor profiles:

| Arrangement | New instances | Placement basis |
| --- | --- | --- |
| Front table | Plate, bread, coffee cup | Existing tabletop, z = .7466 m |
| Service counter | Knife block, plate, knife | Existing lower counter, z = .90 m |
| First food table | Bowl and spoon | Existing tabletop, z = .7466 m |
| Second food table | Pot and plate | Existing tabletop, z = .7466 m |
| Destination table | Plate | Existing tabletop, z = .7466 m |

These heights and mesh bounds are **authored placement/evaluation information**,
not fall height or physical-factor inputs. No mass, density, temperature,
sharpness, class label, desired score or phi is attached to an estimator.
The pot/bowl asset names do not establish heat. No plume emission, humans or
robot payload were added. Static props have not been validated as movable food.

The accepted local scene is `scenes/generated/restaurant_service_v2.usda`, with
`restaurant_service_v2_additions.usda`. Generated scenes and raw experiment
outputs remain ignored; the assets and build/evaluation code are tracked.
Original source fingerprints are unchanged:

```text
restaurant_base.usda  18c5e658e4b3a01f8acaf2aca9e7729edcb73fff3ea1796aeae5ad4c39d42870
restaurant_robot.usda 16043dba5a04ed228aff42b0026ead2979ab8b72c8d9899170874d5e1c9d9524
```

The actual Isaac preview is in `outputs/restaurant_service/preview-v2/`.
Five RGB views were rendered with zero physics-time advance and zero wheel
commands. All saved source bytes and prop bounds were unchanged. Replicator
added only an in-memory render-product relationship to the root; its diff is
retained, and no renderer edits were saved to the scene.

## Stationary acceptance and rejected attempts

The restaurant mode is explicit in `cra_live_object_isaac_demo.py`; legacy
fixture defaults are unchanged. It creates no sphere/puff fixture and requires
the persistent object map, static positive-occupancy prior and an independent
authored-layout monitor. The new FoodBowl component gate requires at least three
fresh accepted owned visual updates, inferred/measured mass and motion, an
accepted visual contact proxy, positive policy-return ownership and four finite
numeric policy scores. Unsupported potential/thermal inputs may use the user's
configured missing-factor priors; this is not complete sensor readiness.

Two failed attempts remain separate from the accepted result:

* `stationary-v1`: the first composed root omitted the original time range,
  causing repeated acquisition epochs. The builder now copies the source's
  60 time codes/s and 0--1,000,000 time range. A regression rejects zero-length
  source timelines before creating output. Wheels were never enabled.
* `stationary-v2`: a 640x480 acquisition did not match the camera's authored
  1920x1200 aspect ratio. Observed bowl returns projected above the visible
  bowl, and SAM association correctly rejected foreign table/chair returns
  despite high model mask quality. No association threshold was weakened.
  Restaurant mode now explicitly uses 640x400 and rejects an aspect mismatch.
  Projection formulas and the legacy 640x480 fixture acquisition are unchanged.
  This empirical correction is validated for this selected camera/asset view,
  not a general camera calibration study.

`stationary-v3` passed all five component checks across 18 processed stationary
fields, with three uniquely matched accepted bowl assessments and no wheel
commands. The latest bowl assessment was:

| Quantity | Result | Basis / limitation |
| --- | --- | --- |
| Rough mass | .4898 kg | Category-free network inference; confidence .25; bounds .001--5.032 kg; uncalibrated domain extrapolation |
| Potential policy factor | .25 | Configured prior: usable fall-height/interaction inputs unavailable |
| Kinetic policy factor | .1875 | Current estimated motion, rough mass and uncertainty |
| Contact policy factor | .1805 | `visual_silhouette_geometry_proxy`; not physical sharpness/hardness |
| Thermal policy factor | .25 | Configured prior: no validated negative-plume domain for this new asset |
| Aggregate phi | .62549 | Unchanged numeric prior-aware aggregation |

The latest SAM mask had quality .8992, 51 owned return hits, .9444 owned
coverage and zero foreign-return hits. Its prompt came from an observed owned
LiDAR return, not an authored bowl coordinate. Full four-factor SENSOR readiness
remained false. Earlier accepted stationary phi values ranged up to .8828;
uncertainty-dependent output was preserved rather than tuned toward low risk.

## Short live navigation result

`outputs/restaurant_service/short-live-v1/` passed its own 18-frame stationary
gate before wheels, then reached the 2.2 m waypoint and stopped. There were no
scheduled observation pauses. The nominal remained obstacle-unaware; online
MPC and analytical CBF retained the existing exact geometry/DTCBF, field-age,
optimizer-age and verified-command-expiry checks. Lambda stayed 1.25.

* 109 live fields, including 108 while translating.
* Minimum independent clearance: 1.4205 m; minimum analytical CBF margin: .91076.
* Stopping drift: .00242 m; final state within .10 m of the goal.
* About 11.02 simulation seconds of motion; 245.02 wall seconds for that stage.
  The requested simulation pacing was .05x, not real time.

The clearance monitor projects original authored mesh faces through z =
.10--1.80 m at .025 m resolution, conservatively includes new prop XY boxes,
and checks the actual asymmetric base footprint with its policy margin and
pixel allowance. It is **oracle abort/evaluation information only**, never
perception, CRA, nominal, MPC or CBF input. It is a raster proxy, not an exact
PhysX contact distance, and does not validate an elevated payload.

The full JSONL stream audit covered 5,420 per-track assessments: ordered source
IDs/times, .13333--.15000 s source-to-commit ages, four finite scores, phi in
.60648--.68359, fixed lambda, and 21,438 explicitly labeled .25 missing-factor
prior uses. It recorded 242 evidence-supported factors and 240 retained-property
uses. These are repeated per-track counts, not distinct fully perceived objects.

Only **one fresh accepted visual instance** appeared during motion. Bowl-center
diagnostics changed among IDs 9/77/122; proximity matching is evaluation only,
not evidence of persistent identity. Mass/contact can expire or lose association
and revert to labeled factor priors. The controller still retains observed
positive occupancy under its static-scene prior. No all-fresh four-factor
assessment occurred during motion. The legacy first-track counters are not an
all-object perception summary; the complete stream establishes these counts.
This wide-aisle test establishes scene/dataflow/arrival plumbing, not a forced
avoidance maneuver, reliable visual identity continuity or CRA superiority.

## Reproduction

From the repository root, use new output paths. The already installed Python
environment is `C:/Users/tayli/anaconda3/envs/env_60_isaacsim_yolo/python.exe`;
Isaac runs with `C:/isaacsim/python.bat`. The standalone build/survey scripts
adjust Windows USD DLL lookup only within their own process.

```powershell
& 'C:/Users/tayli/anaconda3/envs/env_60_isaacsim_yolo/python.exe' scripts/scene_creation/build_restaurant_service_scene.py --output scenes/generated/restaurant_service_new.usda
& 'C:/Users/tayli/anaconda3/envs/env_60_isaacsim_yolo/python.exe' scripts/evaluation/survey_restaurant_layout.py --output outputs/restaurant_service/layout-monitor-new
& 'C:/isaacsim/python.bat' -u scripts/demos/restaurant_service_preview_isaac.py --scene scenes/generated/restaurant_service_new.usda --output outputs/restaurant_service/preview-new
```

The accepted stationary command used the following options. Add `--motion
--maximum-seconds 60` only after the stationary gate passes (motion mode also
repeats that gate). Select the working Isaac Python environment as in the
existing project launcher; do not install another runtime.

```text
scripts/demos/cra_live_object_isaac_demo.py
--output outputs/restaurant_service/stationary-new
--restaurant-scene scenes/generated/restaurant_service_new.usda
--restaurant-layout-survey outputs/restaurant_service/layout-monitor-new/geometry-survey.json
--persistent-object-risk-map --static-occupancy-memory-prior
--stationary-frames 18 --vectorized-fields --save-sensor-snapshots
--plume-view-fov 120 --simulation-rate .05
--qp-iteration-limit 1000 --nonlinear-rescue-iterations 20
--visual-reassessment-interval 5 --live-visual-tracks 1
--stream-live-records --immediate-replan --planner-result-max-age .5
```

The tracked survey reproduced the earlier accepted mask pixel-for-pixel:
328,969 occupied pixels, 5,555,775 projected faces, 2,598 meshes. Its generated
manifest additionally fingerprints the mask. An independent build reproduced
all 11 accepted prop bounds and source clock settings. Unit tests cover asset
hashes/UVs/collision, layered preservation, timing, component provenance gates,
asymmetric clearance, height clipping and changed-source/mask rejection.
Final regression: **1,743 unit tests passed**; repository Ruff and code/document
diff checks passed. The full staged whitespace check reports the preserved
vendor source/license line endings and the USD serializer's final blank line;
those asset bytes were not silently rewritten. All 28 original staged asset
files match the intake hashes. Raw logs remain under
`outputs/restaurant-service-*.log`.

## Next bounded work

1. Stationary perception validation for the pot, utensils and other selected
   props; retain explicit priors where evidence is unsupported. Improve crowded
   table/support association and bowl identity continuity before stronger live
   perception claims. No invented low thermal observation from an unvalidated
   no-plume view.
2. Enable the longer route through bounded field coverage/reacquisition rather
   than silently extending property timestamps. The current fixed field and
   120 s property expiry cannot establish the 30.88 m route at .2 m/s. Then run
   matched geometry/controller/budget comparisons and repeat trials.
3. Later add selected human assets and a separate moving-actor validation;
   static occupancy memory cannot support moving people. Payload placement,
   footprint/stability and hot-plume examples need separate gates as well.

This validated child work targets the live development branch. PR #117 remains
draft and unmerged into main while its stricter fresh moving-view perception
acceptance is open; this short prior-assisted result does not close that gate.
