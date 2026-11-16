# Stationary restaurant service-container test

The larger floor-standing containers pass the real camera/LiDAR ownership and
CRA component checks. They do **not** establish an angular-versus-rounded hazard
ordering: the rounded bin has a slightly higher silhouette score in this view.
Do not present either bin as a knife-sharpness demonstration.

## Native results

Frozen runtime `a5a4ac7`, recipe `configs/restaurant_container_bench.json`:

| Prop | Accepted / matched observations | Fresh supported contact updates | Median outline cue | Median contact factor | Median CRA phi |
|---|---:|---:|---:|---:|---:|
| Angular service bin | 18 / 18 | 17 | 0.375848 | 0.116033 | 0.596008 |
| Rounded lidded bin | 18 / 18 | 17 | 0.420314 | 0.121116 | 0.598332 |

Both independent stationary gates and frozen-source execution checks passed;
zero nonzero wheel commands. The first observation lacks a supported motion
interaction estimate, so it is excluded from the 17-update contact comparison.
The final angular/rounded instances own 419/364 LiDAR returns and 1372/1141 mask
pixels. Their independent visible-centroid errors are 0.186150/0.157769 m, within
the unchanged 0.20 m gate. Masks are accepted by the existing SAM quality,
boundary and ownership checks; no supplied category, density or risk label.

The native records are `outputs/container-bench/angular-v3` and `rounded-v3`.
`comparison.json` and `comparison.png` summarize the original observations;
the local `summarize.py` replays the existing silhouette estimator on the archived
accepted masks for the cue column. Factor scores and phi come from native runtime
records, not from that replay. Gate SHA256:

- Angular: `ad9f70dc73dbd338ba20753efb4bf7d6ce7bc173acd5893db8056915a6d3f25d`
- Rounded: `f5b24da2c7ec883145b7df3642fe0fa740c705d99e6eea47fa35b320062ba600`

## Setup and interpretation

Both props are 0.32 m wide/deep and 0.48 m tall, with matched colors and world
position `(28.15, 6.10)`. The robot observes from `(26.3, 6.1)` with the front Owl
raised 0.60 m, level, using its actual lens/extrinsics and 640x400 RGB capture.
The separate benchmark layer disables the distant background server; it does
not remove furniture, change target acceptance or establish human acceptance.
Source scene files remain unchanged. This is a virtual sensor-mount experiment;
no physical mast collision/dynamics model has been added.

The camera/SAM mask, associated LiDAR shape and tracked motion supply the inputs.
Mass is the existing category-free ABO RGB/LiDAR rough estimate, uncalibrated in
Isaac. The existing simulator ego-pose reference remains explicitly oracle.
Object/fixture truth is used for independent scoring only.

At the final observation, potential and thermal use their labeled 0.25 priors;
kinetic is evidence-supported with a conservative score near 0.1875 because the
uncalibrated mass has low confidence. Ground placement in the scene does not
silently become measured fall-height evidence. No plume is not a temperature
measurement. These results are not complete fresh four-factor sensor acceptance.

Contact uses the existing maximum of circularity deficit, elongation, concavity
and corner cues. The rounded cylinder looks narrower in the accepted image mask,
and its elongation term dominates. At the final frame, angular/rounded elongation
is 0.377/0.421; the angular maximum turning angle is larger (1.531 versus 1.030
radians), but the total proxy is not a pure corner detector. The results expose
this limitation rather than justify tuning the estimator to reverse the order.
Physical sharpness, material hardness and hidden surfaces remain unmeasured.

## Preserved earlier attempts

- `angular-v1`, runtime `0e1e6ac`: all 18 masks accepted, but independent target
  matching failed. At the closer robot x=26.8 view the upper-front-surface centroid
  is 0.214227 m from the authored center, outside the original 0.20 m gate.
- `rounded-v1`, same source/view: passed, retained as a separate setup.
- `angular-v2`, runtime `1708db9`: farther x=26.3 view passes every target check,
  but the separate distant background-human mask gate fails. This is not counted
  as a full component pass. The final prop-only runs isolate the requested object
  without lowering target or human gates.

The farther view exposes more of the lower container to the unchanged LiDAR
vertical field of view. No mask threshold, matching distance, uncertainty prior,
controller margin or factor mapping was relaxed.

## Demo staging and remaining work

`scenes/generated/restaurant_service_demo_v3_container_candidate.usda` is a new
additive scene: the angular bin replaces the old floor bin, with the full scene's
server retained. Its recipe is
`configs/restaurant_service_demo_v3_container_candidate.json`. This is an ordinary
visible-object candidate, **not** a validated high-contact hazard or final v3 video.
Accepted v2 is unchanged.

The inherited human commands in this partial candidate are placeholders. The
approved straight, one-way walk from the far end still needs implementation;
do not use the old circuit for the final v3 demo. The pot/plume ownership test
and a convincing exposed-edge/contact example also remain pending. New full-scene
stationary/human gates, actual moving-view observation and full arrival/stopping
must pass before promoting a new demonstration. No wheels were run in this task.

## Reproduction

Build each benchmark scene with
`scripts/scene_creation/build_restaurant_container_bench.py --base
scenes/generated/restaurant_service_demo_v2d.usda --output <new-scene.usda>
--shape angular|rounded --props-only`. Preserve old outputs and update a separate
recipe to the new generated scene. With clean committed source, run
`scripts/evaluation/run_restaurant_demo.py --config
configs/restaurant_container_bench.json --case angular|rounded --stage stationary
--output <new-output-folder>` using the installed environment. The runner launches
the installed Isaac interpreter, fingerprints sources/scenes and requires the
actual report; a zero launcher exit alone is not acceptance.

Focused scene, mount, component, runner and SAM regressions: 70 passed. Full
unit regression: **3641 passed in 123.05 seconds**. Ruff passes for
`robot scripts tests`, and `git diff --check` passes. The full log is
`outputs/container-bench/unit-regression.log`. Early local test attempts hit a
temporary-directory setup problem; rerunning with an explicit new workspace
temporary directory resolves that environment issue without changing tests.
