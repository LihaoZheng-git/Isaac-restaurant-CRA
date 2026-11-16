# Raised-camera visibility survey

The user requested a simplified stationary check of whether a higher robot
camera can see the pot and sharp utensils, before carrying the mount into v2.
The accepted v2 scene, prop locations, robot dimensions and navigation archive
are unchanged. This experiment raises the optical sensor only; a physical mast
and its collision/inertial properties have not been modeled.

At runtime revision `4032675`, the paused native survey captured 16 actual
front Owl images: raises of 0, 0.20, 0.40 and 0.60 m, at two existing prop-facing
observation poses and two points/headings on the authored wide-corner route.
The robot is repositioned between paused views, not driven between them.
Physics stays off, timeline stays at zero, and no wheel commands are issued.
Original lens calibration and the runtime 640 x 480 resolution are retained.
All 72 hashed source files and both demo/food prop bounds remained unchanged.
This is capture-integrity acceptance and qualitative visibility review only.

| Raise | Optical center world z | Qualitative observation |
| --- | --- | --- |
| 0 m | 0.44821 m | Targets partly visible; low viewpoint and substantial peripheral self-occlusion. |
| 0.20 m | 0.64821 m | Less peripheral obstruction, still below the table surface. |
| 0.40 m | 0.84821 m | Near tabletop level; holder and pot visible. |
| 0.60 m | 1.04821 m | Preferred tested candidate: clearer view over the table edges, with level pitch. |

The scene uses ground near world z=0. These are optical-center coordinates,
not a new robot body height. No claim of an optimal mount is made. Targets were
not wholly absent at the original height, so low mounting is not established
as the sole cause of earlier segmentation/association failures. The pot and
stylized plume are visible with the raised camera; the utensil holder and fork
silhouettes are visible too, but the individual tips remain small. Route-facing
targets are on the right side of the image rather than centered.

Local artifacts are under `outputs/raised-camera-visibility/survey-v4/`:
`report.json`, `visual-review.json`, 16 raw images, `height-sweep.png`,
`comparison-facing.png` and `comparison-route.png`. Yellow boxes in comparisons
are manual locators, not detector masks. Authored poses/prop names only select
and explain the preview; no CRA estimate is computed.

The first sandbox startup was interrupted before scene loading. Two subsequent
startups failed on the existing Kit/project vision-package import-order issue;
the explicit traceback is retained in `survey-v3/failure.txt`. The final script
uses the same pre-Kit import ordering as the native perception demos. Nothing
was installed, upgraded or replaced. The failed launch logs remain available.

Reproduce with the existing Isaac launcher and configured project Python:

```text
C:/isaacsim/python.bat -u scripts/demos/restaurant_camera_height_preview.py --scene scenes/generated/restaurant_service_demo_v2d.usda --output outputs/raised-camera-visibility/new-survey
```

`configs/restaurant_service_demo_v2_raised_camera.json` carries the selected
candidate into the existing v2 long-service configuration. Compared with the
wide-corner recipe, only the camera raise changes in startup/near/motion and
human-component commands; the actor-only command is identical. It includes
those four cases only: legacy nonhuman prop CLI recipes reject raised mounts
and are not silently presented as raised-camera tests.

The candidate has NOT been run through perception or navigation. Fresh
stationary image/LiDAR association, human coverage and all normal prerequisites
must precede new wheel trials. Visibility alone establishes no SAM acceptance,
contact/plume factor improvement or four-factor CRA readiness. The camera
extrinsics will be re-read by the existing runtime mount path. Keep accepted v2
and all main/primary/PR117 holds intact.

Validation: 19 camera-mount/projection regression tests passed in 4.87 s and
44 service-v2/wide-corner regressions passed in 4.97 s. Repository-wide Ruff,
diff checks and CLI help passed. The candidate recipe was compared programmatically
with its parent to verify the three height-only changes. No navigation test was
run and no earlier perception/navigation acceptance transfers to this mount.
