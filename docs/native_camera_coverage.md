# Existing-camera coverage diagnostic

The 0ed0f3c 13 s crossing lost fresh whole-human SAM association while the
person was cut off by the front Hawk image edges. High model mask quality did
not override the boundary gate. The mass source and short passing-selection
lease were not renewed; [the failed diagnostic](human_fragment_coverage.md)
remains preserved. A saved actual RGB frame is
`outputs/guidance_progress/native-fragment-coverage-13s-v1/sensor-snapshots/001260.png`.

A zero-wheel USD inventory with Isaac's resolver found existing front/left/
right/back Hawk stereo cameras and four Owl cameras. The front Hawk has an
authored maximum FOV of 150 degrees; the front Owl has 235. The inventory is
not proof of estimator compatibility or usable coverage of every pixel.

Both live and parked-human demos now accept `--rgb-camera front_owl` using
the camera wrapper's existing path selection. The default stays
`front_hawk_left`; `front_hawk_right` is also an existing option. Read the
selected camera's actual calibration and extrinsics, enforce its native
aspect ratio and retain its prim path in audits. This selects **one camera
per trial**; it is not multi-camera fusion, a synthesized wider image or a
change to the robot's authored hardware. Explicit mount pitch/raise, if used,
remain separately audited session hypotheses.

The LiDAR projector's positive forward-depth gate remains unchanged. Thus a
235-degree rendered image does not establish 235-degree LiDAR association;
rearward rays stay unassociated and all unmatched physical occupancy retains
its existing policies. All SAM quality, truncation, foreign-return, ownership,
physical evidence, original mass/identity/selection clocks and safety checks
stay unchanged. Raw fisheye appearance may still fail the pretrained visual
models. No perception integration claim follows from selecting the camera.

The first test is a parked-robot close-crossing component with zero wheel
commands and the existing stationary person/mass gate before actor movement.
It will use the authored Owl view with zero additional pitch or raise. A new
moving-robot experiment additionally requires the user-approved .25 m centroid
uncertainty prior and stationary validation of that configuration first.

Validation before the native diagnostic: 73 relevant camera-fusion, perception,
mass and coverage tests and repository Ruff passed. The preceding full source
baseline passed 2,251 unit tests. The camera-selection plumbing does not alter
estimator mathematics or weaken failed image-boundary checks. Native results
will be recorded separately; main/long-service remain held.

## Parked close-crossing result at 0d6b485

`outputs/guidance_progress/native-owl-parked-close-v2` passed both the native
actor and human-sensor gates. The robot stayed parked throughout. The camera
used its authored 1920x1200 polynomial calibration, acquired at 640x400,
235-degree maximum FOV, zero additional pitch and zero raise.

The independent sensor gate recorded **nine fresh idle and twelve fresh steady
walking rows**, median stationary relative speed .01969 m/s and median moving
velocity error .16170 m/s (unchanged .25 m/s limit). All four numeric-score and
no-property-borrowing checks passed. Maximum completion age was .25 s. Eight
capture/worker rejections remain in the output; the test never renewed them.
The native actor arrived/stopped with .28395 m independent minimum clearance,
and the robot received zero wheel commands. The useful visible-person frame
is `robot-current.png`; actor motion is in `human-motion.mp4`.

This is actual RGB/LiDAR component evidence with a configured adult mass prior
and explicit contact/potential/thermal priors/proxies. It is not measured mass,
physical sharpness, temperature, complete four-factor sensing or moving-robot
acceptance. The old cropped moving view and this different parked fixture are
not matched causal camera comparisons. A first v1 launcher was rejected for
pre-creating the demo's required fresh output directory, before simulation;
it remains separate from the accepted v2 run.

The next bounded moving diagnostic uses the same 13 s encounter and 20 s
partial-duration limit, existing front Owl with zero mount adjustment, and
the user's separately approved .25 m centroid uncertainty prior. It must
first pass its own 60-frame stationary component/envelope gates. Preserve
fixed lambda, original factor/identity/selection clocks and all exact checks.

After adding the explicit audited-Owl ray/scale regression, all **2,252 unit
tests** passed in 75.79 s and repository Ruff passed. The parked component's
median kinetic score was .12615 in idle and .20832 during steady walking;
these are uncertainty-aware scores with adult mass and other-factor priors,
not measured human energy or a causal navigation comparison.

## Moving Owl / wider-envelope diagnostic at 5b75a47

`outputs/guidance_progress/native-owl-coverage-13s-v1` passed its own 60-frame
stationary component/envelope gate before wheels. It used the existing Owl
with zero added pitch/raise, approved .25 m centroid uncertainty (.65 m total
body radius), fixed lambda and all original clocks/checks. This separate
camera/geometry configuration is not a matched causal comparison.

The 20 s partial-duration run ended at `duration_limit`, **not passed**, 1/3
waypoints, 199 live fields/136 while translating, .158716 m full-sweep clearance,
.0000138 m stop drift and .367957 minimum CBF margin. All 199 independent body
containment rows passed; worst remaining margin .047759 m. Eight supported
moving-kinetic rows passed their unchanged minimum-five gate, with .15212 m/s
median world-velocity error and .00825 m/s relative-speed error. The observed
selection/provenance/fixed-lambda checks passed across 64 identity-prior policy
rows, with original side-policy age <=6.3 s. Fresh current image associations
continued during the close crossing; no clock extensions were introduced.

The robot still waited: **zero simultaneous robot/actor motion rows**, failed
walking-window arc (-.118646 m against the .20 m gate), failed behind projection
(-.004078 m), no route completion/rejoin. Thus overall human-navigation and
passing gates remain failed. Across 13,850 object assessments there were 68
fresh instances and zero complete four-factor assessments. The 848.5 navigation
wall seconds are paced simulation, not real-time acceptance. Preserve the
earlier Hawk and smaller-envelope failures and main/long-service holds.

Saved numerical replay shows that the greedy projected references can hit an
unsafe configuration within their horizon. Unintegrated, outputs-only turn-
prefix prototypes were rejected; a bounded local beam prototype found complete
exact-safe paths in the original 19.8 s trusted field, including the mapped
analytical guidance condition. This is offline candidate-generation evidence,
not integrated code, wheel/arc acceptance or paper equivalence. The next bounded
controller work must retain the obstacle-unaware nominal, primary objective,
original field/source/execution clocks and every exact safety check.
