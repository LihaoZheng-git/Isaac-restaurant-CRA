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
