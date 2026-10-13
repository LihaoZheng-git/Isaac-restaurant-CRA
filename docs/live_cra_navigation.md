# Live CRA navigation integration

The accepted next target is continuous robot navigation with refreshed physical
evidence and safety fields. No scheduled observation pauses belong in the final
restaurant route. Safety stops remain permitted. Existing category-free wheel
results used frozen perception and must not be relabeled as live integration.

## Sensor epochs and background inference

`robot/perception/sensor_epochs.py` keeps a bounded rigid ego-pose history,
interpolates translation and rotation without extrapolation, and maps scan-time
chassis points into the camera-time optical frame. It retains pose source/status,
rejects future/stale sensor pairs and large pose gaps, and derives actual robot
translation velocity from historical poses. The existing centroid tracker already
compensates ego motion when supplied the matching scan-time pose; its static-world
test now explicitly covers a translating/turning robot.

`robot/perception/live_inference.py` runs one background inference job without an
accumulating backlog. Worker inputs must be immutable copies; all Isaac/USD reads
and wheel commands remain on the simulation thread. Completed results keep their
original sensor timestamp. Late, future, failed and missing results are rejected,
not retimed or converted to zero-risk evidence. The default result-age gate is
0.35 s, matching the existing live-field limit.

The stationary Isaac diagnostic is
`scripts/diagnostics/live_sensor_epoch_isaac.py`. It captures real camera/LiDAR
metadata with only zero wheel commands, checks stationarity, and preserves the
on-disk scene. Transient renderer/physics changes to the in-memory root are
reported separately; no layer save is called. The earlier audit wrongly tested
unchanged in-memory serialization across normal initialization and is retained
as a rejected report.

The observed camera annotator metadata is empty in this installation. Camera
timestamps are wrapper **read times**, not independent acquisition timestamps.
The bounded simulation integration uses this declared pose-time approximation;
it cannot claim hardware synchronization, per-ray LiDAR deskew or calibrated
moving-camera projection accuracy. USD poses/extrinsics remain oracle references.
Reprojection/association quality must be validated during the driving diagnostic.

Core validation: **51 focused tests**, **1,572 full unit tests**, repository-wide
Ruff. No package installation, dependency upgrade, new wheel trial or saved scene
change occurred in this first step. Current controls, fixed lambda 1.25, CRA
confidence/aggregation and unknown handling are unchanged.

The final stationary audit passed eight distinct LiDAR/camera reads with 0.10 s
pair age, 0.00096 m robot drift and 0.00154 degrees rotation drift. The scene file
was unchanged. Artifact: `outputs/live_cra/sensor_epoch_stationary_final/report.json`.

## Remaining integration gates

1. A separately named moving-camera RGB plume method, with registration quality,
   source association and continuous cue validation. Preserve the older fixed
   clean-reference method and its stationary-only gate.
2. Refreshed SAM/mass/contact/geometry/motion evidence joined to the existing
   transactional Safe-SAGE field runtime. Account for changing field values in
   MPC/CBF through its temporal derivative; preserve all unassessed occupancy.
3. Stationary combined readiness, then actual bounded wheel trials with repeated
   accepted perception/field updates, goal completion, safety margins and stopping.
   Record processing latency and safety stops; do not claim real-time performance
   merely because the simulator can wait for computation.
