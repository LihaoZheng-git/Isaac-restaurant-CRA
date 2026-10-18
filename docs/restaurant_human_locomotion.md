# Native Isaac human locomotion

The official `F_Business_02` human can now perform native articulated
idle / locomotion / stop episodes in the existing restaurant. Nova Carter
receives only zero wheel commands. This component gate establishes actor
motion, rendering and collision-envelope behavior, **not moving-human
perception, kinetic estimation or robot-crossing acceptance**.

## Native asset and runtime contract

The motion source is the official Isaac 6.0
`Isaac/People/MotionLibrary/HumanMotionLibrary.usd`, referenced from the
same NVIDIA asset root as the character. Its audited SHA-256 is
`9a6f8b5df9426d609f29c48feace8933d636f8c7a9b15711b88313550bb49b8d`;
the setup stops if the top-level library is unavailable or changes. It
contains walk and jog motions. The installed native API does not expose the
selected clip name, so the faster test is reported by its requested and
observed speed, without claiming a separately verified sprint animation.
The character's 101-joint retarget tags allow native retargeting despite the
earlier direct standing-clip joint-name mismatch.

Only already installed extensions are used: behavior core 110.1.4,
navigation core 110.1.4 and retarget core 110.1.0. NVIDIA geometry, textures
and animation assets remain external references under the terms linked in
[the character audit](restaurant_isaac_human.md). No NVIDIA bytes are added
to Git. The motion library's transitive clip files are not individually
fingerprinted by the top-level library hash.

All character setup and camera geometry are unsaved process/session edits.
Metre/Z-up NavMesh settings are explicit unsaved root metadata, with the
bounded include volume `(27,7,-.2)` to `(31,14,2.5)`. Native automatic agent
avoidance is disabled for this controlled episode. The target is
`(29.8,12.5,0)` from the original server location near `(29.8,8.9,0)`.
These locations, requested speed and native joint/root poses are **oracle
scene/evaluation information**, never estimator or CRA inputs.

The previous static collision capsule is disabled in the session. A separate
Z-axis kinematic guide capsule is attached to the animated rig: radius .4 m,
overall height 1.7 m. This is an explicit scene collision envelope, not
calibrated anatomy. No physical mass is authored. Acceptance reads its actual
PhysX tensor world pose; ordinary USD transforms do not reflect native Fabric
animation reliably. Update-phase lag is monitored with a .03 m bound.

The independent fingerprinted authored-layout raster rejects actor clearance
at or below .02 m. The parked robot has separate configured/settled footprint
clearance checks. Neither monitor supplies perception, CRA or control inputs.
Acceptance additionally requires actual task arrival, observed moving speed,
relative left/right foot motion, root grounding, stop drift at most .03 m,
parked-robot drift at most .02 m and zero wheel commands. RGB recording must
decode at least 20 frames and contain at least 10 distinct image arrays.

## Validation records

Accepted local records and rendered frames are under
`outputs/restaurant_service/human-walk-v10/` and `human-fast-v1/`.
The diagnostic preserves raw physics/root/foot records, image epochs, PNG
frames, a decoded MP4, setup checks and source-file hashes. No restaurant,
robot, dressed-prop or saved static-human layer is changed on disk.

| Metric | Walk (.7 m/s request) | Faster (1.6 m/s request) |
|---|---:|---:|
| Observed steady median speed | .692865 m/s | 1.691414 m/s |
| Relative foot-motion range | .877300 m | 1.418943 m |
| Stop drift | .019444 m | .011333 m |
| Minimum independent actor clearance | .307024 m | .302612 m |
| Decoded / distinct RGB frames | 81 / 77 | 57 / 48 |
| Endpoint, grounding, collider, parked robot, video gates | pass | pass |

Both source-file checks passed. Configured/settled parked-robot footprint
clearance was 1.66539/1.66427 m in the faster trial. Both actor-only episodes
remain distinct from the existing stationary perception gate and from
PR #117's unmet fresh moving-view perception gate.

Full unit regression: **1,788 passed** (including 14 actor-motion contract
tests); repository Ruff passed. Tests reject sliding without limb motion,
wrong speed, missing arrival, collider lag, robot/wheel motion, insufficient
clearance, continued motion after arrival, ungrounded/nonfinite geometry and
duplicate epochs. They also preserve unrelated NavMesh metadata and disk files.

Earlier `human-walk-v1` used an ordinary-USD collider monitor and could not
measure the moving physics body. The corrected tensor readback in v2 showed
one-step following, including a .0402 m startup transition; subsequent episodes
explicitly settle native idle before their recorded idle/walk/stop window.
Trials v1-v8 did not pass the recording gate. In this installed Kit build,
late activation of `omni.anim.behavior.core` caused both the known robot camera
and the new diagnostic camera to produce no RGB. An isolated zero-wheel
comparison restored both streams when loading the same extension at
application startup. The accepted script uses that startup order, an RTX
camera and motion BVH; no dependency or environment upgrade was made.

Warnings remain: deprecated authored robot fisheye projections/NavSchema,
large source collision triangles, and renderer transform-history storage
despite a request of 128. Shutdown can also report invalid-attribute queries.
The recorded success does not resolve those warnings or validate long-exposure
moving sensors. No real-time claim follows from the recorded simulation.

## Reproduction and next gate

Use the installed Isaac launcher and a fresh output directory:

```text
C:/isaacsim/python.bat -u scripts/demos/restaurant_human_locomotion_isaac.py
--scene scenes/generated/restaurant_service_isaac_server_v2.usda
--layout-survey outputs/restaurant_service/layout-monitor-v1/geometry-survey.json
--output outputs/restaurant_service/human-motion-new
--speed 0.7
```

The separately tested faster request is `--speed 1.6`. The configured .5..2 m/s
range is an engineering guard, not acceptance of every speed in that interval.
The previous stationary human's 1.921 kg catalog-network result remains
unvalidated. At this actor-motion checkpoint, the approved configurable
human-mass PRIOR still needed a domain/association path; the later
[human perception component](restaurant_human_perception.md) implements and
tests that separate opt-in.
Next: parked-robot RGB/LiDAR tracking and current relative-velocity/kinetic
checks on this articulated actor, with no permanent static occupancy memory
or unchanged-shape bounds borrowing for the moving human. Only after that gate
should a separate controlled robot/human crossing be tested.

This validated child branch targets the live development branch. Draft PR #117
and the main branch retain their separate hold.
