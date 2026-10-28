# Quadruped command-interface feasibility comparison

The tested stock Spot policy is **not ready to replace Carter at the current
low navigation speeds**. It accepts body `(vx,vy,wz)` commands, but this
installation/setup did not track the tested 0.15–0.20 m/s translation commands
closely enough and exceeded the unchanged 2 cm stopping gate. This is a
bounded interface result, not a claim about every quadruped or a proven
internal policy deadband. No quadruped CRA/Safe-SAGE navigation was enabled.

## Final native result

Frozen source: `f3e6479e7c41fc467aebf5f7b728160cb55ae808`.

The independent `standing-spot-v7` and tracking run's own standing gates passed
with 15 s initial settling, then 160 distinct observations over approximately
3.2 s: maximum XY drift 0.000100 m, yaw drift 0.000114 rad and tilt 0.034062 rad.
The robot was freely standing, with no mount. This settling occurs before
assessment/movement; no scheduled navigation observation pause was introduced.

| Movement phase | Requested body command (m/s, m/s, rad/s) | Settled XY RMSE (m/s) | Yaw RMSE (rad/s) | Following zero-command displacement (m) |
| --- | --- | --- | --- | --- |
| Forward | (0.20, 0, 0) | 0.201330 | 0.015031 | 0.075126 |
| Lateral | (0, 0.15, 0) | 0.149853 | 0.006587 | 0.042998 |
| Yaw | (0, 0, 0.30) | 0.065857 | 0.036255 | 0.019379 |
| Combined | (0.15, 0.10, 0) | 0.179748 | 0.008178 | 0.064007 |

Translation RMSE acceptance remains <=0.08 m/s, yaw <=0.15 rad/s, stopping
<=0.02 m. Forward/lateral/combined tracking and their stopping phases failed;
yaw and its stopping phase passed. Settled median translation was near zero
in the failed moving phases. The trace retains the initial response and
subsequent drift. Stop displacement starts at the first zero-command sample,
including braking; it is not only late settled drift. All requested axes were
exercised. No speed floor, policy-command boost or relaxed stopping threshold
was used to improve the result. The exact cause requires further investigation.

The final report SHA-256 is
`8fd6fd250dbb46beaf871729e6a4af83267e01af14b266b13b757dd016bc1b4b`.
Local artifacts are under
`outputs/quadruped_feasibility/tracking-spot-v7/`: `report.json`, `records.json`,
`native.mp4`, `native-current.png`, `command-tracking-v1.png` and its immutable
report-hash companion. The sibling invocation/launch files preserve the actual
command and source. Saved reports, not launcher exit codes, establish results.

## What was and was not compared

Carter's already accepted restaurant run remains unchanged: three goals,
0.922686 m minimum independent clearance, 0.002754 m stopping drift, and a
separate smoothing failure at a 0.70 s nearby hold. It is a **historical
navigation reference**, not a matched actuation experiment or causal platform
comparison. See `predictive_guidance_continuity.md`.

A new Carter flat-floor bench did not earn a standing gate. Its first two
attempts stopped on root-height setup guards. A strictly non-accepting geometry
diagnostic later recorded XY drift 0.005490 m and yaw drift 0.044622 rad
(>0.03) during its observation window. No Carter tracking trial was authorized
by that diagnostic. Carter's inability to execute lateral velocity is explicit;
its bench supports only forward/yaw phases and never projects a lateral command
onto the wheels. No new matched human-passing comparison has been completed.

Go2 has no accepted standing gate here: attempts stopped during startup, most
recently at root z=0.249937 m. The common height envelope is not a calibrated
Go2 fall detector; do not infer general Go2 instability from this setup.

Before quadruped navigation, a low-speed-capable locomotion policy or separately
reported tracking adapter must pass command/stop gates, then body-mounted RGB
and LiDAR need standing and walking validation, a swept-footprint/tracking-error
treatment, and the original CRA/field/freshness/continuity/MPC/analytical/
clearance/arrival/stopping gates. Existing Carter and CRA modules are unchanged.
No primary/main/PI-demo/long-service/PR117 hold is released.

The latest user preference remains observed approaching motion/predicted
encounter for pronounced passing. Quiet-person early preparation is a separate
ablation; no such policy was added to a quadruped. The waypoint nominal must
remain obstacle-unaware and runtime CRA must remain visual/physical.

## Audited candidates and implementation

Installed Isaac Sim: `6.0.1-rc.7+release.42383.32955d8d.gl`;
`isaacsim.robot.policy.examples` 5.2.12. Stock Spot and Go2 wrappers use 48
observations and output twelve joint-position targets. Spot's audited config
uses 500 Hz physics / 50 Hz policy (dt .002, decimation 10, action scale .2).
Go2 uses 200 Hz / 50 Hz (.005, 4). Carter uses its unchanged controller and
60 Hz physics. Different plant/timing configurations are explicit, not a matched
latency benchmark. No package upgrade, new training or installed-source patch.

Official root:
`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0`.

| File | SHA-256 |
| --- | --- |
| Spot `spot_policy.pt` | `d95d675e8b080d5c84a2a2779d165ff7ecd3e34d24b233c083242e615a3d589a` |
| Spot `spot_env.yaml` | `397440818604ca7cf0cb79c6012325f2bba292029ad0c2d5df5fd64b531d2dbc` |
| Spot `spot.usd` | `dbe17178d8507aa31a82e8f8582606c2da5694ea8d2e24c77e50f1ff149689cb` |
| Go2 `physx_policy.pt` | `984c802b8acd68c3199605ff36603f91ac9f3192eb1ebcfd80b0e764831abe0f` |
| Go2 `physx_env.yaml` | `da282802fc3bec2f4d1a80818276680b91f73a716e26069bfee52e3fdf4d19af` |
| Go2 `go2.usda` | `29405c5d2a29b1a80e268df5de23898cb8f48ce95646352c25aeca512bfcb3df` |
| Carter `nova_carter.usd` | `462c3e91703c2f3aead2f15520a76c026bca1327fa61c40b268688d8711f938f` |

Local manifests live in `asset-audit-v1`, `go2-asset-audit-v1` and
`carter-asset-audit-v1` under the experiment output directory. Runtime verifies
the root USD against its downloaded original. Go2 references further Menagerie
payloads; used USD-layer identities are retained, not a complete dependency or
texture-license audit. Python source headers specify Apache-2.0, while the
package license notice is proprietary. Do not describe all weights/assets as
Apache or redistribute them; assets stay ignored and the installed wrapper is
imported. See the [official example](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/robot_simulation/ext_isaacsim_robot_policy_example.html)
and [additional materials terms](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/).

`robot/control/planar_velocity_command.py` rotates fixed-frame velocity using
current yaw and rejects out-of-domain commands (translation norm .25 m/s,
yaw .8 rad/s) rather than silently clipping. It is not integrated into the
safety controller and cannot send arbitrary planar commands to Carter wheels.
`robot/evaluation/quadruped_feasibility.py` validates strict-JSON root samples,
source ordering, phase completeness, platform root conventions and gates.
The standalone bench requires a fresh output path and a matching independently
passed standing report/source/assets/settling duration before tracking, and
repeats its own standing gate. External camera and root truth are evaluation
only, not robot sensing, mass, CRA or navigation input.

## Setup diagnostics and preserved failures

All output cases below remain unchanged. The experiment inventory stores their
original report hashes and source revisions, including missing-report failures.

| Case | Source | Outcome |
| --- | --- | --- |
| `standing-v1` | `bd09473` | Video saved, result report missing; launcher zero is not acceptance. Exact failure lacked traceback. |
| `standing-v2` | `dd2cf43` | Failed yaw .074433 rad after 2 s settling. |
| `standing-v3` | `be93e9c` | Failed yaw .045683 rad after 10 s settling. |
| `standing-go2-v1` | `4c7ebca` | Startup abort using common .8 m spawn; not a valid policy comparison. |
| `standing-spot-v4`, `standing-go2-v2` | `22bfe17` | Startup aborts after training-height/material correction. |
| `standing-spot-v5` | `c1b3eae` | After one startup restore, failed XY .030279 m / yaw .062681 rad with 10 s settling. |
| `standing-go2-v3` | `c1b3eae` | Startup root-height abort; no movement gate. |
| `standing-spot-v6`, `tracking-spot-v6` | `c1b3eae` | Standing passed at 15 s; movement suite failed; older render-timeline samples retained. |
| `standing-carter-v1` | `61d8716` | Incorrect root-height setup guard, no observation acceptance. |
| `standing-carter-v2` | `2218ed5` | Startup height abort before observation. |
| `geometry-carter-v1` | `f3e6479` | Explicit diagnostic-only, always false; yaw gate also failed. |
| `standing-spot-v7`, `tracking-spot-v7` | `f3e6479` | Final physics-clock standing pass / movement failure above. |

Native float32 vectors are copied into finite Python scalar lists; exceptions
are exported before `SimulationApp.close()` so failures cannot disappear behind
an exit zero. Initial root poses/gains/targets are restored ONCE after physics
warm-up, with pre-restore pose/velocity and DOF ordering saved. No resets occur
during settling, observation or movement. Training initial heights .5 m Spot /
.4 m Go2 and explicit multiply friction/restitution modes are recorded. The
installed Spot wrapper's explicit-USD branch has an uninitialized local, so
its stock default official root is verified instead of patching installed code.

The final source checks stage metadata (1 m/unit, Z-up) and samples the physics
clock. Earlier render-timeline epochs are not rewritten. Carter's restaurant
placement z=.07382 is not its z=0 flat-floor rest convention; audited drive-wheel
cylinders span z~0..0.28 at root zero, radius .14. Carter's explicit root-z gate
(-.02.. .1 m) is separate from Spot's .3.. .8 m; root z is not body clearance or
fall height. Raw caster BBox extents were stale/huge and must never enter control
or be advertised as a completed footprint audit. `--stage geometry` uses broad
root monitoring only for zero-command diagnosis, always reports false and can
never authorize tracking. No acceptance bound was relaxed to obtain movement.

Final source `f3e6479` passed all 3,118 unit tests in 107.37 s, including eleven
focused interface/evaluation tests. Ruff, compilation, CLI help and whitespace
checks passed. The earlier expanded source's 108.79 s test run is preserved
separately. No native process remained running when results were committed.
