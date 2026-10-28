# Separate quadruped feasibility comparison

The user approved a comparison with a laterally moving quadruped, rather than
replacing Carter immediately. The motivation is the control interface: Carter
executes `(v, omega)` through unicycle dynamics; a legged velocity policy may
accept `(vx, vy, omega)`. An arbitrary planar velocity is still an idealized
command until actual tracking is validated. Existing Carter results, source
clocks, CRA uncertainty policy, safety checks and merge holds remain unchanged.
The latest user preference is to demonstrate passing in response to observed
approaching motion and a predicted encounter, rather than carrying the separate
quiet-person early-preparation policy into the preferred demonstration.

## Candidate and provenance

The installed Isaac Sim is `6.0.1-rc.7+release.42383.32955d8d.gl`, with
`isaacsim.robot.policy.examples` version 5.2.12. Its stock
`SpotFlatTerrainPolicy` observes 48 values and accepts body-frame
`(vx, vy, wz)`. It outputs twelve joint-position targets. The official matching
configuration specifies physics dt 0.002 s, decimation 10, action scale 0.2,
and flat-ground friction 1.0. No environment upgrade or training is used.

Downloaded unmodified files from the configured official asset root
`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0`:

| File | SHA-256 |
| --- | --- |
| `Isaac/Samples/Policies/Spot_Policies/spot_policy.pt` | `d95d675e8b080d5c84a2a2779d165ff7ecd3e34d24b233c083242e615a3d589a` |
| `Isaac/Samples/Policies/Spot_Policies/spot_env.yaml` | `397440818604ca7cf0cb79c6012325f2bba292029ad0c2d5df5fd64b531d2dbc` |
| `Isaac/Robots/BostonDynamics/spot/spot.usd` | `dbe17178d8507aa31a82e8f8582606c2da5694ea8d2e24c77e50f1ff149689cb` |

The local manifest is under `outputs/quadruped_feasibility/asset-audit-v1/`.
Runtime verifies the root USD against these actual downloaded bytes. The stock
Python source headers specify Apache-2.0, while its package license notice is
proprietary. Do not describe all weights/assets as Apache-2.0 or redistribute
them. This bench imports the installed wrapper and uses NVIDIA assets locally
under the existing Isaac installation's terms; downloaded assets stay ignored.
See the [official example](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/robot_simulation/ext_isaacsim_robot_policy_example.html)
and [additional materials terms](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/).

The installed Spot wrapper has an uninitialized `is_newton` local in its
explicit-USD-path branch. Use its stock default official USD path with verified
bytes, not an installed-source patch. Policy/config paths point to the local
audited originals. Compatibility is not established merely by finding files.

## Gates and comparison scope

`scripts/demos/quadruped_velocity_isaac.py` runs a separate open-space bench.
Its external camera and root-state samples are evaluation information only;
they are not the robot's perception system or CRA inputs. First settle for 2 s,
then observe at least 3 s at zero velocity commands, requiring <=2 cm root XY
drift, <=0.03 rad yaw drift, <=10 degree tilt and base height 0.3..0.8 m.
This is a freely standing robot, not a fixed mount.

Tracking requires an independent passed standing report with matching asset
and source hashes and repeats its own standing gate before movement. Four
bounded body-command phases exercise forward `(0.2,0,0)`, lateral `(0,0.15,0)`,
yaw `(0,0,0.3)` and combined `(0.15,0.1,0)` commands. Every phase is followed by
zero commands. Settled tracking RMSE limits are 0.08 m/s and 0.15 rad/s; these
are explicit engineering acceptance limits, not calibrated error bounds. The
2 cm stopping gate includes braking displacement from the first zero-command
sample, not just late settled drift. Whole phases retain upright/height checks.
Zero command is a gait-policy command, not a promise of zero physical motion.

`fixed_to_body_velocity` rotates fixed-frame planar commands using current
body yaw. `bounded_body_velocity` rejects commands beyond a 0.25 m/s translation
norm or 0.8 rad/s yaw rate; silent clipping would alter a safety-checked command.
These helpers are not integrated into runtime Safe-SAGE and cannot command
Carter's wheels.

Only after locomotion gates pass can new body-mounted RGB/LiDAR sensing be
validated while standing and then walking. A human-passing comparison also
needs a conservative measured/configured swept footprint, tracking-error
treatment and the same CRA/MPC/analytical/geometry/continuity/arrival/stop gates.
Avoidance must remain online with an obstacle-unaware waypoint nominal. Early
preparation/guidance-tracking ablations should not be silently imported into a
paper-style baseline. Changing platforms is not a causal CRA comparison;
matched baseline/CRA evaluation would need to run on the selected same platform.

Native results pending. All 3,114 unit tests passed in 106.77 s, including seven
focused interface/evaluation tests. Ruff, compilation and CLI help passed.
No complete perception, navigation, smoother-passing or platform-replacement
acceptance is established yet. No main/PI-demo/long-service/PR117 release.

The first native standing run at `bd09473` saved images/video but no result
report. Its wrapper exited zero; this is explicitly **not a passed gate**.
Preserve `standing-v1` and its sibling launch/invocation files. Native
float32 velocity components need strict-JSON conversion; the follow-up adds a
typed observation copy and a strict JSON regression, plus exception export
before `SimulationApp.close()` so a future failure remains visible. The first
run's missing traceback does not independently establish its exact failure.
