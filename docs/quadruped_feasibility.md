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

At `dd2cf43`, `standing-v2` exported a valid **failed** report: 161 zero-command
observations, XY drift 0.005832 m, yaw drift 0.074433 rad (>0.03), tilt
0.004613 rad. No locomotion trial followed. Per-second median yaw rate decayed
from about -0.047 rad/s during second 1 to -0.0165 during second 5. The next
separate retry may use explicit `--settle-seconds 10` before the unchanged
3.2 s observation and acceptance gates. The legacy 2 s default stays. This
tests an initialization transient, not relaxed tolerances or a passed old run.
Tracking requires the same explicit settling period as its independent gate.

`standing-v3` at `be93e9c` also failed yaw after 10 s settling: 160 samples,
XY drift 0.003892 m, yaw drift 0.045683 rad, tilt 0.003937 rad. Do not relax the
0.03 rad limit or send locomotion commands based on this result. The second
installed candidate is `Go2FlatTerrainPolicy`, also with 48 observations and
body `(vx,vy,wz)`. Its matching official PhysX configuration uses dt 0.005 s
and decimation 4 (200 Hz physics / 50 Hz policy), preserved separately from
Spot's timing. An explicit `--robot go2` must match its own manifest and
independent standing report. Sampling remains 50 Hz on either candidate.

The Go2 audit is `outputs/quadruped_feasibility/go2-asset-audit-v1/`:

- `physx_policy.pt`: `984c802b8acd68c3199605ff36603f91ac9f3192eb1ebcfd80b0e764831abe0f`
- `physx_env.yaml`: `da282802fc3bec2f4d1a80818276680b91f73a716e26069bfee52e3fdf4d19af`
- `go2.usda`: `29405c5d2a29b1a80e268df5de23898cb8f48ce95646352c25aeca512bfcb3df`

These are unmodified NVIDIA asset-root files; the root USD references further
Menagerie payloads. Keep used-layer identities in native reports and do not
describe this root-file audit as a complete dependency/texture license audit.
Neither candidate has sensor or navigation acceptance yet.

The first Go2 run (`standing-go2-v1`, source `4c7ebca`) stopped during startup
at base height 0.248827 m, before a standing observation window. This reused
the Spot example's 0.8 m spawn height; the audited Go2 training initial height
is 0.4 m (Spot's is 0.5 m). Do not call that attempt a stationary-policy
comparison or relax its abort threshold. The next source uses each policy's
audited training initial height and verifies it from the loaded configuration.
It also explicitly matches the documented multiply friction/restitution
combine modes, instead of leaving PhysX's default combine mode implicit.
Both setup changes are recorded; old results remain failed and unchanged.

At `22bfe17`, corrected-height Go2 and Spot retries stopped during startup,
before standing windows. Go2 reached 0.244596 m height and -0.466726 rad roll;
Spot reached 0.454342 rad roll after its body was pushed upward to about 0.88 m.
The stock first-post-physics initialization lets warm-up advance the authored
posture before new joint positions/gains/targets are installed. The next
separate setup restores the audited initial root pose/zero velocity ONCE after
warm-up, installs initial joint targets and executes the first zero-command
policy update before the next tick. `initialization.json` preserves the actual
pre-restore state and DOF ordering. No resets occur during the settling,
standing or movement phases. This is an initialization hypothesis/retry, not a
retroactive pass, mount, body-motion override or alteration of acceptance gates.

## First valid Spot interface result, before the Carter comparison

At `c1b3eae`, Spot's 10 s retry (`standing-spot-v5`) completed but failed XY
0.030279 m and yaw 0.062681 rad. The terminal yaw rate had fallen to about
-0.00166 rad/s. A separately labeled 15 s settling trial (`standing-spot-v6`)
then **passed** all unchanged gates: 160 samples, XY drift 0.000100 m, yaw drift
0.000114 rad, maximum tilt 0.034062 rad. No source/evidence clock is renewed by
this initial settling. It precedes any movement assessment.

The independent and own standing gates passed before `tracking-spot-v6`.
That complete open-space suite **failed** low-speed translation tracking:
forward/lateral/combined RMSE 0.201330 / 0.149853 / 0.179669 m/s at commands
0.2 forward / 0.15 lateral / (0.15,0.1) combined. Corresponding settled median
body translation was close to zero. Yaw tracking at 0.3 rad/s passed
(RMSE 0.036255 rad/s). Zero-command displacement after forward/lateral/combined
was 0.075126 / 0.042998 / 0.063994 m, exceeding the unchanged 2 cm stopping
gate; yaw-stop translation 0.019379 m passed. This is evidence about this
policy/setup at these low commands, not every quadruped or a proven internal
deadband cause. No commanded translational speed was increased to get a pass.

Go2 with the one-time restore (`standing-go2-v3`) still crossed the startup
root-height abort at 0.249937 m; it has no accepted standing or movement gate.
The common quadruped height envelope itself is not a calibrated Go2 fall
detector. Do not infer general Go2 instability from this failed bounded setup.

The next source adds Carter on the same flat bench, using the unchanged
`NovaCarterController`, 60 Hz physics, 0.07382 m configured initial root height,
and only its executable forward/yaw phases. Lateral and combined phases are
explicitly unsupported, not projected or falsely passed. Its different root
origin requires explicit 0.03..0.2 m height bounds; this does not change Spot's
0.3..0.8 m bounds, 2 cm XY/stop, 0.03 rad yaw or tracking gates. This is an
interface calibration, not matched human navigation or a causal CRA comparison.
The official Carter root SHA-256 is
`462c3e91703c2f3aead2f15520a76c026bca1327fa61c40b268688d8711f938f`.

New samples and phase switching use `SimulationManager.get_simulation_time()`
in the physics callback. Previous bench samples used the rendering timeline,
which can label multiple physics ticks with one render epoch. Earlier Spot
50 Hz samples had no duplicate epochs, but the new clock is necessary for a
60 Hz Carter comparison. Keep the older records unchanged and repeat matching
independent/own standing gates at the new source before movement.

The expanded comparison source passed all 3,118 unit tests in 108.79 s,
including eleven focused interface/evaluation tests. Ruff, compilation,
CLI help and whitespace checks passed before its native source freeze.

The first Carter attempt (`standing-carter-v1`, `61d8716`) stopped on its
incorrect height guard before observation. The zero-motion geometry audit
confirms the new stage is 1 m/unit and Z-up. Carter's authored drive-wheel
collision cylinders span approximately z=0..0.28 at root z=0: its wheel radius
is 0.14 m. The restaurant's 0.07382 m initial placement is not the root resting
height on this separately authored z=0 flat floor. The corrected bench starts
Carter at root z=0 and explicitly checks root z in -0.02..0.1 m; Spot's bounds
and every XY/yaw/tracking/stopping limit stay unchanged. Root z is a frame
convention, not physical body clearance or fall height. Runtime checks stage
units/up-axis and the audited Carter drive-wheel radius before moving.

The audit also found stale enormous caster-cylinder BBox extents; its raw
all-collider minimum is invalid as a physical robot envelope. Do not feed that
diagnostic into control or claim a completed swept-footprint audit. The actual
drive-wheel cylinders and configured radius establish only this flat-ground
root convention. Previous failed reports are preserved, not re-scored.
