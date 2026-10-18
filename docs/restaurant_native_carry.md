# Native carried-bowl development diagnostic

The official Isaac adult can pick up and carry the existing audited CC0 Kenney
`bowl-soup` mesh using `pickup_object(..., snap_to_hand=True)`. This is native
hand attachment, not a bowl manually translated alongside a person. The first
four zero-wheel trials passed the attachment/geometry subcomponent but failed
the complete actor arrival/stopping gate. **Carrying navigation is not ready;
keep this branch draft.** The separate standing/slow-crossing navigation step
in PR #130 is already merged into development; the main hold is unchanged.

## What the diagnostic checks

`restaurant_human_carry.py` references the existing local converted asset in the
session layer, checks its original asset audit and records its fingerprint.
The chosen width is .30 m. A kinematic rigid-body/convex-hull collision proxy
supports native pickup; the visible cavity/food mesh remains intact. No mass
API, measured weight, temperature, fall height or risk profile is authored.
The bowl's initial pose is fixture setup only. The native snap bypasses grasp
planning and table pickup; no fluid or spill dynamics are simulated.

The observer reads actual native hand transforms, native held-object state and
PhysX tensor prop poses. It requires a stationary attachment gate before the
actor walks, bounded hand-relative position/orientation, an upright/raised
prop, positive independent authored-layout clearance and actual prop motion.
All robot wheel commands remain zero. These oracle joint/physics references
are evaluation only, never CRA estimation inputs. Source scene and asset bytes
are checked independently. The camera is a diagnostic view, not a new robot
sensor or evidence source.

`--carry-bowl` is limited to the .5 m/s actor path from the existing native
start to `(29.8,10.8,0)`. It is intentionally absent from the wheel-navigation
runner. A carried bowl has **no validated separate sensor ownership** yet.
An attached bowl, its asset name or its authored elevation never supplies
thermal, potential or mass evidence.

## Actual outcomes

| Trial | Native stop command | Complete result |
| --- | --- | --- |
| `native-carry-v1` | Idle after MoveTo success | Failed: .570080 m post-task root drift and final endpoint miss; attachment passed |
| `native-carry-v2` | Zero preferred speed plus Idle | Same failed root trajectory; attachment passed |
| `native-carry-v3` | Zero speed, explicit current-facing Idle, longer observation | Failed: .617287 m root drift; attachment passed; no imported custom actions were available |
| `native-carry-v4` | Keep auto-braked MoveTo terminal state, zero speed, no new Idle | Actual final arrival passed, strict stopping failed (.298263 m root drift versus unchanged .03 m limit); attachment passed |

In v4, maximum hand-relative position error was .000677 m, maximum bowl tilt
14.6547 degrees and minimum independent bowl/layout clearance .423205 m. The
native API reported zero linear velocity and zero preferred speed while actual
root/collider/prop poses still moved. Task success or that API velocity is not
physical stopping acceptance. The original failed reports remain unchanged.
No teleport, animation freeze, looser stopping threshold or environment upgrade
was used to manufacture a pass.

The actual image/video under `outputs/restaurant_service/native-carry-v4/`
shows working native carrying and the unresolved endpoint transition. The
12 new unit cases cover session-only source/collision authoring, true prop
motion, attachment failures, invalid/repeated observations, posture and layout
rejections. They do not replace Isaac execution.

## Next bounded gate

The additional `native-carry-sensor-v1` run uses the existing parked-robot
RGB/LiDAR worker and unchanged full-instance gates. Its result is separate
from attachment and stopping. Its human benchmark passed: three stationary
and six steady-moving fresh visually associated adult mass priors, .013513 m/s
median moving velocity error, four numeric policy factors and no property
borrowing. All 127 video frames were distinct, and source scenes/assets stayed
unchanged. However, overall stopping still failed (.395015 m root drift), and
separate bowl ownership remains unvalidated. Maximum result completion age
was .416667 s, exceeding the live control runner's .35 s cap; this is not a
freshness/latency acceptance result for carrying robot navigation.

The current attachment summarizer also rejects nonfinite prop poses and names
its terminal observation explicitly: a passed attachment check never means
the actor passed stopping. Current unit coverage includes 12 new cases.
The full regression passed **1,880 unit tests**; repository Ruff and whitespace
checks passed. No environment, dependencies or model weights were changed.

Before robot carrying interactions, settle the native endpoint behavior or
explicitly select a continuously walking server scenario with its own bounded
trajectory acceptance. Do not relabel a failed stop as a pass. Then validate
fresh body/bowl ownership and retained physical occupancy with zero wheels.
Potential needs justified support/landing evidence; a hand height is not fall
height. Visible plume is a separate cue method, not soup identity/temperature.
The moving-view plume gap, factor-confidence/scale study and matched navigation
baselines remain separate work in `restaurant_final_experiment_plan.md`.
