# Sensor-controlled crate trial

The question is whether Carter can navigate a straight 4 m service waypoint
beside the broken crate using its camera, LiDAR, continuous CRA and Safe-SAGE.
The independent clearance measurement previously ended two closer-crate runs.
For this experiment the wheel loop records the measured robot poses. After
navigation, the evaluator computes the original disk and layout clearances,
including the first sampled crossing of 2 cm. A separate PhysX observer
reports robot/crate contacts throughout the run. Its event counts have no
connection to the controller.

The primary recipe is `configs/restaurant_crate_sensor_navigation.json`.
It uses the existing six-step unicycle MPC, physical wheel commands, analytical
CBF and fixed field lambda 1.25. The earlier controller added a guidance-tracking
objective and, in another variant, a longer search over turning commands.
Neither is enabled in this primary recipe. The same sensor mounting, sensor
inference, labeled factor priors, 5 cm field grid, straight route, existing
velocity limits, 45 s simulated duration and simulation pacing of 0.025 apply.
Robot localization and LiDAR deskew use the simulator chassis pose. Setup
validation verifies the intended parked fixture using the authored scene.

Observed occupied cells persist under this isolated scene's static prior.
The short recent-scan queue is separate. A gap in fresh, close-range returns
therefore establishes a gap in current sampling. An object-specific mass/contact
record requires its own continuing association.

Run `scripts/evaluation/run_restaurant_demo.py` with case `crate-static`, the
primary recipe, a new output directory and the stationary stage. Use that
source for the motion stage with a second new directory. The motion stage
performs another 60-field parked check. Report goal arrival, stopping drift,
full chassis passage, recorded contact events, and the original 2 cm benchmark
individually. The crate's mesh-derived bounding box is a geometric cross-check.
The PhysX contact observer requires an identifiable contact in each reported
phase as a positive control for interpreting zero crate events.

The six-step native run on revision 17c5ef8 reached its 45 s time limit at
local (1.44055, -0.05901). All 415 live fields were committed with the external
clearance veto disabled. PhysX recorded 10,804 robot/floor contact headers
during navigation and zero robot/crate contacts. Minimum clearance to the
crate alone was 0.007607 m against the circle and 0.038248 m against its
bounding box. The original combined disk/layout score reached -0.004259 m.
The recalled crate cells were still in the safety field.
At source 29.9 s the planner converged to zero. Forward probes intersected
the remembered crate group's padded occupied cells. At 14.9 s both turn
directions had still been available. On that frozen approach field, ideal
closed-loop replays reproduced the stall with six steps and reached goal
tolerance with twelve. The twelve-step case had 0.0500 m minimum padded
chassis-to-crate-box clearance over 1,008 ideal control samples.

The native follow-up uses `configs/restaurant_crate_sensor_navigation_horizon12.json`.
Relative to the six-step sensor protocol the only command difference is MPC's
prediction horizon (12 instead of 6 intervals at 0.2 s). It runs forward from
the original parked start using current sensor fields.

## Native crate encounter and return to the waypoint

Both native recipes passed their independent and in-run stationary checks
(60 fields each, zero wheel commands). The longer horizon cleared the crate
in the original restaurant aisle. Both runs ended at the 45 s simulation
limit with zero of one final waypoints reached.

| Observed quantity | Six steps (1.2 s) | Twelve steps (2.4 s) |
| --- | ---: | ---: |
| Entire padded chassis beyond the far edge of the crate | Never | 22.383 s |
| Closest padded chassis to mesh-derived crate box | 0.03825 m | 0.07105 m |
| Closest padded chassis to crate evaluation circle | 0.00761 m | 0.05258 m |
| Minimum across the original layout and disk benchmarks | −0.00426 m | 0.03350 m |
| PhysX crate-contact reports during navigation or braking | 0 | 0 |
| Maximum lateral departure from the waypoint line | 0.05901 m | 0.42690 m |
| Final distance to waypoint | 2.56013 m | 0.54018 m |

The twelve-step run committed 401 sensor fields, including 303 while
translating. Its minimum recorded analytical CBF margin was +0.049995.
PhysX reported only floor contacts on the robot (10,804 headers while driving
and 240 during braking). The stop drift was 0.00129 m. Independent geometry is
sampled at recorded poses. The crate box encloses the mesh, and the old 2 cm
disk/layout criterion is reported separately from the simulation contacts.

Replaying the original crate record against each new LiDAR source recovered
its mass and contact evidence on 233 of the 401 live fields. The checks found
zero value, clock or point-membership discrepancies on reused evidence. The
original intrinsic estimates were 2.896 kg (rough visual estimate) and 0.519
(outline cue). The CRA score on these associated rows ranged from 0.621 to
0.839 as live interaction and uncertainty changed. All-field evidence retention
failed. The observed bypass alone cannot establish continuous recognition or
the size of CRA's contribution relative to a fixed-risk controller.

The remaining failure occurs on the return to the original waypoint. The
leftward detour puts Carter alongside BarStool_25, BarStool_26 and BarStool_27.
Stored LiDAR returns at the first blocked cells correspond to the stool seats
and include original parked observations. At local (3.53, 0.41) the
obstacle-unaware waypoint controller requested a clockwise turn. The immediate
analytical filter allowed that command, but the MPC's forward check rejected
the padded swept footprint as the rear swung out toward the stools. The
achieved final pose was (4.331, 0.427) versus the (4, 0) objective. The goal
position itself has room for the sampled orientations. On four recorded states,
extending prediction to 20 and 30 intervals did not yield a planned endpoint
even 1 cm closer to the route line. This is a local planning failure in the
sensed aisle. There is no basis here for another external stop or deleting the
observed stool geometry.

There are two relevant implementation details inside the sensor controller.
First, the scheduler holds the robot when it needs a valid current plan or
field. The twelve-step run recorded 164 ticks waiting for a plan, 103 when a
new field invalidated a plan and 14 with a stale field (60 ticks/s). Two
sensor-time alignment attempts were rejected. On the last tick a current
field and plan were available, the waypoint requested (0, −0.6), and the
selected wheel-body command was (0, 0) with no scheduler stop reason.
Second, the MPC checks Carter's padded footprint along its predicted motion
with 0.025 m position and 0.21 rad heading allowances. This is a conservative
sensor-geometry check in the repository's unicycle adaptation. The runtime
uses a 0.05 m occupancy grid and 16 yaw slices. In offline replays of the
2.69 m and 3.53 m route positions, 32 and 64 slices did not admit the
clockwise probes. Checking the continuously oriented rectangle against
the original occupied cell squares also rejected the nominal turn and a
gentle (0.15 m/s, −0.10 rad/s) bend within the original pose allowances.
Coarse orientation can reject individual samples early. These positive
occupancy overlaps further along the maneuver persist with finer geometry.
They measure predicted uncertainty coverage, distinct from actual contact.

An offline scan of the final observed field at zero chassis heading provides
a check on the encounter itself. On the original route line, the padded
footprint is blocked from about 1.275 m through 2.724 m along the route.
Beside the crate at 1.85 m, the sampled allowable left offset was
0.225–0.275 m. The centerline opens again from about 2.725 m to the goal.
These fixed-heading samples use the final accumulated scan. They do not
establish that Carter could reach those poses from its actual heading at the
time, or supply commands to the native run.

As a final feasibility diagnostic, a forward-only search using the **last
frozen sensor field** found an ideal unicycle trajectory from the original
start to within 0.00449 m of the original waypoint. Its peak lateral departure
was 0.12722 m. Traversing this curve in 21 s failed four rolling DTCBF checks.
Slowing only a portion of the approach gave a 41.0 s traversal. All 206
rolling twelve-step windows then passed the existing dense and DTCBF checks
(minimum DTCBF margin +0.00797), and the analytical filter admitted each
checked command. This establishes model feasibility on that field and
motivates testing joint speed and steering selection earlier in online MPC.
The diagnostic had later observations available from the start, used ideal
kinematics and never supplied a trajectory to Isaac. It cannot count as
an online goal-reaching result. The local records are
`final-field-reachability.json` and `final-field-retiming.json` in the
experiment's output directory.

Frozen native sources and videos are under
`F:/CRA-experiments/crate-sensor-navigation/{primary,horizon12}-{stationary,motion}`.
Sampled geometry, evidence replay, maps and occupied-cell provenance are under
`outputs/crate-sensor-navigation/`. These paced Isaac runs use simulated
ego localization and a static occupancy prior. Keep the historical failures
and the released V3 video separate from this controller experiment.

Validation on the final code revision ran all 3,966 unit tests and the
repository Ruff checks. SHA-256 verification matched all 408 files in the
previous demo and experiment preservation manifest. These checks are
independent of the two native waypoint failures.
