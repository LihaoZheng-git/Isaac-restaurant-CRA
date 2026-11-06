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
