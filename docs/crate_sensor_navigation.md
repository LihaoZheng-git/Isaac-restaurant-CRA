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
