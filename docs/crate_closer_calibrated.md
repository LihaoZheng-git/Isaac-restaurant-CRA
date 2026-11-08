# Calibrated closer-crate placement

The user requested a closer obstacle placement to test a more noticeable online
detour. This separate trial moves the crate's mesh-bounds center from world
(17.50, 3.40) to (17.50, 3.42): 2 cm closer to the unchanged route at y=4.05.
It remains 3 cm farther from the route than the earlier y=3.45 stress case,
which bent substantially but did not return to the goal past the sensed stools.

The recipe `configs/restaurant_crate_closer_calibrated.json` derives from the
B=1.7 calibrated boundary baseline, changing only the scene path. Keep the
scoring function and palette, mass/contact evidence rules, live motion,
potential/thermal priors, fixed lambda=1.25, N12 unicycle MPC, analytical CBF,
straight 4 m nominal route, physical limits and all existing gates. Stronger
B=2.9 is not selected. The N12 isolated controller differs from full V3's
N6/guidance-tracking variant.

Build the separate additive layer using
`scripts/scene_creation/build_crate_route_probe.py --setback-cm 3` with the
existing local licensed assets. The builder refuses to overwrite any existing
scene or manifest and verifies all source hashes. The near crate edge becomes
y=3.782094, leaving 26.79 cm from the nominal route center. Carter's padded
half-width is 29.66 cm, so the straight zero-yaw padded footprint overlaps the
crate box by about 2.87 cm. This is an evaluation-only geometric motivation,
not a runtime obstacle input or proof of a feasible detour. Runtime surroundings
still come from camera/LiDAR, with simulated ego localization and the existing
static occupancy prior.

Run a fresh independent60 parked gate and then the motion stage's own60 gate
before one driven trial. Use `scripts/evaluation/run_restaurant_demo.py`, case
`crate-static`, the new recipe and fresh output paths. The old scene's parked
proof cannot qualify this scene. Record actual bend beside the crate, waypoint
arrival, stopping drift, sensor-field CBF, padded-box clearance, historical
circle metric, passive PhysX contacts and qualified retained-evidence coverage.
Keep the circle score evaluation-only; do not introduce an authored-world veto.

Compare against the completed B=1.7 baseline at `838c3ec`: 8.6657 cm bend,
29.3833 simulated seconds, 1/1 goal, 9.6643 cm goal error, 2.6615 mm stop drift,
3.7116 cm padded-box clearance and zero recorded crate contacts. Its historical
circle score and continuous crate retention check failed separately (78/282
qualified fields). A new scene trial establishes placement sensitivity only;
fresh observations differ, so it is not a causal CRA-versus-fixed-risk result.

Native validation is pending at this protocol snapshot. Preserve every old
placement, failure and V3/V4/V5 demo. No full-demo replacement or main/primary/
PR117 promotion follows from a larger curve without arrival and audits.
