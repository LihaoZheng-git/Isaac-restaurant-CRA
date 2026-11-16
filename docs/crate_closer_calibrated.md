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

## First placement result and bounded follow-up

At frozen revision `2ab49e4`, the 2 cm closer placement passed independent60,
own60 and native arrival/stopping: 1/1 goal in 27.50 simulated seconds,
9.5883 cm goal error, 2.6480 mm braking drift and .132738 minimum CBF margin.
The bend was 8.1085 cm, slightly smaller than the baseline's 8.6657 cm. Padded
crate-box clearance fell to 1.1414 cm (330 recorded poses below the historical
2 cm diagnostic); the enclosing-circle minimum was -6.5474 cm. PhysX recorded
zero crate contacts and no nonfloor headers, with floor positive controls.
These separate geometry results are preserved, not relabeled by native arrival.

Source/point and boundary/admission audits pass, but continuous crate retention
does not: 79/270 fields qualify, with median phi .490817 versus .519938 in the
baseline. The independent parked run retains one sensor-epoch rejection. All
native sources remained frozen. The first 48 focused tests and Ruff passed;
two initial CLI allowlist test failures were corrected before the native run.
Detailed results and the full-path figure are in
`outputs/crate-closer-calibrated/native-audit-v1/`.

This small shift reduces clearance without establishing a more visible curve.
The next bounded placement uses the existing original scene at (17.50,3.45),
5 cm closer than the calibrated baseline, via
`configs/restaurant_crate_original_calibrated.json`. It changes only the scene
path, with the same calibrated B=1.7 control configuration and fresh independent60
and own60 before one driven trial. The prior y=3.45 uncalibrated return-stall
remains a failure; this is a separate calibrated comparison, not a repeat
relabeled as success. Native validation of this follow-up is pending.

The read-only layout review found that stools repeat along the whole short
route. Moving the crate along x does not clearly remove the return constraint.
An opposite-side placement would be only 4.28 cm from the stool boxes and adds
association and field-side uncertainty, so it is deferred. The original y=3.45
position has no surveyed furniture-box overlap and plausible initial camera
projection, but neither proves sensor acceptance or native feasibility. See
`outputs/crate-closer-calibrated/roomier-placement-review-v3.json`.

## Original-position calibrated result

The separately calibrated y=3.45 trial at `561b5d9` passed both fresh 60-source
parked checks and native arrival/stopping. It reached the 4 m goal in 26.6833
simulated seconds with 9.6478 cm final error and 2.7552 mm braking drift. The
alongside bend was **19.9544 cm**, versus 8.6657 cm for the calibrated y=3.40
baseline. The entire padded chassis passed the crate and returned to the goal.
Minimum padded crate-box clearance was 4.6490 cm, minimum CBF margin .089971,
and PhysX recorded zero crate contacts in parked, navigation and braking phases.
The historical enclosing-circle score remains a separate FAIL: 1.9742 cm versus
its original 2 cm diagnostic. It did not veto the sensor-only controller.

Source/point/boundary audits pass and frozen sources are unchanged. Original
crate evidence qualified on 63/262 fields, so continuous property retention
still fails. Median retained phi was .406991; no score or boundary override was
used. Separate live observations differ across placements; this is one trial
per position, not a causal risk-only comparison or repeatability result. Both
earlier calibrated placements and the old uncalibrated return-stall remain.
Full-path comparison and audits are in
`outputs/crate-closer-calibrated/original-audit-v1/`. The separate H264 preview
in `outputs/crate-closer-calibrated/preview-original-v1/` preserves all 345 native
frames, including parked checks and braking; full decode and source hashes pass.

The user now authorizes a closer-bin short test and, if both props produce a
useful detour with arrival, a separately saved full restaurant demo using the
selected positions. Full service uses N6/guidance100 rather than the isolated
N12 controller and therefore requires its own fresh gates and full mission.
Existing V5, failures and main/primary/PR117 holds remain protected.
