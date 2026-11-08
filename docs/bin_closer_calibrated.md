# Calibrated closer-bin pass

The user requested a more visible bin/crate detour before a separately saved
full restaurant demo. The calibrated crate at world (17.50,3.45) has completed
its isolated test with a 19.95 cm bend and arrival; see
`docs/crate_closer_calibrated.md`. This bounded next test moves only the rounded
bin to (17.50,3.65), 25 cm closer to the same y=4.05 straight waypoint route
than the previous matched-bin position. The smaller bin needs a different
center offset to put its near surface similarly close to the robot's route.

Build the new additive scene with
`scripts/scene_creation/build_crate_route_probe.py --matched-bin --setback-cm 5 --closer-bin`.
The builder preserves old scenes, mesh size, ground contact and source hashes.
`configs/restaurant_bin_closer_calibrated.json` copies the existing calibrated
B=1.7 bin command, changing only the scene path. Keep N12 unicycle MPC,
analytical CBF, lambda=1.25, calibrated scoring/colors, property evidence rules,
live kinetic factor, priors, 4 m nominal goal, 45 s navigation duration and
physical limits. Runtime obstacle information remains sensor-derived; scene
geometry and PhysX contacts only score the result.

The bin measures approximately .296 x .296 x .500 m. Its near edge is y=3.798,
about 1.41 cm farther from the route than the selected crate's near edge.
This gives approximately 4.46 cm overlap with the padded straight robot
footprint. The surveyed scene has no furniture-box overlap at this placement.
These authored-geometry checks motivate a test; they do not establish native
perception acceptance or a feasible path.

Require a fresh independent 60-source parked check, the motion stage's own60
check, then one driven trial. Report full chassis passage, lateral bend,
goal arrival/error, braking drift, sensor-field CBF, padded-box and historical
circle clearances, passive contacts, source/point/boundary audits and retained
evidence coverage separately. The old y=3.40 bin pass used uncalibrated scoring,
so it is not a matched calibrated native baseline. Do not infer risk-only
causality from different objects, geometry and fresh evidence.

If this test produces a useful detour and arrives, the proposed new full-demo
bin position is (21.20,3.65), keeping its along-route location separate from
the crate at (17.50,3.45). The full N6/guidance100 controller differs from this
isolated N12 setup and needs all fresh prerequisites and a new full mission.
Preserve accepted V5 and every earlier failure; main/primary/PR117 holds stay.
Native bin validation is pending at this protocol snapshot.

Implementation checks: 54 focused tests pass (7.23 s), repository Ruff and
diff checks pass. The first pytest attempt had 44 passes and ten sandbox TEMP
setup errors; using a fresh workspace basetemp resolved setup without code or
assertion changes. Independent scene composition verifies all 135 source
hashes, exact +25 cm bin translation, unchanged dimensions/height and inactive
crate. Standalone USD cannot resolve the remote Carter reference; native Isaac
parked validation is still required before motion.

## Native result

At frozen `1f9382c`, independent60 and own60 passed, followed by native arrival
and stopping. The bin produced a **20.8074 cm** alongside bend and the complete
padded chassis passed it. Goal arrival took 32.7833 simulated seconds, with
9.7171 cm final error and 2.6279 mm braking drift. Padded bin-box clearance was
10.7139 cm; historical enclosing-circle clearance was 4.6825 cm, passing its
original 2 cm diagnostic. Minimum sensor-field CBF margin was .080004.
PhysX recorded zero bin/nonfloor contacts with floor positive controls in all
three phases. Source freeze, source/point and boundary audits pass.

Original bin mass/contact qualified on 201/315 fields; continuous retention
still fails. The retained rough mass was .455366 kg and outline cue .0860923,
with live calibrated contact .03068--.03891 and phi .27414--.28259 (median
.277342). This is lower than the separate closer-crate trial's median .406991,
but different geometry and fresh observations prevent a risk-only causal
trajectory comparison. The bin's slightly larger bend is not evidence of a
higher assessed risk. Potential/thermal remain priors and mass is uncalibrated.

The approach/return included pauses and slow recovery. Saved command windows
show new-field plan invalidation/waiting, then a low-speed recovered MPC seed;
this is not a pause-free or smoothing result. No controller, score, time-limit
or safety-check change was made to complete the run. Audits and full-path plot:
`outputs/crate-closer-calibrated/bin-native-audit-v1/`. The separate H264 preview
in `preview-bin-v1/` preserves all 406 native frames including parked/braking
and pauses; frame/source checks and full decode pass.

Both bounded placements have now produced visible detours and reached the goal.
Proceed to a new additive full scene and calibrated recipe with the crate at
(17.50,3.45) and bin at (21.20,3.65), subject to the existing full-service fresh
prerequisites, mission and source/geometry/evidence/video audits. Preserve V5.
