# Observed-only crossing encounter diagnostic

The 11-second observed-only retry completed the route but failed curved-bypass
acceptance. During supported native walking, the robot was at world
y=7.5336--7.6747 m, the actor crossed near y=8.9 m and their closest moving
center separation was 1.300220 m. See [the complete preserved outcome and
profiling correction](runtime_profiling.md). Arrival is not a passed bypass;
the negative walking arc and positive behind projection remain failed.

## Separately declared 20-second fixture

Before launching this diagnostic, choose a fixed elapsed-navigation crossing
delay of 20 seconds rather than 11. The completed prior trace places the robot
farther along the route around that time. This is an authored scenario choice,
never runtime evidence or a progress-dependent actor trigger. It is a new
development fixture, not a repair or matched-input replay of the old trial.
Whether the controller actually produces a closer, valid encounter is measured
afterward; the new timing does not guarantee acceptance.

Keep every other invocation argument unchanged: observed occupancy without
added circular control cells, human-only b=-1.7 applied policy override,
unchanged computed CRA and nonhuman policies, lambda 1.25, the separate weight
100 planar-forecast guidance-tracking ablation, forward-only actuator limits,
QP 1000/rescue 20, 90 simulated seconds, .025 pacing, original 12 s progress
watchdog and all source/identity/field/execution clocks.

The new trial must pass its own 60-frame zero-wheel human gate before motion.
Retain strict missing-fragment continuity, independent .02 m clearance abort
including braking, full exact MPC/DTCBF/dense geometry and per-step analytical
filtering. Score actual crossing, supported current human motion, original
selection identity/time, >=.20 m walking-window trailing arc, <=-.10 m behind
projection, all three waypoint arrivals/rejoin and stopping with unchanged
thresholds. A shadow circle audit is not observed-only coverage acceptance.

Runtime source remains the validated profiling revision 49f56f2; no new control,
perception, physics or safety implementation is introduced by this timing
diagnostic. All 2,297 unit tests/Ruff and trusted original-input planner parity
passed for that source; these are not a new native result. Preserve the new
invocation, full report, JSONL, actor trace and video before further diagnosis.

The user's latest standing authorization covers bounded guarded diagnosis and
retries. Main and long-service remain held, and reverse motion/wider uncertainty
bounds are not authorized by this scenario change.

## 20-second result at 7313228: failure preserved

`isolated-human17-observed-motion-v3` passed its own 60-frame stationary gate,
then failed `nearby_track_lost` at 21.8 navigation seconds: 1/3 waypoints,
.155888 m minimum independent clearance, .742975 minimum CBF margin and
.000908 m stop drift. It published 212 fields (177 translating), with 80
all-track visual instances and zero complete four-factor assessments. The
unpublished stop field was not committed for control. The native launcher
returned zero despite the full-gate exception; the saved failed report governs.

Fragment 617 was last observed at source 29.6 s and missing at source 29.8 s,
.954215 m from the robot. The current human track 75 remained observed. No
body-prior coverage was enabled. The unchanged .15 s loss threshold stopped
the trial before the .02 m clearance abort. Human diagnostics retained four
supported kinetic rows/one simultaneous row, but full crossing and the required
five kinetic rows failed because the trial stopped early. Walking arc -.005633 m
failed; behind projection -.738769 m alone passed. Arrival/rejoin failed.

The original-centroid query had zero current returns within .10 m; its nearest
current parent return was .112315 m away. The bounded velocity query also had
zero returns within .10 m (nearest .148421 m) and exceeded the unchanged .25 m
centroid gate. Do not widen either threshold or infer that missing geometry is
free. Original fragment surface points were not retained by this watchdog;
their exact overlap with current returns cannot be reconstructed from these
centroid-only records. Preserve this gap before implementing stronger measured
surface diagnostics/association. Current raw XY/IDs and the unpublished field
remain in the output directory.

This is a failed closer encounter, not a passed bypass or a causal comparison
against the earlier 11-second case. Total launcher time was 1,109.569 s.
