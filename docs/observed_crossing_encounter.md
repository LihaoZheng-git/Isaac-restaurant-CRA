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
