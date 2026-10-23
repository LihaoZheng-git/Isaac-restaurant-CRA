# Live checked-wait field admission

This is a separate opt-in controller-policy experiment, following the ORACLE
diagnosis in [late_crossing_wait.md](late_crossing_wait.md). It does not integrate
the virtual checked rear-reference search into native control.

`--checked-wait-transition` requires the encounter-guidance/comparison mode.
Physical CRA, fixed-lambda primary comparisons, body geometry, occupancy,
obstacle-unaware nominal, original source epochs, command leases and all exact
MPC/analytical acceptance gates remain unchanged.

For a currently selected perpendicular human, the worker constructs the proposed
lambda-zero field and its actual temporal derivative. It checks a full stopped
horizon from the acquisition pose: exact DTCBF, dense geometry, future analytical
inequalities and unchanged first-command CBF acceptance. Rejection rebuilds a
normal lambda-1.25 field against the **same previous accepted field and derivative**.
That alternative is not an accepted command; original MPC and analytical checks
still apply. Rejected waits are not counted as intentional yields.

The asynchronous worker stages tracking, evidence and field assignments. Only
after result freshness, current-pose stopped-horizon admission (for a selected
wait) and the existing positive-geometry coverage guard does the simulation
thread accept that worker state and publish the field. A stale result is
discarded. A current-pose wait rejection stops the bounded trial, retaining an
unpublished diagnostic; it never silently publishes a stale-pose wait. The
existing unpublished temporal-gap recovery baseline remains separate and cannot
supply a control field. Shared model image caches are not accepted evidence.

Source-pose checks use the packet's declared acquisition pose transformed into
the fixed field anchor, including yaw. Those pose references retain their
existing oracle provenance. An independent checker with the same horizon,
padding, footprint, actuator limits and gamma avoids sharing the planner's
mutable optimization state. No actor phase, authored direction, true position
or desired phi enters this decision.

`checked_wait_transition` records original selection/current source clocks and
the selected field policy. Accepted live waits additionally record a
`checked_wait_current_pose_admission`. The independent policy scorer permits a
normal field during crossing only with the rejected-candidate audit. It still
requires the existing crossing/parallel observations and intentional-wait,
arrival, stopping, clearance, identity and containment gates. The old strict
lambda-zero variant is preserved separately.

Validation status and native results will be appended after the frozen-source
stationary gate and bounded crossing experiment. No native acceptance is claimed
by the implementation alone. Ideal zero-command prediction does not model all
braking inertia; native stopping/clearance remains an independent gate. This is
not exact paper equivalence, real-time acceptance or complete fresh four-factor
perception. Existing main, long-service and PR #117 holds remain unchanged.
