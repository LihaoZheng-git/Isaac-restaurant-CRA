# Live crossing direction and onset coverage

This bounded follow-up to [live_checked_wait.md](live_checked_wait.md) keeps the
accepted checked-wait controller, physical CRA, priors, footprint, body envelope,
fixed primary lambda, source clocks and strict waypoint/stopping gates intact.
It does not add a native checked rear-reference or a new nominal planner.

Two additional native cases are planned, separately from the existing
left-to-right/10 s accepted reference:

| Case | Authored actor fixture change | Required acceptance |
| --- | --- | --- |
| Mirrored crossing | Right-to-left at 10 s, departing north in the left aisle | Parked moving-actor component first; own 60-frame stationary gate; full live arrival/stopping/source/policy/containment audit |
| Later onset | Original left-to-right fixture at 11.5 s | Own 60-frame stationary gate; full live arrival/stopping/source/policy/containment audit |

Direction and timing are scene/evaluation information only. The runtime policy
continues to use observed LiDAR velocity, current validated human identity and
the obstacle-unaware robot route direction. Authored motion never supplies
kinetic evidence, a preferred passing side, factor scores or phi.

The mirrored departure targets are `(27.8, 8.9)` then `(27.8, 12.5)`, reflecting
the original targets about the robot's northbound route at x=28.8. This changes
only the actor fixture and independent direction-aware scoring. Original
left-to-right defaults, original right-to-left stop-at-endpoint mode, geometry
and visual evidence gates are preserved. Native path following/articulation and
independent raster clearance must validate the new leg before robot motion.

The legacy closest-encounter report retains its historical left-excursion
diagnostic. The selected wait-and-go acceptance already scores the actual
crossing-plane projection using the independently observed walking direction
and excludes the curved-arc criterion. A mirrored trial cannot acquire a pass
by changing the required arrival, waiting, clock, identity or geometry criteria.

All failures and the earlier accepted case remain separate records. One
direction/timing pair does not establish crowded tracking, all onset combinations,
real time, calibrated human factors, causal CRA advantage or full fresh
four-factor perception. Existing main, long-service and PR #117 holds remain.

Validation and native results are pending the source freeze and component gate.
