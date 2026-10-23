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

## Source and component validation

Frozen source `50165de8ccf04cd6f6e5e1ed85e129188ac0c843` passed 2,623 unit tests
(84.96 s), 89 focused tests (5.94 s), Ruff across robot/tests/scripts,
compilation, diff checks and both CLI help checks. A source/argument audit proves
that the estimator, multi-object worker, checked-wait guard, MPC, CBF, encounter
selector and waypoint controller match the accepted `684f302` reference. All
launch arguments match apart from output, declared direction and onset.

The first parked wrapper precreated a directory that the native script required
to be fresh. The CLI rejected it before Isaac startup; preserve the v1 log and
harness-failure record. The fresh v2 runner uses a nested native output and does
not change the source, environment or acceptance gates.

The mirrored parked diagnostic **passed with zero robot wheel commands**:
actual right-to-left crossing and upper-aisle departure, no return, articulation,
grounding, collider alignment, layout clearance and sensor checks passed.
Median actor speed was .512960 m/s and minimum independent layout clearance
.295257 m. There were 48 fresh visually associated adult mass-prior rows,
including three stationary rows and nine steady moving rows; median moving
velocity error was .134665 m/s. The separate human contact-prior gate passed.
These are configured adult mass/contact priors and sensor motion observations,
not calibrated mass, physical sharpness, temperature or robot-navigation proof.

## Mirrored 10 s native trial

The trial passed its own 60-frame zero-wheel stationary gate. Its **original
overall native report is FAIL** and remains unchanged, despite actual arrival:

| Check | Recorded result |
| --- | --- |
| Actual route / time | 3/3 goals, `goal_reached`, 23.3167 simulation seconds |
| Braking-inclusive monitored clearance / stopping drift | .498531 m / .002304 m |
| Live fields / fields committed during translation | 232 / 211 |
| Source/current-pose admitted wait fields / rejected-normal candidates | 1 / 26 |
| Parallel classifications with normal guidance | 70 |
| Accepted walking-window checked-yield plan sources | **0**, required at least 3 |
| Longest independently observed walking-window wait | 1.183333 s |
| Executed commands joined to accepted checked-yield plans | **0** |
| Independent body containment | 232/232; max center error .166690 m within unchanged .25 m prior |
| Minimum executed analytical margin | .205523 |
| Accepted visuals / object assessments / complete fresh four-factor rows | 91 / 15,270 / **0** |

The independent audit passed original source/frame/selection/pair checks,
current-pose admission, normal-alternative policy mapping, body containment,
actual arrival, stopping, analytical margins and unchanged scene/root layers.
The raw overall report correctly fails crossing-zero observation count and
intentional-yield requirements. Its launcher exits 0 after the script prints an
acceptance exception; neither exit status nor positive clearance can override
the saved failure. The robot passed behind the actor's walking plane, but
ordinary safety stops and an observed wait are not accepted checked-yield plans.

The single admitted waiting field was at source 21.5 s. The classification then
became uncertain during the actor turn and later parallel, retaining normal
guidance as required. The narrow admitted window did not produce an accepted
walking-window yield plan. Original-request-clock auditing confirms that planner
requests used field sources 21.4 and 21.6 s and never requested the admitted
21.5 s field. This scheduling miss explains the lack of an accepted yield plan;
one admitted field would still not satisfy the unchanged three-source gate.
Two trusted local numeric snapshots, at source 19.0
and 19.8 s, also rejected full stopped horizons against their unchanged normal
fields. Those metadata-only offline probes neither supply runtime fields nor
prove that every other stopped state is infeasible. Do not lower the gates,
renew clocks, freeze a crossing classification or relabel earlier stops to
obtain a pass.

## Later-onset trial

The left-to-right 11.5 s case passed its own 60-frame zero-wheel stationary gate,
then reached all three goals and stopped. Its **original overall report is FAIL**
because intentional-yield source coverage did not pass:

| Check | Recorded result |
| --- | --- |
| Actual route / time | 3/3 goals, `goal_reached`, 24.1000 simulation seconds |
| Braking-inclusive monitored clearance / stopping drift | .467338 m / .002240 m |
| Live fields / fields committed during translation | 240 / 209 |
| Source/current-pose admitted wait fields / rejected-normal candidates | 5 / 28 |
| Parallel classifications with normal guidance | 74 |
| Encounter field-switching audit | PASS |
| Accepted walking-window checked-yield plan sources | **1**, required at least 3 |
| Longest independently observed walking-window wait | 2.450000 s |
| Verified checked-yield zero-command execution during transverse walking | **0** |
| Independent body containment | 240/240; max center error .176855 m within unchanged .25 m prior |
| Minimum executed analytical margin | .173944 |
| Accepted visuals / object assessments / complete fresh four-factor rows | 92 / 15,764 / **0** |

All independent source/frame/selection/pair, current-pose, normal-alternative,
containment, actual-arrival, stopping, analytical and scene/root checks passed.
Do not replace the original overall failure with those component passes.

Original request/result/execution joining found **three** accepted checked-yield
plans across all phases. Their field sources were 23.0, 23.2 and 23.3 s; actual
requests were at 23.166667, 23.333333 and 23.500000 s. The independent actor
departure handoff was at 23.133333 s. All .500000 s of verified zero-command
execution therefore occurred **after** transverse walking, during departure.
Only the first field source was inside the walking window. Original clocks are
retained; this is not an expired-command finding or an excuse to count the later
plans as earlier walking-window yields. Actor phase remains evaluation-only.

## Coverage conclusion and next bounded correction

The original left-to-right/10 s reference retains its full pass. Both new cases
establish actual arrival and stopping with accepted source/geometry/analytical
checks, but **neither passes the full intentional-wait acceptance**. They are
retained stress results, not repaired timing/direction robustness. Native curved
rear passing, guide mode and a causal CRA comparison remain separate work.

The next implementation target is the handoff from a newly admitted waiting
field to planning/execution: avoid silently missing a short admissible window,
retain the real request and field clocks, and revalidate every resulting command
against the latest field. That alone is not a claimed fix: the mirrored run had
only one admitted field, and the later case became admissible too near departure
to establish the required walking-window coverage. Diagnosing that short window
must preserve temporal derivatives, motion/identity confidence, unchanged
classification handling and exact horizon checks. Do not freeze an old field,
relabel safety stops, lower the three-source gate, accept uncertain crossings or
change priors/phi to make the result pass. Any broader prediction or changed
waiting-behavior study belongs in a separately declared bounded variant.

These single runs are sufficient to identify the limitation, not to prove
general impossibility or compare speed/risk causally. The primary fixed-lambda
comparison and the accepted reference remain intact. Existing main, long-service
and PR #117 holds remain unchanged; this coverage branch stays draft/unmerged.

## Local records

- `outputs/paper_flux/run_live_crossing_coverage_v2.py` and each trial's
  frozen-source `invocation.json`.
- `outputs/paper_flux/live-crossing-mirror-parked-v1/` (harness failure) and
  `live-crossing-mirror-parked-v2/native/` (accepted parked component).
- `outputs/paper_flux/live-crossing-mirror10-v2/`, including the original failed
  report, commands, full JSONL fields, independent audit and native video.
- `outputs/paper_flux/live-crossing-mirror-normal-wait-probe-v1.json` (offline
  diagnostic only), `live-crossing-mirror-wait-scheduling-v1.json` and
  `live-crossing-controller-contract-v1.json`.
- `outputs/paper_flux/live-crossing-late115-v2/`, including original failed
  report, independent audit, full fields, commands and video.
- `outputs/paper_flux/live-crossing-late-yield-handoff-v1.json` and
  `live-crossing-late-yield-summary-v1.json` (post-hoc original-clock joining).
- `outputs/paper_flux/audit_live_crossing_coverage_v1.py`,
  `plot_live_crossing_coverage_v1.py` and `live-crossing-coverage-v1.png`.

Outputs are ignored local experiment artifacts, not PR attachments. This is
bounded coverage evidence with retained failures, not universal safety,
real-time operation, a native curved rear-pass result or causal CRA superiority.
