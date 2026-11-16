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

## Frozen-source validation

Source `21f0968da0dc881ff0055610cdd2ba1b8f391614` passed 2,618 unit tests
(91.83 s), Ruff across robot/tests/scripts, compilation, diff checks and CLI
checks. A later focused run passed 31 adapter/policy tests. Tests cover rotated
anchors, staged accept/discard, unpublished gap recovery, actual field rebuilds
against the same accepted temporal pair, alternative failure rollback and the
separate guarded/legacy policy scorers.

The first restricted native launch stalled before scene loading and was stopped
with zero wheel commands. Preserve `live-checked-wait-stationary-v1` and its
startup diagnosis; its launcher exit code is not acceptance. A fresh desktop
launch needed simulator GPU/cache access, with no package or environment upgrade.

The independent stationary run and the motion run's own stationary gate each
passed **60 frames with zero wheel commands**. They validate the selected human
component, not every restaurant asset. The other bowl-observation failure remains
in the records. Human mass is still a visually associated 70 [45,100] kg prior;
the low blunt-body contact prior is .1. Potential/thermal use configured .25
factor priors when unavailable. The renderer-history warning remains unresolved.

The separate native trial retained the continuing-away left-to-right actor
fixture, 10 s onset, .5 m/s actor speed, .4 m body radius plus .25 m centroid
uncertainty, original three strict .10 m checkpoints, N=6/.2 s MPC, QP/rescue
budgets 1,000/20, .5 s optimizer age cap and .025 simulation pacing. Actor modules
were verified unchanged since `84e503d`. Authored actor truth supplies only
evaluation/abort checks. Physical CRA lambda stays 1.25; the existing human
boundary-flux -1.7 ablation stays separate from the primary comparison.

| Native acceptance item | Result |
| --- | --- |
| Original overall report | **PASS**, actual 3/3 goals in 23.4667 simulation seconds |
| Braking-inclusive monitored clearance / stopping drift | .644488 m / .002315 m |
| Live fields / fields committed during translation | 234 / 210 |
| Wait fields admitted at source and current poses | 9 |
| Rejected wait candidates rebuilt as normal fields | 24 |
| Current parallel classifications with normal guidance | 74 |
| Accepted walking-window checked-yield plan sources | 4 |
| Longest independently observed walking-window wait | .683333 s |
| Verified executed zero commands joined by original optimizer-request clocks | 22 rows, .366667 s during walking |
| Independent body-prior containment | 234/234 rows; maximum center error .174666 m within .25 m prior |
| Minimum executed analytical margin | .063909 |
| Object assessments / accepted visual instances / complete fresh four-factor assessments | 15,424 / 84 / **0** |

The independent audit passed source/frame/selection-clock matching, the previous
accepted temporal pair, both-pose wait admission, rejected-normal policy mapping,
actual arrival, stopping, containment and unchanged scene/root layers. The
report's observed wait includes physical slowing/stopping; the shorter executed
zero-command duration counts only commands joined to accepted checked-yield
plans using **optimizer-request clocks**, never field clocks. Rejected candidates
and freshness stops are not relabeled as intentional yields.

Local reproducible records (ignored outputs, not PR attachments):

- `outputs/paper_flux/run_live_checked_wait_v2.py`, with frozen-source invocation
  records; the native report itself does not serialize a revision.
- `outputs/paper_flux/live-checked-wait-stationary-v2/`.
- `outputs/paper_flux/live-checked-wait-motion-v2/`, including original report,
  JSONL fields, commands, human truth, validations and `human-navigation.mp4`.
- `outputs/paper_flux/audit_live_checked_wait_v2.py` and the motion folder's
  `independent-checked-wait-audit.json`.
- `outputs/paper_flux/plot_live_checked_wait_v2.py` and
  `outputs/paper_flux/live-checked-wait-native-v2.png`.

## Scope and remaining work

This accepts the live checked-wait **integration for one prior-assisted native
crossing/departure trial**. It does not repair every historical late/mirrored
native failure, integrate the virtual checked rear-reference, or validate native
guide-mode navigation. The old failures and accepted strict fallback are retained.
The preceding ORACLE timing/mirror matrix remains separate evidence. A later
native mirrored/timing comparison needs its own stationary and full motion gates.

There is no causal CRA or meaningful speed advantage claim: the historical
strict encounter run took 23.50 s, while this single trial took 23.47 s with
slightly different clearance. Ideal zero-command prediction does not model all
braking inertia; native stopping/clearance remains an independent gate. Current
pose admission checks the original field under the existing bounded source-age
contract; it does not reconstruct a new acquisition epoch or establish general
dynamic robustness. This is not exact paper equivalence, real-time acceptance,
calibrated human factors or complete fresh four-factor perception. Existing
main, long-service and PR #117 holds remain unchanged.
