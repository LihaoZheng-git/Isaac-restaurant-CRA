# Rear-pass feasibility diagnosis and conditional virtual result

The user's follow-up authorized continuing diagnosis after the first15 failed
curved-strategy probes. This branch implements a bounded reference-search
component and an **evaluation-only** controller adapter. It has conditional
ORACLE virtual acceptance, not a repaired original case or native wheel
acceptance. The validated native stop-and-go configuration stays available.

## Direction and inputs

The reference uses the current obstacle velocity transverse to the configured
nominal route. Its trailing direction is the negative of that transverse
velocity. For a northbound robot, a left-to-right pedestrian produces a target
on the left/behind side; the mirrored motion produces the right/behind target.
The code uses vectors rather than an unqualified clockwise sign, and tests both
mirrors in rotated frames. No authored actor phase supplies a controller input.

The virtual benchmark itself supplies ORACLE identity, velocity, geometry and
risk; it is not RGB/LiDAR validation. Historical case keys are preserved: with
the virtual robot along +X, `cross-left-right` actually moves the actor along
+Y, and the reverse key moves along -Y. Plots label actual rotated velocity
directions, not those historical aliases. This naming discrepancy does not
change the velocity-based trailing rule or rewrite old records.

## What the audit established

Five internally generated snapshots from the original late ORACLE encounter
retain their original field epochs and actuator/solver configuration. A bounded
1,944-profile audit at each snapshot decomposes endpoint geometry, raw static
DTCBF, full temporal DTCBF, dense geometry and first-command analytical checks.
It does not substitute zero temporal evidence or execute commands. For example,
at navigation12.0s / field epoch13.0s,518 profiles passed exact/dense/first-
analytical checks and41 had >.01m trailing translation. The largest trailing
displacement in that pool was only.021434m over the1.2s horizon. These are local
feasible candidates, not a complete bypass or proof of global feasibility.

This found a reference-generation gap, but not a sole cause. An initial utility
search offered earlier turns and large later curves while the original late
case still failed the walking-window arc. Requiring productive references alone
then produced two `analytical_emergency` failures in the late left/mixed probes
despite .442736m clearance. Those failures remain preserved. Full future mapped
analytical checks were added alongside full exact DTCBF/dense geometry and the
unchanged first-command filter check; no original safety check was removed.

The traces also exposed a route/reference interaction: a fixed lateral offset
of an intermediate waypoint can keep pulling sideways after the human clears,
while the nominal controller still requires reaching that waypoint within.10m.
A curve can therefore be postponed, cut short, or followed by a return toward an
old checkpoint. The accepted original waypoint controller has not been changed.

## Implemented bounded component

`robot/safety/checked_rear_pass_reference.py` provides:

- Current-motion/context gates and a mirrored trailing target. Configured
  behavior assumptions are explicit:1.4m lead or.15m existing trailing offset,
  1.2m outgoing lateral release, .7m reference offset,1m forward carrot, .8m
  waypoint fade and .10m nominal-goal region. These are engineering behavior
  parameters, not measured safety margins, extra body extent or CRA factors.
- A bounded pool of at most576 turn/translate sequences, within the existing
  actuator bounds and original full horizon. References need >=.01m endpoint
  displacement; a pure-turn horizon cannot become the selected reference.
- Full original temporal DTCBF and dense geometry acceptance, mapped analytical
  inequalities at every predicted interval using the original elapsed forecast,
  and unchanged first-command analytical acceptance. No partial prefix is
  returned. Rejected, modified or emergency first commands cannot be selected.
- An explicit selection utility combining local target progress, heading
  preparation and trailing displacement. This is a separate control ablation,
  **not** optimization of the unchanged primary MPC objective or a new theorem.

`robot/evaluation/checked_rear_pass_benchmark.py` supplies the ORACLE-only adapter:
current qualified crossings use normal lambda1.25 for the rear reference when
there is maneuver room, or lambda0 checked waiting otherwise. Outgoing/released
encounters restore normal guidance. Unrelated occupancy, physical risk/normal
flux and original selection/source clocks are preserved. Its chosen full
reference is filtered on every virtual physics step. It does not register a
native runtime option or read Isaac actor truth in a sensor controller.

Some early ignored prototype logs retained a stale description saying fixed
lambda1.25 even after the adaptive policy was introduced. Their recorded per-row
lambdas and probe source are authoritative. The reusable adapter has explicit
behavior/policy records and is used for the results below; old logs are not
silently edited.

## Controlled results and limits

The original11.5s onset and original checkpoints remain a failed curved case.
The corrected adaptive prototype selected waiting and completed all three
original routes, with zero curved-strategy passes. Moving onset to8s alone while
keeping checkpoints2.2/3.1/3.9 also failed all three curved-strategy gates
(largest walking arc about.0633m). Those are not repaired by later results.

A separately declared fixture uses onset8s and checkpoints1.0/1.5/3.9, keeping
the same start, final goal and actor crossing at x3.3. All three checkpoints are
still mandatory; none is skipped. It lets the avoidance happen within a longer
waypoint segment. In the reusable implementation, matched to stop-and-go under
the same fixture and original N6/.2s, QP1000/rescue20 and safety gates:

| Case (historical key) | Rear-reference arrival | Walking-window trailing arc | Rear minimum clearance | Stop-and-go arrival | Fixed-primary arrival |
| --- | --- | --- | --- | --- | --- |
| cross-left-right |20.0s |.343852m |.704761m |26.8s |20.2s |
| cross-right-left |20.0s |.348371m |.703986m |26.8s |20.2s |
| cross-then-parallel |20.0s |.343852m |.704761m |23.8s |20.2s |

All three rear-reference trials pass the unchanged .20m walking arc, behind-
plane, arrival, positive-clearance and analytical criteria. Stop-and-go reaches
the goal in all three but makes no curved-pass claim. This is a single-run
comparison of complete control variants in an easier **different scenario**,
not a causal CRA advantage or repair of the late/dense-checkpoint failure.
The unchanged fixed-lambda primary also reaches all three goals, with zero
lateral excursion and .728257m minimum clearance. Stop-and-go has 1.233810m
minimum clearance. Rear-reference arrival differs from the primary by only
one .2s planning interval; this is not evidence of a meaningful speed advantage.

The robot crosses the human's original walking plane around16.95–16.99s, after
the transverse window8–12s. Behind projections are -3.67m in the straight
departures and -1.17m in the mixed case. Thus this is a timely trailing-side
precautionary curve plus later passage, not validated close timed passing beside
a still-near perpendicular walker. No real-time, physical-braking, human mass/
contact/temperature calibration, fresh four-factor sensing or native claim
follows. Native factor priors and main/long-service/PR117 holds remain unchanged.

## Pending navigation decision

The user was asked whether intermediate waypoints should remain mandatory
.10m checkpoints or become route guides while the final goal remains mandatory.
That answer changes the navigation objective/evaluation, so no guide-style
acceptance or native route change has been made while it is pending. Any guide
variant must remain obstacle-unaware, based only on pose/route progress, with
unchanged online MPC/CBF safety and final-goal arrival/stopping. Old checkpoint
results remain separately reported. Selected actor/scenario stationary gates
remain mandatory before future wheels.

## Reproduction and artifacts

Run the zero-wheel script with a fresh output folder:

```text
python scripts/evaluation/compare_checked_rear_pass.py --output outputs/rear-pass-comparison --actor-onset-seconds 8 --waypoint-layout before-crossing --controller rear stop-go
```

Default onset11.5 and `legacy` waypoint layout remain unchanged. The manifest
records declared fixture settings, Git revision and source hashes. Existing15
assessment flags/metrics were rechecked unchanged under the extended scorer.

- `outputs/paper_flux/rear-pass-feasibility-v1/`: trusted internal snapshots,
  audit source and full exact profile audits; no externally supplied pickle.
- `outputs/paper_flux/rear-pass-checked-search*`: earlier attempts and retained
  failures; `probe_checked_rear_search*` are their ignored source recipes.
- `outputs/paper_flux/rear-pass-adaptive-search*` and
  `rear-pass-adaptive-local*`: declared onset/layout prototypes.
- `outputs/paper_flux/rear-pass-reusable-early8-before-waypoints-v1/`: the matched
  reusable rear/stop-go results and source manifest. Its top-level diagnostic
  `reason` retained the default no-crossing text even for a selected reference;
  `selected`, the nested context and actual trajectory are authoritative. The
  final component corrects that reporting issue without changing selection.
- `outputs/paper_flux/rear-pass-reusable-early8-before-waypoints-v2/`: final
  rear-reference rerun with corrected diagnostic reason and source hashes.
- `outputs/paper_flux/rear-pass-reusable-primary-early8-before-waypoints-v1/`:
  matched unchanged fixed-lambda primary comparison.
- `outputs/paper_flux/rear-pass-feasibility-v1.png`: recorded endpoint paths,
  actual mirrored motion directions and matched goal-progress curves.

Validation: all **2,573 unit tests passed** in 103.20s. After the final
diagnostic-reason correction, all 52 focused reference/adapter/benchmark tests
passed in 10.39s. Ruff passed for `robot`, `tests` and `scripts`; the modified
runtime/evaluation modules and comparison script compiled. No packages or
environment settings changed, and no new Isaac wheel trial was launched.

The component is implemented and tested; native integration and any change to
the original waypoint semantics are not complete.
