# Final restaurant experiment plan

This is the experiment roadmap; actual pilot outcomes are recorded separately
in [restaurant_baseline_comparison.md](restaurant_baseline_comparison.md). It builds on the
existing restaurant, native Isaac human, real food/tableware meshes and
obstacle-unaware service waypoints. A matched short baseline pilot has run;
replicated paper collection is blocked by a standing-actor idle-drift failure.
The old main-merge hold on complete fresh
moving-view four-factor perception remains separate and unchanged.

## Questions and bounded cases

The paper should separate three claims: physical-cue estimation, correct live
per-object CRA/field/control dataflow, and a navigation difference under matched
geometry. Arrival alone demonstrates neither accurate perception nor a causal
CRA advantage. Four numeric factors can include explicitly configured priors.

| Case | Purpose and inputs | Gate and intended comparison |
| --- | --- | --- |
| Standing native server | Natural standing-idle sway; RGB person/SAM association, live LiDAR geometry/motion, visually associated adult mass prior | All waypoints and stopping; independent human/layout clearance; fresh factor/provenance counts; no claim of rigid zero velocity |
| Crossing native server | Same scene/robot route; selected slow transverse native walk at requested .5 m/s, earlier actor trigger and audited +15 degree RGB pitch | Parked-view component and coupled crossing passed separately, including nine owned moving-kinetic/seven simultaneous observations. Same-mount standing arrival/stopping also passed. Faster .7 m/s and close-view failures remain separate; replicated baseline comparisons are pending |
| Table-side mixed props | Existing bowl/pot, utensils and ordinary restaurant furniture; detected cues determine scores | Stationary per-object ownership and factor reports first. Positive support/landing drop and contact/plume observations must be demonstrated in this actual view; otherwise preserve priors and report the case as prior-assisted |
| Carried server pilot / later bowl-factor study | Audited existing bowl mesh attached through the native hand system; cross once then continue in the upper aisle | The pilot requires parked human perception, actual hand/prop attachment, retained observed occupancy and independent body/bowl clearance, then robot arrival/stopping. Repeated lower-aisle return trials remain failed on track loss. This pilot does not establish separate bowl assessment. A bowl-factor comparison additionally requires its own RGB/LiDAR ownership and justified fall/plume evidence; pose/asset name never supplies fall height or heat |
| Longer service mission | Left-side pickup to destination through validated local regions | Implement and validate field-region transitions and source-clock continuity first; do not concatenate short-run successes into long-route acceptance |

Use the accepted 3.9 m `human-service` route for the first matched human study.
The original route and the rejected overlap/stall trials remain recorded.
Experimental camera-mount changes require their own stationary gate and audit;
all compared methods must use exactly the same accepted mount and calibration.
Do not loosen instance truncation/association gates to make a selected actor pass.

## Primary comparison

The live runner now implements `robot/evaluation/cra_experiments.py`'s three baseline identities:
geometric velocity-aware CBF, fixed-class Safe-SAGE-style policy, and continuous
CRA Safe-SAGE-style policy. The adapters, immutable manifests, failure-preserving
reports and trajectory plots have unit and native-pilot validation. That does
not establish superiority or complete four-factor perception. Keep lambda
**1.25**, identical nominal waypoints, occupancy, tracker,
sensor streams, mass assumptions, source-age limits, solver budgets, speed limits,
exact geometry verification and stopping gates. The policy adapter is the
intentional change. A risk-dependent lambda is a separate ablation.

Freeze factor scales, priors and confidence handling before primary collection.
The current human diagnostic shows rising relative energy/nominal kinetic risk,
but the .25 uncertainty prior and changing confidence can reverse the final
policy-score ordering. Report that sensitivity explicitly. Changing those
calibration assumptions is a research decision; no such change was made to
obtain the standing/crossing arrivals.

The refined protocol uses nominal static/standing cases and exogenous crossing
triggers at 7/8/9 navigation seconds, each with five repeated runs per method
(75 trials). These timing variants supersede the older proposed lateral shifts;
actor path/geometry stays unchanged. Only the nominal bounded pilot has run.
Validate full-duration standing stability before that repeated collection.
Record actual geometry and initial states; native animation/inference scheduling
need not be deterministic. Do not select only favorable repetitions. Replay comparisons
may share recorded observations, but remain offline and cannot count as wheel
experiments. Do not change hazard geometry between methods and attribute the
resulting clearance difference to CRA.

Report completion, collision/independent-clearance rejection, minimum clearance,
travel and stop time, path length, safety interventions, stale/optimizer failures,
minimum CBF margin, source-to-field latency and simulation/wall-time ratio. Also
report each object's score/phi traces, fresh supported-factor counts, original
evidence ages, prior/retention fractions, instance rejection reasons and guard
events. Keep sensor, prior-assisted and oracle groups separate. Retain every
failure and include denominators; five repetitions are descriptive, not a claim
of broad statistical safety. No experiment may bypass branch checks or the
stationary gate.

## Deliverables and order

1. Native-human standing and farther/slow crossing control gates passed;
   retained failures and current scope are in `restaurant_human_navigation.md`.
2. Native bowl attachment/walking and parked-robot human perception now have
   actual RTX images/video; strict carrying stop and separate bowl ownership
   remain unresolved. The user selected continuous crossing rather than a
   stopped carrier. Its single-crossing upper continuation now passed the
   3.9 m robot mission and stopping, separately from failed repeated returns.
   Retain the distinction between attachment, human perception and bowl
   assessment; potential/thermal remain priors and velocity transients remain.
3. Matched adapters and short pilots are implemented. The first complete pilot
   retained six passes/nine attempts; a separately declared .18 m human recipe
   retained five full passes/six human attempts, including all three crossings.
   All six robots arrived, but geometric standing failed the .25 m idle-drift
   fixture bound at .286741 m. Preserve that failure and fix/validate the future
   standing fixture before the repeated study. Diagnose phi saturation by factor
   and prior sensitivity offline, without lowering unknown scores for navigation.
4. Stop before the long service demonstration for the user's scenario discussion.
   Expand the mission only after region-transition validation. Record a final
   service video with score/phi/provenance traces and safety interventions.

The result contract now retains signed inflated-monitor clearance for failed
outcomes and distinguishes overlap from measured physical contact. Preserve
failed trials and raw diagnostics rather than omitting them from summaries.

This study supports the selected simulation restaurant scenes. It does not
establish calibrated kilograms, physical sharpness, temperature, crowded tracking,
real-time control, exact planar/unicycle equivalence to Safe-SAGE, or deployment
safety. A plume is a visible RGB cue; no-plume observations do not establish cold.

The first live baseline implementation and refined prespecified timing matrix
are in [restaurant_baseline_comparison.md](restaurant_baseline_comparison.md).
Exogenous crossing-time variants replace new lateral placement shifts for the
first comparison; actor geometry remains unchanged. This prospective document
and older component results are not completed comparative results.
