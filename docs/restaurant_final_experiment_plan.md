# Final restaurant experiment plan

This is a prospective protocol, not an experimental result. It builds on the
existing restaurant, native Isaac human, real food/tableware meshes and
obstacle-unaware service waypoints. The current development work is still
validating coupled human crossing. The old main-merge hold on complete fresh
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
| Carried bowl | Audited existing bowl mesh attached through the native hand system, with retained occupancy and separate RGB association | Zero-wheel attachment/ownership gate before carrying motion. Pose/asset name never supplies fall height or heat. Plume remains a separate unresolved moving-view component |
| Longer service mission | Left-side pickup to destination through validated local regions | Implement and validate field-region transitions and source-clock continuity first; do not concatenate short-run successes into long-route acceptance |

Use the accepted 3.9 m `human-service` route for the first matched human study.
The original route and the rejected overlap/stall trials remain recorded.
Experimental camera-mount changes require their own stationary gate and audit;
all compared methods must use exactly the same accepted mount and calibration.
Do not loosen instance truncation/association gates to make a selected actor pass.

## Primary comparison

Reuse `robot/evaluation/cra_experiments.py`'s three baseline identities:
geometric velocity-aware CBF, fixed-class Safe-SAGE-style policy, and continuous
CRA Safe-SAGE-style policy. Their experiment contracts exist; these identities
do not establish that all three are wired into the current live restaurant
runner. Review/implement that bounded integration before collecting comparison
results. Keep lambda **1.25**, identical nominal waypoints, occupancy, tracker,
sensor streams, mass assumptions, source-age limits, solver budgets, speed limits,
exact geometry verification and stopping gates. The policy adapter is the
intentional change. A risk-dependent lambda is a separate ablation.

Freeze factor scales, priors and confidence handling before primary collection.
The current human diagnostic shows rising relative energy/nominal kinetic risk,
but the .25 uncertainty prior and changing confidence can reverse the final
policy-score ordering. Report that sensitivity explicitly. Changing those
calibration assumptions is a research decision; no such change was made to
obtain the standing/crossing arrivals.

Use three prespecified, independently placement-checked actor configurations
(nominal and two small lateral shifts), each with five repeated runs per method.
Record actual geometry and initial states; native animation/inference scheduling
need not be deterministic. Do not select only favorable repetitions. Finalize
the two bounded shifts before looking at comparative outcomes. Replay comparisons
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

1. Finish native-human coupled control validation, retaining failed attempts.
2. Validate native carried-bowl attachment/ownership with zero robot wheels and
   produce actual RTX images/video. Treat unresolved perception as a result.
3. Wire and test the matched baseline adapters; run a small bounded pilot before
   the prespecified repeated study. Diagnose phi saturation by factor and prior
   sensitivity offline, without lowering unknown scores for navigation.
4. Expand the mission only after region-transition validation. Record a final
   service video with score/phi/provenance traces and safety interventions.

This study supports the selected simulation restaurant scenes. It does not
establish calibrated kilograms, physical sharpness, temperature, crowded tracking,
real-time control, exact planar/unicycle equivalence to Safe-SAGE, or deployment
safety. A plume is a visible RGB cue; no-plume observations do not establish cold.
