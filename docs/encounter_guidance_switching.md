# Encounter-dependent guidance (separate controller variant)

The user approved steps 1–3: observed encounter classification, per-human lambda
switching and controlled navigation comparisons. Rear-pass planning is deferred.
The primary comparison retains fixed lambda 1.25; encounter-dependent switching
is a separate ablation, not risk-dependent lambda or a replacement safety theorem.

## Step 1: current motion classification

`EncounterMotionClassifier` uses only current LiDAR track centroids/inferred world
velocities, validated human track IDs and the configured obstacle-unaware active
route-segment direction. Legacy callers retain the robot-to-waypoint bearing.
No actor pose, authored route, speed or phase enters it. The existing
human identity adapter supplies current positive ownership and original class
clocks; the classifier neither creates nor renews those identities.

Default engineering gates are configurable: >=20 current points, >=3 consistent
observations, <=.35 s source gap, <=.35 m/s velocity change, <=.2 m position
prediction residual and <=15 degree route-direction change. The legacy waypoint
bearing additionally requires >=.2 m goal distance; an explicitly supplied unit
route-segment direction remains defined near the waypoint.
Speeds <=.12 m/s may classify stationary; moving classification requires
.2..2 m/s. The interval between is uncertain. Absolute velocity-route angle
distinguishes parallel (entry <=30 degrees, hold <=40) and transverse crossing
(entry >=60 degrees, hold >=50). Parallel direction is reported separately for
same/opposite travel. The middle band is uncertain; no state is carried through
missing identity, invalid continuity or unresolved motion. Every result retains
its actual scan epoch, consistency count and state-start epoch. This policy
hysteresis is not physical evidence persistence or a guaranteed collision model.

Step 1 unit tests cover both crossing directions, both parallel directions,
stationary actors, rotated world frames, angular hysteresis/turns, source gaps,
missing/replaced identities, sparse returns, excessive speeds, changed waypoint
bearings and invalid inputs. Steps 2–3 integrate this classifier transactionally
and validate the bounded live transition described below.

## Step 2: per-human field integration

The opt-in `--encounter-guidance-ablation` requires the verified bare-human
object-map/passing path, explicit body prior and yield strategy, without extra
guidance-tracking cost or the global-zero option. Every current verified human
is classified; lambda0 eligibility additionally requires the original current
qualified passing selection (including its original eight-second expiry).
Thus the new classifier does not bypass the existing selection/identity gates.

Only the selected stable crossing parent, its body-prior cells and positively
proven current visual fragments get lambda0. Unmatched support/environment,
unverified auxiliary returns, unrelated tracks and all fallback class policies
retain lambda1.25. Parallel, stationary or uncertain classification restores
normal guidance and disables intentional yielding even if an older passing-side
preference is still latched. It does not force unsafe waiting: original exact
temporal DTCBF, dense geometry and analytical zero-command checks still apply.

The classifier is copied with the other worker state and committed only after a
complete field succeeds. A temporal gap cannot advance its accepted history.
The applied per-row audit retains classification and selection epochs; physical
CRA/lambda1.25, risk/flux/side, all raw occupancy and factor provenance are intact.
`fixed_lambda=null` explicitly identifies the mixed per-obstacle field; it is
never reported as the primary fixed-lambda experiment. Legacy defaults and the
always-zero ablation remain separate.

## Step 3: comparisons and acceptance

`scripts/evaluation/compare_encounter_guidance.py` runs five controlled cases:
both transverse crossing directions, both parallel directions, and crossing then
parallel departure. The three variants are primary fixed lambda1.25 without the
new yield preference, the earlier global-zero/yield variant and the new
encounter-dependent/yield variant. Their actor onset (11.5 s), geometry, nominal
waypoints, risk/flux and numerical budgets are identical. This compares complete
controller variants, not a causal lambda-only intervention.
The native `--encounter-comparison` mode requires all arrival, safety, source,
identity and flux checks while reporting the original curved-bypass score
separately: the primary baseline is not required to demonstrate the proposed
curved passing behavior. Legacy demo acceptance remains unchanged.

The first matrix is **ORACLE virtual control**: authored moving disks, class
identity/velocity and risk, no visual estimation or physical wheels. It uses
the actual field pipeline, temporal derivative, N6/.2 s unicycle MPC, exact
DTCBF/dense geometry and 60 Hz analytical checks. It records arrival, independent
physical-disk/robot-footprint clearance, lateral deviation, checked yields and
all failures. The virtual model has no physical stopping inertia or asynchronous
latency. A one-second zero-command tail cannot validate physical robot braking.
The fixed grid/out-of-domain oracle fixture scope remains explicit.

The optional recorded-sensor replay instead uses saved measured/inferred tracks,
original source times, verified identity rows and recovered waypoint context.
Its matched flag is reconstructed from recorded observation count. It only
replays policy classification: no new perception, field, controller or motion
acceptance follows. Do not describe either benchmark as a live sensor result.

The critical live Isaac validation is the existing crossing-then-departure
fixture with the new option, its own60-frame stationary component/envelope gate,
unchanged three-waypoint arrival, containment, .02 m independent clearance,
source/command freshness and actual stopping requirements. A separate mandatory
policy audit requires >=3 selected crossing-zero fields and >=3 observed
parallel classifications with restored normal guidance. Full fresh four-factor
perception, curved bypass and real-time acceptance remain separate gates.

## Retained first results and route-context correction

At2acae83, all2,506 units/Ruff passed. The full ORACLE matrix is **14/15** passes:
fixed and encounter variants passed5/5 each; global-zero passed4/5. Its oncoming
parallel run failed `analytical_emergency` at12.8 s with .354489 m clearance and
negative analytical margin -.057559. Positive clearance does not pass that run.
An earlier commentary claim of15/15 was incorrect and was explicitly corrected.
The original matrix remains in `encounter-oracle-matrix-v1`; it is not sensors.
The240-field recorded-sensor policy replay classified32 crossing/63 parallel
rows. That replay is not a recomputed field or navigation result.

The native11.5 s onset run (`encounter-native-encounter-v1`) reached3/3 goals in
24.4 s, with .467149 m clearance/.002367 m stop drift. Current policy, motion,
containment and source checks passed (33 crossing-zero/65 parallel rows), but
**overall FAILED**: only one accepted checked-yield source occurred during
transverse walking, against the unchanged three-source gate. The2.45 s observed
wait included safety stops. Many zero horizons were correctly rejected by full
predictive/geometry checks; optimizer source ages were short, so this is not
evidence for a latency-only diagnosis. Two additional accepted yields occurred
after the actor started turning, with actual root speed .042/.142 m/s, and must
not be counted as walking. Keep the old report and its failure.

The separately declared10 s onset retry (`encounter-native-encounter-v2`) exposed
another issue: near the first waypoint, the legacy bearing's <.2 m unresolved
direction gate disabled crossing classification/selection despite valid current
motion. It also remains a failed policy-transition trial even though navigation
completed. Moving the actor earlier alone does not fix that context issue.

The bounded correction supplies a separate validated unit direction for the
configured active nominal route segment (previous waypoint to active waypoint;
initially anchor to first waypoint). Only the new encounter variant consumes it.
It is obstacle-unaware configured navigation context, **not** an authored human
route, perception evidence, or a new path planner. Physical `waypoint_world_xy`,
CRA route interactions and the nominal controller are unchanged. The classifier
and crossing selector no longer need an unstable robot-to-waypoint bearing when
that direction is provided; absent it, their original near-goal guard remains.
All motion/identity/source/consistency and exact safety gates stay unchanged.
Genuine route turns still reset classification through the existing direction-
change gate. Tests separately verify the legacy rejection and new segment path.

The corrected controller's validation is below; neither earlier failure is
retroactively repaired. Matched native baselines were held automatically when
the first two candidate reports failed. Each subsequent trial used its own
stationary gate and frozen tracked source; all native processes have now exited.

## Final bounded validation (control source 912d1a2)

All 2,509 unit tests and Ruff passed before native motion. The corrected
encounter variant also passed all five renewed ORACLE virtual cases in
`encounter-oracle-candidate-segment-v1`; the earlier global-zero oncoming failure
remains unchanged. Both crossing directions and both parallel directions have
virtual coverage, **not** a five-case native sensor validation.

Three native Isaac trials used the same continuing-away left-to-right pedestrian,
10 s exogenous navigation onset, original three robot goals, body prior
.40+.25 m, human boundary flux -1.7, N6/.2 s MPC, QP1000/rescue20, .5 s candidate
age/.2 s verified execution interval, 90 s bound and .025 simulation pacing.
Every run passed its own 60-frame zero-wheel human/envelope gate first. The
physical CRA lambda stayed 1.25, with unchanged exact DTCBF, geometry, analytical
CBF, source/identity, independent clearance and braking checks. There is no
extra guidance-tracking cost in these three variants.

| Native variant | Arrival | Navigation time | Minimum clearance, braking included | Stop drift | Live fields / containment |
| --- | --- | --- | --- | --- | --- |
| Primary fixed field lambda1.25 | 3/3 | 23.35 s | .617182 m | 2.326 mm | 233 / 233 |
| Separate global-zero field | 3/3 | 22.85 s | .634683 m | 2.340 mm | 228 / 228 |
| Encounter-dependent field | 3/3 | 23.50 s | .655669 m | 2.278 mm | 234 / 234 |

The encounter variant's **original full report passed**. Its independent audit
verified 32 selected crossing-zero fields, 74 parallel classifications and
unchanged lambda1.25 on unrelated occupancy. It had nine accepted checked-yield
sources during transverse walking and 2.283333 s of observed waiting. Joining
commands by their **original optimizer-request epochs**, not field epochs,
verified 68 physics rows / 1.133333 s of actual checked-yield zero commands during
walking. Safety stops are retained separately and do not count as accepted yield
plans. All 234 independent body-prior containment rows passed; maximum center
error was .172673 m within the unchanged .25 m prior. The analytical margin
minimum was .061844. The robot resumed and reached the unchanged final goal.

### Baseline scoring correction and preserved reports

The two original baseline reports remain `passed=false`: neither made a
passing-side selection in this fixture, so the inherited proposed-strategy
three-selection requirement failed. Global-zero also had zero accepted
intentional yield sources, despite .85 s of observed stopping. Neither baseline
is claimed to pass the proposed intentional-wait/curved-bypass strategy. Both
did reach all goals and pass their recorded human-source, numeric policy, flux,
containment, clearance and stopping checks (minimum analytical margins
.164769/.158384). Their original reports and launcher exceptions are unchanged.

`summarize_variant_passing` now separates shared comparison criteria from the
proposed feature. Baselines need valid normal-flux/lambda mapping and route
completion; **any actual selection** still needs correct human provenance and
original source-age checks. The encounter variant additionally requires its
unchanged full switching **and** intentional-wait strategy gates. Arrival,
physical safety, source freshness, stationary readiness and containment remain
mandatory separate caller checks. Legacy non-comparison acceptance is intact.
This is an evaluation correction after the frozen native runs, not a controller
change or relaxation of the candidate's earlier failed wait/policy thresholds.

Each native folder has a separate `variant-comparison-validation.json`; all
three pass its **recorded shared comparison** checks. It records the original
report SHA256 and evaluator SHA256. This companion does not rewrite the native
report or assert a new run. The older reports did not serialize the in-memory
fixture-root invariant; that unrecorded invariant cannot be independently
reassessed by the companion. Future reports explicitly serialize it and retain
the existing runtime invariant check. The candidate's original successful
full report already included that check in its original acceptance.

### Interpretation and artifacts

These are single, prior-assisted development trials, not statistical superiority
or causal CRA comparisons. The encounter variant is slightly slower here; no
speed advantage is claimed. Its contribution is the verified per-human policy
transition and intentional checked wait. The virtual matrix's largest lateral
deviation was under .01 m, and the native paths are nearly straight: **no
pronounced curved rear-pass is validated**. Restoring lambda does not force a
visible curve when the departing pedestrian is no longer blocking the route.
Rear-pass planning remains the deferred fourth step.

Human mass70 [45,100] kg and blunt-body contact.10 are configured priors selected
only by validated visual associations; missing potential/thermal stay labeled
.25 priors. The candidate recorded 15,538 object assessments, 79 accepted visual
instances across all objects, and **zero complete fresh four-factor
assessments**. This is not moving-view plume acceptance, calibrated human mass,
physical sharpness/temperature, crowds, real time or exact Safe-SAGE equivalence.
Oracle ego references, RGB read-time approximation and renderer history warnings
remain. Nothing in this work lifts the main/long-service/PR117 merge holds.

Local artifacts (ignored outputs are not bundled into the PR):

- `outputs/paper_flux/encounter-native-{fixed,always-zero,encounter}-v3/`: original
  reports, own stationary gates, source histories, commands, containment and video.
- `outputs/paper_flux/encounter-native-encounter-v3/verified-wait-command-audit-v2.json`:
  correct original-request-clock execution join.
- `outputs/paper_flux/encounter-native-comparison-audit-v3.json`: companion results.
- `outputs/paper_flux/encounter-guidance-native-comparison-v3.png`: checked plot.
- `outputs/paper_flux/audit_encounter_native_comparison_v3.py` and
  `plot_encounter_native_comparison_v3.py`: local audit/plot recipes.
- `outputs/paper_flux/encounter-oracle-matrix-v1/` and the renewed candidate
  matrix: explicit ORACLE control results, retaining the global-zero failure.

The final changes after control source912d1a2 only correct evaluation/reporting
and route-basis wording; no post-run physical/controller behavior was changed.
Final regression: **2,523 unit tests passed in85.43 s**, Ruff over robot/tests/
scripts and Python compilation passed. The first full run had2,375 passes and
148 temporary-directory setup errors in the sandbox's default temp location;
rerunning unchanged tests with a fresh repository-local `--basetemp` passed.
No dependency or environment upgrade was performed.
