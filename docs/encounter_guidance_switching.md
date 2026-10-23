# Encounter-dependent guidance (separate controller variant)

The user approved steps 1–3: observed encounter classification, per-human lambda
switching and controlled navigation comparisons. Rear-pass planning is deferred.
The primary comparison retains fixed lambda 1.25; encounter-dependent switching
is a separate ablation, not risk-dependent lambda or a replacement safety theorem.

## Step 1: current motion classification

`EncounterMotionClassifier` uses only current LiDAR track centroids/inferred world
velocities, validated human track IDs and the obstacle-unaware active waypoint
bearing. No actor pose, authored route, speed or phase enters it. The existing
human identity adapter supplies current positive ownership and original class
clocks; the classifier neither creates nor renews those identities.

Default engineering gates are configurable: >=20 current points, >=3 consistent
observations, <=.35 s source gap, <=.35 m/s velocity change, <=.2 m position
prediction residual, <=15 degree route-bearing change, >=.2 m goal distance.
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
bearings and invalid inputs. Runtime integration and navigation results are
pending on subsequent bounded branches.

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

The corrected controller requires new virtual/stationary/native validation;
neither earlier failure is retroactively repaired. Matched native baselines
were held automatically when the candidate reports failed. No physical wheels
remain active while this correction is edited.
