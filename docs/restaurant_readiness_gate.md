# Bounded CRA readiness before restaurant enrichment

The accepted target is three gates: separated multi-object factor/phi evaluation,
moving-object kinetic evaluation, and repeated combined live navigation. A gate
does not require every input to be measured: unavailable physical inputs keep
their diagnostics and receive explicitly configured numeric factor priors.
These experiments retain fixed lambda 1.25 and obstacle-unaware waypoint nominal.
Restaurant assets are not saved or enriched by these session-only fixtures.

| Gate | Current result | Acceptance boundary |
| --- | --- | --- |
| Per-object factors and phi | Passed, three stationary fixtures | Selected sensor inputs plus explicit support/interaction and missing-factor priors |
| Moving-object kinetic | Passed stationary-robot ordering and bounded simultaneous slow-motion error audit | Scripted separated actors; no crowded tracking or calibrated human mass |
| Combined live navigation | Final-budget three-object and upright-actor arrivals passed; historical three-object arrival also retained | Prior-assisted selected fixtures; low moving ball has an accepted loss-stop regression, not arrival |

These gates support proceeding to the selected restaurant scene build. Each new
asset still needs stationary assessment before motion, followed by matched
navigation comparisons. The accepted scope uses retained visual-property and
missing-factor priors; fresh moving-view plume estimation, calibrated human mass,
physical sharpness/temperature, general occlusion handling and real-time control
remain unvalidated. The older moving-view perception gate on PR #117 remains
distinct from this prior-assisted readiness result.

## Separated multi-object evidence

`cra_live_object_isaac_demo.py --fixture-layout three
--persistent-object-risk-map --plume-control persistent` renders three matte
spheres: a grounded object, a sphere on a .30 m support, and a ground object
with visible opaque puffs. Fixture poses, sizes and support height are authored
oracle **evaluation/rendering** metadata, never factor inputs. The estimators
still receive only copied RGB/LiDAR, simulation pose/extrinsic references, and
explicit domain/interaction priors. No weights or packages are downloaded.

The opt-in 3D connected components preserve original return indices; world-frame
velocity and track association remain planar. This separates the visible raised
sphere from its support in the bounded scene, not general 3D object discovery.
All support, background, unclustered and rejected-mask occupancy stays in the
field. Inference remains category-free through audited SAM and the existing
RGB/LiDAR mass network; exact kilograms are not an acceptance target.

The separate camera visibility opt-ins use observed nearby projected depths and
a first mask depth layer separated by .5 m, with at least eight foreground hits.
They are explicit sparse/separated-depth association assumptions, not complete
occlusion reconstruction. Nearby competing returns still count as foreign;
SAM quality, image-boundary, coverage and ambiguity thresholds are unchanged.
Farther returns excluded from camera association remain in physical occupancy.
Mask-based depth layers must not be described as directly measured visibility.

The support estimator's optional equal-height landing mode requires all candidate
lower landing directions to agree in height. It may return an inferred vertical
drop with an **unknown impact direction/site**, separately from the legacy strict
directional mode. The spherical bottom is only a support-consistency gate.
The 2.5 m landing-search extent and worst-case harmful-fall interaction are
configured priors; object/center height never supplies fall height. Grounded
objects retain the earlier observed-plane/local-continuity prior path.

`scripts/evaluation/cra_object_map_report.py --require-fixture-evidence` matches
recorded assessments to independent oracle fixture centers for reporting only.
Ambiguous matches fail rather than selecting a favorable score. It requires
accepted mass/contact observations for each sphere, grounded low drops, a
positive support-to-landing drop for the raised sphere, and a positive plume cue.
It recomputes .10/.25/.50 prior sensitivity **offline**, preserving the recorded
factor estimates. These sweeps never control wheels or claim calibrated priors.

Earlier failed diagnostics remain in `outputs/cra_readiness/`: v1 had partial
plume-object occlusion and planar support/object mixing; v2 showed a low-contrast
plume miss and ambiguous landing directions; v3 aborted on a new diagnostic
type error (fixed with tests); v4 measured .283 m raised drop and .534 plume cue
but rejected the ground association; v5 accepted ground/plume evidence but
rejected the overlapping raised source. None establishes successful combined
perception. Visible puffs are a proxy fixture, not steam physics or temperature.

The final `multi-stationary-v6` zero-wheel fixture gate **passed** all seven
independent checks over 18 fresh fields. Ground/plume spheres each had 18 accepted
visual updates; the raised sphere had 14. Latest inferred mass outputs were
.421/.127/.280 kg (rough model outputs, not mass truth). Ground/plume local drops
were 0 with explicit continuity priors; raised support-to-landing drop was .25046 m.
The ground plume cue was 0; the visible-puff cue was .53137; raised thermal stayed
unavailable and used its .25 numeric prior.

| Fixture | Thermal/contact/potential/kinetic control scores | Computed phi | Offline phi with .10/.25/.50 priors |
| --- | --- | --- | --- |
| Ground sphere | .1171/.1627/.1999/.1876 | .5195 | .3049/.5195/.7715 |
| Raised sphere | .2500/.0852/.2253/.1875 | .5682 | .3272/.5682/.8436 |
| Visible-puff sphere | .1586/.0858/.1952/.1875 | .4970 | .2660/.4970/.7605 |

All factors include their recorded uncertainty/interaction policy. The positive
plume cue raises its thermal factor relative to the observed no-plume ground
source, but aggregate ordering also includes different visible geometry, mass,
route context and missing factors; these are **not** controlled thermal/potential
causal comparisons. Priors noticeably affect phi, so .25 remains an explicit
uncalibrated engineering setting, not an optimized probability.

Validation: full unit suite 1,687 passed; final affected regression 84 passed;
repository-wide Ruff and diff checks passed. Artifacts and prior failed attempts
are retained under `outputs/cra_readiness/`; `object-validation.json` records
per-factor provenance, association failures and the offline sweeps. Reproduce
with 18 stationary frames and `--save-sensor-snapshots`, then run the report's
`--require-fixture-evidence`. No wheel motion occurred in this gate.

## Moving-object and combined navigation gates

The moving-object test must keep velocity current, distinguish sensor estimates
from independently authored motion truth, invalidate moved support priors, and
report stale occupancy. A stationary development gate precedes any robot wheel
motion. Static occupancy memory is not appropriate for moving actors.

`--kinetic-diagnostic` animates the sphere only after the zero-wheel acquisition
gate and freezes accepted intrinsic cues. It records a 12 s stopped/slow/stopped/
fast-reverse/stopped profile. The .12/.24 m/s scripted velocities are independent
oracle evaluation truth; the tracker/CRA receives actual simulated RGB/LiDAR and
ego pose references, not actor truth. .4 s intervals around phase changes are
excluded explicitly to avoid claiming instantaneous accumulated-scan/filter
response. Per-return ego deskew still does not correct actor motion.

The first `kinetic-stationary-v1` run passed speed/energy ordering: stopped median
.0173 m/s; .12 m/s truth produced .0980 m/s; .24 m/s truth produced .1814 m/s.
Median vector error was .0286/.0659 m/s for slow/fast phases. It has measurable
underestimation, not exact velocity calibration. The repeat gate additionally
requires current returns to own the computed policy, robot stationarity, scene
invariants and bounded old-occupancy age.

The moving-actor opt-in may translate earlier accepted instance bounds by the
**observed track centroid displacement**, after the original identity/competitor
and property-persistence checks. It is an unchanged-shape, planar-translation
association PRIOR, not a mask measurement, actor truth or rotation/occlusion
solution. Original bounds/property source times are retained; unmatched returns
remain separately occupied with priors. Legacy static bounds stay the default.
Moved support is invalidated independently. Positive occupancy has an explicit
.15 s expiry in this diagnostic and never establishes free space or preserves
moving actors indefinitely. Unit tests verify original epochs, translated
ownership and expiry; no confidence/energy/phi threshold is tuned to the fixture.

The final `kinetic-stationary-v2` gate **passed** all eight checks, including
policy ownership, scene invariants and robot stationarity. Steady-phase usable
counts were 12/22/12/22/12. Median relative speeds were .0174/.0955/.0161/.1805/
.0159 m/s; slow/fast median vector errors were .0283/.0686 m/s. Energy severity
ordered .000622 < .001384 < .002553 for initial stopped/slow/fast phases. These
engineering upper-bound severities are small under the unchanged 100 J scale;
aggregate phi also responds to current route interaction, support invalidation,
confidence decay and factor priors, not velocity alone.

Maximum robot displacement was .01618 m and yaw .000293 rad; zero wheel commands
were issued. Maximum retained-occupancy source age was .10 s under the .15 s limit.
Full regression: 1,691 passed, with repository-wide Ruff and diff checks passed.
The sensor plot is `outputs/cra_readiness/kinetic-stationary-v2/kinetic-validation.png`.
This is a stationary-robot dynamic-actor gate, not yet simultaneous robot/actor
navigation or a general moving-human model.

Combined navigation must use each obstacle's computed policy with all occupancy,
fresh exact MPC/DTCBF and analytical CBF checks, actual waypoint arrival, clearance
and safe stopping. Record simulation pacing, latency, field/evidence/prior counts,
identity failures and safety stops. Slowed simulation is not real-time operation.
Repeated combined acceptance is required before claiming restaurant readiness.

The first `combined-live-v1` three-hazard navigation trial **failed** its 100 s
simulation budget: final local state (1.67695, -.34533, .27263), 184 committed
fields/168 translating updates, minimum independent fixture clearance .08894 m,
minimum analytical CBF margin .17076, stop drift .000938 m. Pacing was .1
simulation s/wall s; wall duration after stationarity was 1,233 s. Clearance and
safe stopping do not replace waypoint arrival. Latency/temporal-gap rejections
and safe stops remain recorded; this is not a successful pass-by.

The live demo now exposes a separately recorded frozen-property vision retry
interval (legacy 1 s; bounded experiment 5 s). It never slows current geometry,
motion, route interaction or numeric scoring, renews sources or relaxes freshness.
Stage timers separate perception/ownership/occupancy from field construction.
An offline representative **uniform-prior** sensor-occupancy profile identified
Python configuration-space propagation and connected-component work; it is not
a four-factor or navigation result.

The opt-in vectorized configuration expansion preserves the exact legacy
risk/absolute-flux/known-ID/first-source overlap priority, all occupancy, metadata
and counters. Vectorized equal-policy connected-region grouping preserves
eight-neighbor geometry and deterministic seed labels; nearest-distance tie
normal sums are aggregated exactly. Tests compare every yaw's metadata and
complete guidance/h_full values, gradients and forcing to the legacy path
(1e-12 tolerance). No phi rounding, discarded geometry, PDE tolerance, safety
constraint, nominal or lambda changes are involved. Motion acceptance of these
optimizations is still a separate gate.

`combined-live-v2` also **failed** the 100 s budget with 5 s vision retries:
216 fields/187 translating, final (1.61156, -.37671, .27181), clearance .14714 m,
CBF margin .14131, stop drift .000905 m, 1,264 wall seconds. Faster field
construction was then verified locally (1,699 unit tests and Ruff passed).
The optimized `.05`-paced `combined-live-v3` reached a similar tight-layout
standoff; it was explicitly aborted, with 280-field checkpoints preserved at
source time 38.1, local state (1.67886, -.36601, .23151). It has **no completed
arrival or safe-stop report** and is not acceptance. These mixed-risk tight
layouts expose a bounded unicycle-navigation/liveness limitation; numerical
field equivalence is not global convergence.

The separate selected readiness layout moves the visible-plume sphere from
local x=2.2 to x=3.0 and uses waypoint (3.8, 0), preserving sphere/support sizes
and all estimator inputs/gates. This is a different, explicitly separated
scenario, not a successful replay of the failed tight layout. A 20 s no-progress
watchdog can only fail a run; arrival remains mandatory.

Its first stationary gate (`separated-live-v1`) correctly blocked wheels: mass,
contact and fall checks passed but the 140-degree rectified plume source was
unresolved. The opt-in calibrated 120-degree view preserves more native source
detail; the narrower view is recorded and retains unchanged confidence, source
timestamps, clean-reference/background/coverage and pixel-count gates. Native
source mask pixel counts are carried separately, and an unresolved native
source cannot pass by remap upsampling. Mass/contact/fusion remain in native
camera pixels. `separated-live-v2` passed all seven independent stationary
checks before its wheel trial; motion acceptance is still recorded separately.

The combined moving-actor gate also compares actual relative speed with the
independent actor velocity minus the source-epoch ego velocity. It requires at
least five distinct, owned, evidence-supported simultaneous-motion updates and
median world-velocity/relative-speed errors below .10 m/s, excluding .4 s phase
transitions. Actor truth never supplies estimator inputs. This remains a
separated translating-sphere diagnostic, not human/crowd acceptance.

Full regression reached 1,701 passed; the final source-deduplication change also
passed its affected kinetic tests, and Ruff/diff checks passed. Two earlier
invocations had 113 temporary-directory setup errors (Windows access denied),
not assertion failures. The successful run used a fresh workspace `--basetemp`;
no package or environment upgrade was made.

Long multi-object diagnostics can produce nearly a gigabyte of indented records.
The explicit `--stream-live-records` opt-in appends every complete record once to
`live-fields.jsonl`; `live-fields.json` is only the latest-five checkpoint in
that mode. The report names the storage contract. It requires a fresh stream,
retains every original source epoch and factor/provenance record, and changes
neither scores nor control verification. This avoids repeated full-history
serialization hitches; simulation pacing and any remaining latency are still
reported, with no real-time claim.

`separated-live-v2` **failed** its progress watchdog at local x=3.0025 m before
the (3.8, 0) waypoint: 568 fresh fields, 434 translating updates, .13594 m
minimum independent clearance, .17397 minimum analytical CBF margin, .000713 m
stop drift, 2,404 wall seconds at the .075 simulation-rate limit. Its 36,838
object assessments included 37 current visual acceptances and 675 perception-
ready assessments; the rest are not promoted to complete visual estimates.
The .1 s LiDAR acquisition interval remained explicit; late results were rejected
and the robot stopped. This is not successful navigation or proof that numeric
factor completeness establishes fresh live perception.

The next retry (`separated-live-v3`) explicitly uses .05 pacing, the already
implemented fair one-track visual budget (legacy two), and streamed diagnostics.
All geometric tracks still receive current geometry/motion/numeric policies;
postponed visual attempts remain unknown or labeled property priors. Freshness,
factor priors/scales, nominal, lambda, exact MPC/DTCBF and analytical CBF checks
are unchanged. These simultaneous throughput settings are not an isolated causal
performance comparison; only the resulting bounded acceptance is evaluated.

`separated-live-v3` **passed** actual arrival and stopping: final local state
(3.70319, .00881, -.02200), .14417 m minimum independent fixture clearance,
.12222 minimum analytical CBF margin, .002215 m stop drift. It committed 724
fresh fields, 513 while translating, in 1,633 wall seconds after stationarity
under .05 pacing. The root scene and authored file stayed unchanged. Its route
plot is `outputs/cra_readiness/separated-live-v3/navigation-validation.png`.

The complete JSONL audit passed: all 724 records were present, source IDs/times
strictly increased, accepted field source ages were .1333--.3500 s, lambda stayed
1.25, and all 45,768 object assessments had four finite numeric factor scores.
Only 5,051 factor scores were evidence-supported versus 178,021 configured-prior
uses (mostly room/background occupancy and lost observations). There were 44
current visual acceptances and 873 perception-ready object assessments. These
counts must not be described as complete fresh four-factor vision on every
obstacle. If identity disappears, old positive occupancy remains but its policy
can revert to priors; current code does not establish general cross-ID semantic
map persistence. Original property ages/expiry remain enforced.

The demo's separate opt-in `--immediate-replan` removes an avoidable periodic
wait after a rejected plan or an invalidated plan when no job is pending. The
single-worker bound, current state/field revalidation, candidate source-age cap,
.2 s verified interval and per-step analytical checks remain unchanged. Legacy
periodic scheduling is the default. This scheduling variant is validated
separately; it is not retroactively part of the successful v3 run.

### Dynamic coverage failures and continuity stop

`moving-live-v1` (.075 pacing) passed the independent simultaneous-motion
velocity audit: 22 steady usable updates, median relative-speed error .02879 m/s,
world-velocity error .04188 m/s. Navigation **failed** on repeated late inference:
105 fields/90 translating updates, .20152 m minimum fixture clearance, .0000273 m
stop drift. Current code must not reinterpret that velocity result as arrival.

`moving-live-v2` (.05 pacing) also passed velocity (21 usable updates, .03917/
.06163 m/s median relative/world errors) but **failed** the independent clearance
guard at .01744 m, below the .02 m threshold; 169 fields/123 translating updates,
.003994 m stop drift. The low ball disappeared from current LiDAR clusters in
the near sensor blind region; short positive-occupancy retention then expired.
Nearest current clusters in the final five fields were over 2.5 m from it.
Numeric factor priors cannot protect a hazard whose geometry has left the field.

The moving-actor demo now enables `NearbyTrackLossGuard` before publishing a
candidate field. It enrolls real positive geometric tracks within 2 m, remembers
original last-seen source epochs, and stops for unresolved loss over .15 s within
1.2 m of the current robot. A unique one-to-one new-ID match within .25 m may
resolve positive geometry continuity; ambiguous merges/crossings cannot. No
mass/contact/plume/support evidence is transferred or refreshed by this guard.
The engineering radii bound this separated-actor diagnostic, not hidden geometry
or arbitrary actor motion. There is no inferred free space, cached measured
kinetic input, dynamic belief-map guarantee or claim of paper equivalence.

Its offline real-track replay stopped earlier; `moving-loss-guard-v3` then
**passed the expected-stop regression** in Isaac: reason `nearby_track_lost`,
.33322 m minimum fixture clearance, .000916 m stop drift, 84 fields/70 translating.
The candidate exposing the loss was saved as unpublished. The source-only guard
fired without authored target position; the authored monitor independently
measured clearance. Velocity also passed (19 usable updates, .04534/.06757 m/s
median errors). **Navigation remains failed**: this is safe stopping under loss,
not a successful low-moving-object pass-by.

### Separate upright kinetic proxy

The explicit `--actor-shape upright_capsule` fixture is 1.4 m tall with .18 m
horizontal radius, authored only for rendering and independent monitoring.
It is a different, LiDAR-visible person-height geometry proxy, not an articulated
human, calibrated human mass model or retroactive pass for the low ball. The
sphere ground-continuity prior is disabled. No authored height enters estimation
or supplies fall height. RGB/LiDAR association uses the same separated 3D/depth
adaptations and unchanged mask-quality/boundary/foreign-return gates.

`capsule-stationary-v1` passed its separate **kinetic-component** zero-wheel gate
over 18 accepted visual updates: fresh category-free mass, owned silhouette
proxy and confirmed current motion inputs. The generic four-factor perception
readiness remained false. Potential stayed unavailable and used .25; thermal
also used .25 because above-source independent background coverage failed.
The rough network output was .512 kg with .001--5.055 kg engineering bounds,
not human truth. Its elongated silhouette produced a high .81086 contact-proxy
score on an apparently smooth object; this is a proxy-calibration limitation,
not physical sharpness/hardness. Computed stationary phi was .91347--.92023.
Motion recomputes the policy from actual inputs and priors without manual score
overrides; current motion/route context and uncertainty can change phi.

The report requires explicit `--kinetic-actor` to use this narrower component
gate; the original sphere/support/plume evidence gate is unchanged. The proxy's
moving trial keeps every numeric factor, fixed lambda, exact checks and the
new continuity stop. It does not establish moving-human or four-factor sensor
acceptance. Inference deadline comparison now uses the same 1 ns floating-point
tolerance as future-epoch checks; an actually late result by 1 microsecond still
fails. Original clocks and the .35 s policy limit are unchanged.

`capsule-live-v1` kept positive LiDAR tracking and passed the live velocity audit
(20 usable simultaneous **slow-phase** updates, .02428/.02610 m/s median relative/
world errors). There were no qualifying simultaneous fast-phase samples; fast
actor ordering remains the separate stationary-robot diagnostic. The navigation
trial **failed** the optional 20 s goal-distance progress watchdog while taking
a wide detour: final (.79934, -1.03948, -.57300), .23273 m clearance, .232997
analytical CBF margin, .000312 m stop drift, 308 fields/246 translating, 738 wall
seconds. Continuous positive tracking alone is not arrival. Euclidean goal
distance need not decrease during obstacle avoidance, so this watchdog rejection
does not establish global nonconvergence. The separate `capsule-live-v2` retry
disables only that optional watchdog under the same 90 s overall bound; actual
arrival, source freshness, continuity, exact MPC/DTCBF, analytical CBF, clearance
and stopping checks remain mandatory.

The separate all-phase sensor audit of `capsule-live-v1` also found 11 fast-phase
observations (mostly with the robot stopped): median world/relative velocity
errors .03192/.02911 m/s. In 132 observations with a stopped actor and moving
robot, median world-velocity error was .00913 m/s (95th percentile .03142), and
median relative-speed error .00830 m/s. This supports bounded ego-motion
compensation beyond the simultaneous slow-phase subset; it remains an oracle-
ego-pose-referenced simulation audit, not independent physical odometry
validation. See its `all-phase-motion-audit.json`; the navigation outcome remains
failed and is evaluated separately.

`capsule-live-v2` **failed the 90 s duration limit**, with positive tracking and
no continuity stop: final (1.52331, -1.03301, .11099), .19691 m clearance,
.150834 analytical CBF margin, .000733 m stop drift, 604 fields/459 translating.
Its velocity audit passed 22 simultaneous steady updates (20 slow, two fast),
with .02607/.02687 m/s median relative/world errors. Safe waiting and positive
margins are not actual arrival. Late inference results remained rejected.

### Live recovery correction

Inspection found that the live caller always supplied a world-grid negative-y
recovery hint. The recovery pool enforces this hint as a final lateral-progress
constraint, including after the robot has moved below an obstacle and needs to
return upward toward its waypoint. The live caller now leaves that optional
hint unset; per-obstacle Safe-SAGE guidance and the obstacle-unaware waypoint
nominal are unchanged. This does not enforce a passing side or prove convergence.

An explicit MPC opt-in skips recovery candidates whose exact clipped command
cost cannot improve an already verified feasible candidate. This recovery pool
has no state-path objective; ties preserve the original first candidate. The
same pool, objective, exact DTCBF/dense-geometry checks, lambda, factor priors,
field freshness and command execution limits remain. Other callers retain the
legacy exhaustive default. Mirrored regression cases selected identical command
arrays, trajectories and safety margins. One offline case took 3.997 s exhaustive
and .980 s pruned, with identical .12053515625 cost; this is not a representative
runtime benchmark or real-time guarantee.

`capsule-live-v3` retries the unchanged actor, motion profile and 90 s bound after
this correction, with no optional goal-distance watchdog. Actual arrival and
the independent clearance, current sensing, exact-control and stopping gates
remain mandatory.

That retry, at source revision `4055e65`, **failed the 90 s duration limit**:
final (1.94754, -.98004, .37564), .19976 m minimum fixture clearance, .206111
minimum analytical CBF margin, .001975 m stop drift, 473 fields/392 translating,
1,928 wall seconds. The continuity guard remained clear. Its 21-sample live
velocity audit passed (.02390/.02586 m/s median relative/world errors; 20 slow,
one fast). All these component results remain separate from failed arrival.
Of 411 planner results, 32 were rejected as stale; median/95th/max job-to-poll
wall times were .415/29.440/41.332 s. Perception also repeatedly missed its
unchanged deadline. These are diagnostic latencies, not a controlled performance
benchmark: targeted CPU unit tests also ran briefly late in the trial, after
deadline misses had already been observed.

A later separate opt-in rejects an already infeasible exact DTCBF recovery
candidate before dense geometry sampling. Dense geometry cannot repair that
margin; every feasible candidate still receives all geometry checks. The main
solver retains infeasible references for restoration and is unchanged. The
live demo enables this optimization with cost pruning; other callers keep both
disabled. Mirrored winner regressions and explicit feasible/infeasible dense-
check tests cover the distinction. This change was **not** in `capsule-live-v3`.
Slow worker diagnostics now separate actual perception/planning compute from
job-to-poll time, retaining source epochs and recording planner iteration counts.
The logs never renew evidence, supply factor values or accept a stale result.

Software regression after this correction: **1,716 unit tests passed**; Ruff
and diff checks passed. No new models, packages, datasets or global environment
settings were installed or upgraded.

After the separate feasibility-pruning change, **1,719 unit tests passed**.
The next short capsule run is a timing diagnostic, not an endpoint-navigation
acceptance trial; a duration-limit result must remain failed navigation.

`capsule-timing-v1` at revision `7b8d32f` completed that 15 s timing scope and
**did not pass navigation** (`duration_limit`): 129 fields, .23089 m clearance,
.220855 CBF margin and .000885 m stop drift. Its velocity audit passed 21 samples.
Slow-worker records identify several QP calls ending at 4,000 solver iterations
with `exact_safe_seed_feasibility_recovered`; individual compute times included
3.15--4.48 s. Some concurrent perception jobs spent 2.48--4.02 s building fields.
These diagnostics do not prove that every earlier long tail had the same cause.

The demo exposes an explicit `--qp-iteration-limit` (legacy 4,000; bounded
100--4,000) for a separate 1,000-iteration timing comparison. A lower budget may
change the numerical search result and increase reliance on exact-safe recovery;
it does not change the objective or permit an infeasible/expired command. All
field/source clocks and exact-control checks remain. Other MPC callers retain
their existing configuration. The comparison is not a change to lambda or
factor scores, and matched future baselines must report the same solver budget.

`capsule-timing-v2` at revision `3023181` used the 1,000 QP budget while retaining
300 nonlinear-rescue iterations. It **did not pass navigation** (`duration_limit`
at 15 s): 121 fields, .24523 m clearance, .239523 CBF margin, .003782 m stop drift.
The 22-sample velocity audit passed (.02480/.02807 m/s median errors). Planner
job-to-poll median/95th/max times were .359/2.284/34.384 s. The recorded 34.250 s
compute ended in `guidance_rotation_feasibility_recovered`, three SQP steps and
3,300 total solver iterations (three exhausted 1,000-iteration QPs plus 300
nonlinear-rescue iterations). The lower QP budget alone is insufficient.

The demo therefore also exposes `--nonlinear-rescue-iterations` (legacy 300,
bounded 1--300). The next full retry uses 1,000 QP and 20 rescue iterations,
with a separately reported 120 s overall bound. These budgets change numerical
search and can increase exact-safe recovery or stopping; they do not change the
objective, factor priors, lambda, candidate age, verified execution interval or
exact constraints. The original 90 s failures remain failures. Synthetic rescue
and forced-QP-failure regressions now exercise both 300 and 20 rescue iterations.
The short timing runs follow the same fixture but different recorded trajectories;
they are not matched-input timing benchmarks or endpoint acceptance.

Regression after exposing both budgets: **1,721 unit tests passed**; Ruff and
diff checks passed. No environment, model, weight or dataset changes were made.

### Accepted upright-actor endpoint

`capsule-live-v4`, with the explicit 1,000/20 budgets, **passed actual waypoint
completion and stopping** at about 58.1 simulation seconds (120 s configured
bound). It recorded 547 fresh fields/434 translating updates, .21264 m minimum
fixture clearance, .124553 minimum analytical CBF margin and .002068 m stop drift.
Final chassis state was (2.40426, -.01287, .00321), inside the .1 m waypoint
tolerance. Wall duration was 1,268 s at .05 pacing. The scene file remained
unchanged and the continuity guard stayed clear. The observed pass was above
the actor after an initial downward approach; configured passing side remains
soft guidance.

Its live velocity audit passed 22 simultaneous steady observations (19 slow,
three fast), with .02484/.02806 m/s median relative/world errors. The actor was
stopped after 10 s; this is not a continuously crossing-human pass-by. An offline
record audit found the same uniquely matched, positively owned actor track (ID 0)
in **all 547 fields**, with contact and kinetic adapter inputs throughout.
Mass/visible contact persisted as labeled unchanged-property priors with original
source epochs; current motion stayed inferred from current sensor tracking.
Potential and thermal used configured .25 priors throughout. No full four-factor
sensor-only assessment is claimed, and the mass is not calibrated human mass.

All 30,127 object assessments had four finite numeric scores and fixed lambda
1.25. Accepted source-to-commit ages were .13333--.33333 s. The 118,164 configured
factor-prior uses include the many unassessed room tracks; 2,307 retained-property
uses are separate. `stream-audit.json` records this consistency check and uses
authored actor centers only for offline evaluation, never factor/control input.
`navigation-validation.png` and `kinetic-validation.png` retain prior/gap labels.
This bounded result does not prove convergence or isolate a causal budget effect;
the earlier failed trials remain unchanged.

### Accepted final-budget three-object replay

`separated-live-v4` **passed** all seven stationary fixture-evidence checks and
actual waypoint completion/stopping. It reached the waypoint in about 38.9
simulation seconds (100 s bound), with 379 fresh fields/362 translating updates,
.17224 m minimum fixture clearance, .141871 minimum analytical CBF margin and
.002270 m stop drift. Final state was (3.70359, .00909, -.02560); wall duration
was 855 s at .05 pacing. The scene file remained unchanged. The selected ground,
raised/support and positive-plume fixtures retain measured/inferred evidence and
explicit support, interaction, property-persistence and missing-factor priors.

The stream audit checked all 24,489 object assessments: four finite numeric
factor scores, bounded phi (.52335--.77541), fixed lambda 1.25, strictly increasing
source frames/times and .13333--.35000 s source-to-commit ages. Configured factor
priors were used 95,474 times, with 2,905 retained-property uses and 449 complete
prior-mode perception assessments. These repeated per-track counts include
unassessed room geometry and are not counts of distinct fully perceived objects.

This is another successful execution of the separated layout, after runtime
changes. The earlier v3 result used different solver/direction/replan settings;
the pair is not an identical-configuration statistical repeat or evidence of a
causal CRA clearance advantage. Fixed lambda and conservative geometry checks
remain. No oracle factor profile was supplied to either accepted controller.

Both final-budget runs used the following explicit shared options (fresh output
directory and the already audited local Isaac/Python/model environment):

```text
--persistent-object-risk-map --motion --stationary-frames 18
--simulation-rate .05 --planner-result-max-age .5 --save-sensor-snapshots
--visual-reassessment-interval 5 --live-visual-tracks 1
--vectorized-fields --stream-live-records --immediate-replan
--qp-iteration-limit 1000 --nonlinear-rescue-iterations 20
```

The capsule additionally used `--actor-shape upright_capsule --moving-actor
--maximum-seconds 120`. The three-object run used `--fixture-layout three
--fixture-spacing separated --plume-control persistent --plume-view-fov 120
--static-occupancy-memory-prior --maximum-seconds 100`. The optional goal-distance
watchdog was disabled in both. Static occupancy memory was enabled only for the
static three-object scene; the moving actor retained short occupancy expiry and
the nearby-track loss guard. The shared live-demo defaults remain legacy budgets,
so these validated budgets must be selected explicitly.

Mass calibration, physical sharpness/hardness, temperature, moving-view plume
re-estimation, crowded identity crossings, hidden heat/support changes and causal
CRA superiority remain outside these bounded component gates.
