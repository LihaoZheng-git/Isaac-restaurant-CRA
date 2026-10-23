# Live checked rear-reference experiment

This is a separate controller/route-guide ablation, not the fixed-lambda primary
or an exact Safe-SAGE reproduction. The user authorized trying the previously
virtual curved rear-pass component in native Isaac. Existing native wait-and-go
and the failed mirrored/late waiting reports remain unchanged.

`--checked-rear-pass-reference` requires the encounter/checked-wait experiment
and explicit `--waypoint-mode guides`. The same robot route stays obstacle
unaware. Only current, established, positively supported, selected perpendicular
human tracks qualify. Tracked world centroids/velocities and the configured robot
route are rotated into the actual fixed field anchor; actor truth is never an
input. Parallel/uncertain/unselected actors cannot inherit a rear reference.

Source-pose behavior chooses a rear reference, checked waiting, or release.
Rear/released modes retain applied lambda1.25; only the separately admitted
waiting field may use lambda0 for the selected human and its already proven
parts/body-prior cells. Physical CRA, risk/flux, unrelated occupancy and all
original identity/evidence/selection epochs remain unchanged. The existing
source-pair and current-pose waiting admissions and transactional publication
remain active. There is no frozen actor/field or derivative reset.

The rear search runs after the ordinary MPC and uses the existing bounded
576 turn/translate profiles, N6/.2s, original actuator limits, full exact temporal
DTCBF/dense geometry, future mapped analytical inequalities and first-command
CBF checks. At least .01m horizon translation is required. Its explicit utility
is a controller preference, not the original MPC objective. No accepted rear
reference leaves the ordinary checked controller response available. Only an
admitted wait field with a newly checked full stopped horizon may request yield.
The normal live execution lease/current-field validation and per-physics-step
analytical CBF still govern every wheel command.

The optional .2rad current trailing-heading continuation is an engineering
behavior assumption previously tested virtually, not retained identity or
physical evidence. Guide lookahead1m/cross-track1.5m changes intermediate waypoint
semantics; the final .10m arrival and actual stopping remain mandatory.

Acceptance separately requires the existing .20m trailing arc during transverse
walking, behind-plane crossing, current source/identity/policy audit, human
motion, body containment, independent clearance including braking, actual final
arrival and stopping. Additionally, three accepted rear-plan field sources must
occur during walking and actual translating commands must join those accepted
plans by the original **optimizer request clock**. A selected proposal, ordinary
safety stop, departure-only curve or launcher exit0 cannot pass these gates.

Initial native fixture: separately declared8s continuing-away crossing, matching
the earlier virtual timing choice. This never repairs the failed10/11.5s native
waiting coverage. Human mass70[45,100]kg, blunt contact.10, unsupported factors.25,
body radius.40 plus.25 centroid uncertainty and human flux-1.7 remain labeled
priors/ablations. Stationary validation precedes motion; the new trial also needs
its own60-frame zero-wheel gate. No fresh complete four-factor, real-time,
calibrated-human or causal CRA superiority claim follows from this experiment.

Implementation source validation and native outcomes are recorded below when
available. Existing main/long-service/PR117 merge holds remain unchanged.

## Original native attempt (a846ae5)

The independent60-frame zero-wheel component and the trial's own60-frame gate
passed. The original `native-rear-left8-v1/report.json` remains **FAILED**: the
robot reached all three route guides/final goal in21.266667 simulation seconds
and stopped, but its walking-window arc was.153753m, below unchanged.20m.
Behind-plane projection was-.745559m. Independent braking-inclusive minimum
clearance was.767339m, stopping drift.002376m and minimum executed analytical
margin.160314. All212 source-aligned body-containment rows passed (maximum
centroid error.161301m against unchanged.25m uncertainty).

The live experiment did select and execute rear references:26 source fields,
17 selected/accepted plans,16 distinct walking-window plan sources and87 actual
translating command rows/1.45s joined by original optimizer request clocks.
Source/frame/selection/pair/vector-transform, policy, clearance, arrival and
stopping audits passed separately. Seventy-one parallel classifications retained
normal guidance. No wait candidate was requested. There were69 accepted visual
instances/14,203 assessments and **zero complete fresh four-factor assessments**.
The launcher's exit0 accompanied a saved failed report/RuntimeError; it is not a
pass. Full implementation validation passed2,647 units/Ruff/compilation/CLI.

The first actual rear request at source16.7 selected a complete translating
horizon whose first interval was pure rotation `(v=0, omega=.8)`. A separately
opt-in reference-quality correction requires at least.01m predicted translation
in the first.2s interval. This excludes turn-first rear proposals only; it never
forces motion or changes primary/wait commands, actuator limits, physical
evidence, fields, clocks or safety constraints. Default0 retains the original
search. If no such reference passes all checks, the original safe response
remains. On the exact original SHA-verified native input, this rule selects
`(.2,.6)` with every original exact/dense/future/first-command check; this is an
offline numerical result, not a native repair. Native retry acceptance is pending.
Preserve both versions and the original.153753m failure.

The first-interval correction passed2,651 unit tests in103.37s,55 focused
component tests, Ruff/compilation and three unchanged early8s ORACLE guide
cases (both crossings and mixed departure). Virtual walking arcs were.331043,
.308777 and.331043m with arrival/analytical/clearance/behind checks. These virtual
results and the exact-input numerical replay do not pass a native retry. The
retry preserves the8s scenario and every other native controller/physical
parameter; it changes only the explicit.01m rear-reference quality setting.

## First-interval native retry (86eb92e)

`native-rear-first01-left8-v2` also remains **FAILED**: actual3/3 arrival in
20.966667s, .697523m braking-inclusive clearance/.002386m drift and.019259
minimum executed analytical margin passed, as did209/209 body containment
(maximum center error.161561m). The arc was.143690m, below.20m; behind-plane
projection-.784005m passed. There were27 rear fields/18 selected and accepted
plans,17 walking-window rear-plan sources and124 verified translating command
rows/2.066667s. More immediate translating execution did **not** improve the
curved gate. All source/pair/vector/policy/scene checks passed separately.
Seventy visuals/13,978 assessments and zero fresh complete four-factor inputs
retain their original priors/scope. Preserve its failed report and traceback.

The final bounded tuning attempt changes the rear target offset from.7m to
1.2m, retaining the.01m first-interval quality and every other8s native setting.
This is a **preferred reference target**, never a larger body envelope, measured
clearance, physical risk or forced lateral path. All candidates still require
every original safety check and may fall back. No native success is implied.

The rear-only report labels now name rear-or-checked-wait source coverage and
the curved strategy correctly. Older reports used legacy keys containing
`crossing_zero`/`intentional_wait` for those separate rear-variant booleans.
Their numeric counts and acceptance formulas have not changed; the original
reports/SHAs stay intact, and any corrected-label companion is separate.

The1.2m target configuration passed2,651 unit tests in87.61s, Ruff/compilation,
and all three early8s ORACLE curved/arrival cases. Their walking arcs were
.480649/.478668/.480649m (virtual only). The original native-input replay still
selected a bounded(.2,.6) first command with all checks: a larger target is not
a forced larger command. Native final tuning acceptance is pending.

## Final bounded native tuning (c772308; runtime f2e8d25)

The original `native-rear-offset12-left8-v3/report.json` remains **FAILED**.
Its own60-frame stationary gate and all three route guides/final arrival passed
in21.433333 simulation seconds. Braking-inclusive minimum clearance was.704079m,
stop drift.002660m and minimum executed analytical margin.093791. All212
source-aligned body-containment rows passed (maximum center error.163032m against
unchanged.25m uncertainty). The current-human motion/kinetic gate also passed.

The walking-window arc improved to.185079m, still below unchanged.20m. Behind-plane
projection-.771229m passed. There were25 rear fields,16 selected and accepted
plans,15 walking-window rear-plan sources and122 request-clock-joined translating
command rows/2.033333s. Independent source/frame/selection/pair/vector-transform,
policy, physical clearance, analytical, scene/root and stopping audits all passed
separately; they do not replace the failed curved criterion. The simulator's
exit0 again accompanied the saved failed report and RuntimeError.

| Native development attempt | Final arrival | Walking arc (required.20m) | Minimum clearance | Stop drift | Overall |
| --- | --- | --- | --- | --- | --- |
| Original rear reference |3/3,21.266667s |.153753m |.767339m |2.376mm |FAIL |
| First-interval quality.01m |3/3,20.966667s |.143690m |.697523m |2.386mm |FAIL |
| Quality.01m, target offset1.2m |3/3,21.433333s |.185079m |.704079m |2.660mm |FAIL |

The independent recorded actor/robot timing audit reproduced every original
walking arc. Transverse walking was source16.3--19.6s. Maximum *all-phase* bends
were.261930/.265815/.345595m, occurring1.45/1.55/1.866667s **after** transverse
walking ended. Thus native rear-reference dataflow and visible curves exist,
but a sufficiently pronounced curve within the specified walking window is not
accepted. This timing observation is not proof of a sole/global failure cause.
Do not substitute those later bends for the walking metric or lower its threshold.

The final trial recorded72 visual instances/14,243 object assessments and **zero
complete fresh four-factor assessments**. Human mass/contact and unsupported
factor/body priors, oracle ego-pose/extrinsic references, simulation read-time RGB
clock approximation and renderer-history warning remain explicit. None of these
trials establishes calibrated human physics, real-time operation, general crowds,
an exact paper reproduction or a causal CRA advantage. These are development
attempts on one fixture, not held-out tests, replicated estimates or matched
primary-controller comparisons.

No mirrored rear-reference wheel trial was launched after the first direction
failed its acceptance gate. The original accepted checked wait-and-go reference
remains available. The optional rear variant is not promoted to a default or
merged to main. Further timed-curve development should first review encounter
timing and longer-range maneuver planning as separate research choices; do not
keep tuning this one metric or silently change the scenario/uncertainty policy.
All earlier main/long-service/PR117 holds remain.

Local ignored artifacts preserve each invocation, original report/traceback,
complete fields/commands/truth, native video and independent audit:

- `outputs/paper_flux/native-rear-left8-v1/` (sourcea846ae5).
- `outputs/paper_flux/native-rear-first01-left8-v2/` (source86eb92e).
- `outputs/paper_flux/native-rear-offset12-left8-v3/` (sourcec772308).
- `native-rear-timing-comparison-v1.json` and the individual
  `independent-rear-audit.json` files preserve original report SHAs.
- The first two folders' `rear-policy-label-companion.json` changes names only;
  original numeric acceptance and failed reports remain unchanged.
- `native-rear-offset12-results-v3.png` and `human-navigation.mp4` show recorded
  native results. All three functional configurations passed the documented
  unit/Ruff/virtual checks; virtual passes never overwrite the native failures.
