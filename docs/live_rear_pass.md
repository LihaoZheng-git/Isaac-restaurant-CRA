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
