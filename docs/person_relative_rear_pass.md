# Person-relative rear-pass development

This is a separate, disabled-by-default online maneuver experiment on top of
the accepted rear-reference development branch. The user approved implementing
a person-relative arc, and on 2026-10-25 approved adjusting only pedestrian
timing/route for its next fixture, keeping robot limits and all safety gates.
Main, long-service and PR117 holds remain unchanged.

**Current outcome:** experimental and disabled by default. At corrected source
4f0054f, local regression and stationary gates passed and one person-relative
curve was retained across 31 native fields, but no rear command was executable.
All three coupled development trials failed the rear-pass acceptance gates.
Keep the previously validated stop/go configuration as the practical fallback;
the tighter latest fixture is not recommended for the final demonstration.

## Behavior and provenance

`--rear-person-relative` requires the explicit checked rear/consistent-motion/
feasibility experiment. The nominal waypoint controller remains obstacle-unaware.
The original MPC/CBF safety constraints, physical factors, occupancy and lambda
1.25 remain unchanged. An accepted motion-window inference supplies policy motion;
raw physical tracking/CRA kinetic inputs are not replaced.

`PersonRelativeRearMemory` tries at most eleven short constant-velocity passage
times (0.5..3 seconds) on each fresh currently qualified source until one curve
is frozen. Current identity, selection age, straight nominal-route context and
the actual MPC actuator limits are required. No actor state/phase/route enters
this calculation. The separate terminal robot waypoint is configured nominal
route data, not obstacle information. Non-straight segments are rejected.

The passage point lies at the inferred person's crossing plane, in the rear
half-space at least 1.2 m behind forecast motion in the selected fixture. A
0.35 m preferred trailing route offset shapes the curve when that half-space
already includes the route. These are explicitly configured behavior preferences,
never measured clearance, body dimensions or risk evidence. Selecting a region
avoids chasing a fixed-distance point behind a faster pedestrian.

Two tangent-continuous cubic segments connect the original robot pose, passage
point and route rejoin. A 0.25 m path lookahead supplies the rolling target. The
sampled path/turn-duration screen only rejects implausible references; it does
not prove dynamic feasibility or safety. Every selected command horizon still
needs all original exact DTCBF/dense geometry, forecast analytical and unchanged
first-CBF checks, plus current-pose/freshness/execution/independent braking guards.

Once frozen, the curve keeps its original source, person/motion inference,
selection and deadline. It does not chase later images. Current positive identity
and motion must remain compatible: <=0.25 s source gaps, <=0.30 m constant-motion
residual and <=0.35 m/s velocity change. These are policy-hypothesis bounds, not
uncertainty claims about human anatomy. Lost qualification, incompatible motion,
route change or expiry invalidates the curve without restarting that original
selection. Earlier unsuccessful searches may retry only with real fresh sources,
under the unchanged eight-second selection cap. Memory is staged transactionally
with perception; rejected/unpublished proposals cannot commit it.

No valid curve leaves the original checked-wait/normal-MPC fallback. A stopped
command still needs the complete stopped-horizon admission; rejected waiting is
not labeled intentional yielding. Defaults preserve the prior rear experiment.

## Separate stricter evaluation

`person-relative-rear-validation.json` retains all old reports and adds a named
timing/location component. The robot must cross the independently observed
human walking plane during the actual transverse walking window, behind current
motion by at least 0.10 m. A >=0.20 m trailing arc must occur during that window,
within 0.75 m of the crossing plane and 2 m of the human. The 0.75 m location
criterion is a new explicit evaluation scale, not a smaller safety margin.
Actual arrival, safe stopping, source/identity/geometry and real reference
execution remain separately mandatory. Actor truth is posthoc scoring only.

The original accepted left/right native runs fail this stricter component:
their line crossings occurred about 7.21/7.11 seconds after transverse walking.
Their original overall reports remain PASS with hashes unchanged. Trusted local
first-selected numeric snapshots show 1.65/1.37 m current forward lead, versus
the actual 0.25 m/s MPC maximum (nominal maximum is 0.20 m/s). The new short-window
reference is ineligible at those requests. These snapshots are original planning
states/fields, not acquisition-pose worker replays or counterfactual navigation.
Disabled new mode reproduces both original complete command sequences exactly.

Reproduce the zero-wheel recorded audit with:

```powershell
python scripts/diagnostics/cra_person_relative_rear_diagnostic.py `
  --trusted-native-run outputs/paper_flux/native-rear-continuity-left8-v5 `
  --trusted-native-run outputs/paper_flux/native-rear-priorselection-right8-v7 `
  --output outputs/paper_flux/person-relative-recorded-timing-v2.json
```

The diagnostic refuses to overwrite a result and loads only explicitly selected
trusted local SHA-checked planner snapshots. Never load externally supplied pickle
files. Numeric replay never commands wheels or renews source clocks.

## Authorized new actor fixture and validation freeze

The new candidate uses an actor-only y=8.4 m crossing line, 0.15 m symmetric
start/end extension, and the existing robot-world-y trigger at 7.4 m. Walking
speed remains 0.5 m/s. Left starts (27.65,8.4), targets (29.95,8.4), then departs
along x=29.95 to y=12.5; the mirror starts (29.95,8.4), crosses to (27.65,8.4)
and departs at x=27.65. There is no teleport/frozen actor. The actual-position
handoff and every grounding/collider/articulation/layout guard remain unchanged.
New CLI defaults retain original y=8.9, no extension and old actor starts.

The first 0.4 m straight extension failed the independent authored raster and
was rejected before any wheels. Keep that failed precheck. The final 0.15 m/
y8.4 route precheck passed with conservative 0.4 m square actor monitoring:
minimum layout clearance 0.152145 m left and 0.177145 m right. This is authored
abort/evaluation information only, not runtime geometry or native acceptance.

At the source freeze, 2,749 unit tests passed in 99.94 s; 97 focused checks,
Ruff, compilation and CLI checks passed. Tests cover mirrored/rotated geometry,
frozen clocks, current motion/identity/route rejection, transaction discard,
field metadata contracts, original full safety checks and distinct timed scoring.
The region revision supersedes an uncommitted fixed-distance-point prototype;
its earlier component failures remain development evidence, not wheel results.

Require the new parked actor/perception component, independent 60-frame stationary
gate and each wheel trial's own 60-frame gate before live motion. Native outcome
is pending at this freeze. Human/body/factor priors, oracle ego/extrinsics, RGB
read-time clock approximation, no fresh complete four-factor acceptance, paced
simulation and renderer-history limits remain. This is not primary Safe-SAGE,
paper equivalence, real-time control or a general human-navigation guarantee.

## Parked native actor/perception result

At c6ada89 the separately authored left-to-right route passed the native actor
and parked-human sensor gates with zero wheel commands. There were nine fresh
idle, fifteen fresh crossing and twenty-three fresh departure associations,
all retaining the configured adult mass/body-contact priors. Observed median
walking speed was 0.518991 m/s; minimum independent furniture clearance was
0.133828 m, and maximum collider alignment error 2.87e-8 m. Grounding,
articulation, actual crossing, upper-aisle departure and video gates passed.
This establishes the new actor/perception component, not robot navigation or
fresh four-factor estimation. Original report SHA256:
`8625bfc89cfde6d1d12beba1238e38ff3117c1a5fcd6309433eed63443fb6f99`.

Artifacts: `outputs/paper_flux/native-person-rear-parked-left-v1/native/` contains
`validation.json`, `compact-summary.json`, independent actor/sensor traces and
`human-motion.mp4`. The launcher exited zero without a traceback.

A subsequent contract review also rejects a new reference whose proposed
passage time exceeds the original eight-second selection expiry. It could
previously expire before its own planned passage; no clock had been renewed.
The actor/perception component code is unchanged by this stricter maneuver-only
check. All **2,750 unit tests passed in 98.51 s**, with Ruff and compilation.
The independent stationary controller gate and coupled native trial remain
pending at this new source freeze.

## First coupled native results and retry correction

At 2422a87 the independent controller stationary gate and both coupled trials'
own 60-frame zero-wheel gates passed. The two separately timed left-to-right
fixtures **failed the rear-pass experiment**, despite arrival and stopping:

| Actor trigger (robot world y) | Arrival | Minimum clearance, including braking | Stop drift | Minimum analytical margin | Independent body containment | Person-relative curves / executed rear plans |
| --- | --- | --- | --- | --- | --- | --- |
| 7.4 m, extension 0.15 m | 3/3, 24.0167 s | 0.282323 m | 0.002291 m | 0.127754 | 239/239; max center error 0.225544 m | 0 / 0 |
| 7.8 m, extension 0.15 m | 3/3, 27.2333 s | 0.101784 m | 0.002221 m | 0.250798 | 271/271; max center error 0.226593 m | 0 / 0 |

Both failed the unchanged walking-arc/execution/encounter checks and the new
timed/location component. Their crossings of the observed human plane occurred
at source times 26.9481 and 30.8874 s, after transverse walking. Nine/seven fresh
attempts found no passage within the three-second kinematic scope. A subsequent
brief inconsistent motion fit invalidated the selection before any curve froze.
Later normal/checked fallback motion is not person-relative reference execution.
The later trigger also delayed departure while ordinary safety intervention
slowed the approach; changing a trigger does not guarantee a coordinated meeting.

All independent source/fit/raw-velocity/physical checks passed separately:
275/306 source-window fits and 40/79 policy records reproduced. There were
84/98 accepted visual instances and **zero fresh complete four-factor
assessments**. Human/body/factor priors and fixed physical lambda were unchanged.
Original reports remain failed and byte-identical:

- `native-person-rear-motion-left-v1/report.json`: SHA256
  `93d0343aab20ad1dc9022f342999ce18db90afd263def6054f9c32d3fe509aea`.
- `native-person-rear-motion-left-v2/report.json`: SHA256
  `6df20eeb505cbb7a401f6cdb29593ce2278916ba38457ffc0b6293a419de82e4`.

Each native launcher returned zero but the saved gate-failure `RuntimeError`
was recorded. The saved acceptance report, not launcher status, establishes
failure. A trusted first-selected numeric snapshot in each also admitted no
candidate from the old checked rearward pool on that original physical field.
That bounded replay is not a proof that every safe maneuver was impossible.

The retry-state correction distinguishes **no curve created yet** from an
**invalidated frozen curve**. A missing/rejected qualification or source gap
before the first curve may be followed by a new fully qualified current-source
attempt. Nothing is borrowed through that gap. The original selection timestamp
and eight-second cap remain; the new curve's source is that real new observation.
An actually frozen curve still loses eligibility permanently for its original
selection after a gap, incompatible motion, identity loss or expiry. Expired
unfrozen selections also cannot restart. Field temporal continuity, positive
identity, fit acceptance, physical factor/command freshness and every exact and
analytical check remain independent requirements. This repairs premature search
lockout; it does not establish successful navigation or justify longer forecasts.

## Wider actor component

A separate extension 0.25 m / y8.4 fixture starts (27.55,8.4), crosses to
(30.05,8.4), then departs to (30.05,12.5). A proposed robot-y7.9 trigger changes
only actor timing. Robot and human speed limits remain unchanged. Authored
raster margins were 0.052145 m left and 0.077145 m right. These are independent
abort/evaluation data, never controller/factor inputs.

At 2422a87 the new parked native route passed actor and sensor gates with zero
wheels: 8 fresh idle, 14 crossing, 26 departure and 2 endpoint associations;
median speed 0.518044 m/s, minimum actual furniture clearance 0.033824 m and
maximum collider alignment error 2.86e-8 m. The route is tight and bounded;
passing it does not make it an ordinary wide-aisle scenario. Original report:
`native-person-rear-parked-left-v3/native/validation.json`, SHA256
`5065e17ffb988a1bb6038266fafe33b547e7fa758c21441ee5d5815ee42cf777`.
The independent 60-frame zero-wheel controller gate also passed. No v3 wheel
trial was launched before the retry-state correction. Actor/perception code is
unchanged by that correction, but a new independent stationary gate and the
coupled trial's own 60-frame gate are required for the corrected controller.

Keep the accepted old stop/go fallback and all main/long-service/PR117 holds.
No person-relative native success, primary-comparison improvement or complete
fresh physical perception follows from these component/arrival results.

The retry-state correction passed **2,754 unit tests in 104.28 s**, including
75 focused rear/checked-wait checks in 9.77 s. Ruff across `robot`, `scripts`
and `tests`, compilation and whitespace checks passed. Corrected-controller
native acceptance remains pending; old native results retain their original
source revision and labels.

## Corrected coupled result: curve available, rear pass failed

At **4f0054f** the new independent stationary gate and the coupled trial's own
60-frame zero-wheel gate passed. The wider actor/perception code was unchanged
from its separately passed parked component. This trial used the .25 m extension
and y7.9 trigger described above, retaining every robot limit, physical prior,
source/command clock and safety check. It is a different fixture from v1/v2,
not a matched causal test of the retry correction.

One curve froze from source **23.4 s**, retaining selection **22.5 s**, predicted
passage/expiry **26.4 s** and the original three-second hypothesis. All 31 active
fields used current accepted source-window policy motion; no raw fallback was
needed for that reference. Independent checks reproduce the frozen path, current
motion compatibility, original epochs, rear region and unchanged actuator bounds.
There were ten first-curve attempts. This run does not establish that a natural
pre-first-curve gap was exercised; that correction has dedicated unit coverage.

All **30** planner requests carrying the curve were rejected with
`unsafe_configuration_boundary`: **17,280** candidates tested, zero exact-feasible
horizons and zero executed rear commands. The first trusted original numeric
snapshot at source23.5 exactly reproduces the entire rejected command sequence
and candidate counts. Its query cell is occupied by human-envelope track
3000000073 in both adjacent yaw slices; sampled h and guidance are zero. The
snapshot SHA256 is
`233ae08d501c13c4cffbd7b04dd855b4a28f3cf2e733775767320aa7c7e8f0ea`.
This identifies a concrete geometry/admission limit in these requests, not proof
that every possible safe maneuver or planner is impossible. A guidance-weight
change cannot override this exact occupied-configuration rejection.

Preserve the distinction between physical CRA and the existing stronger-human
control ablation: at that snapshot physical CRA phi was **0.68359375**, while
requested human boundary flux **-1.7** maps to applied field risk **1**; physical
lambda remained **1.25**. The latter is an explicit verified-human control
override, not a newly measured physical risk or automatic unknown-factor rule.
Neither those values nor the .65 m body-envelope prior were changed here.

The robot subsequently reached all three configured route guides and the final
goal in **27.5167 simulation seconds**, then stopped with **0.002316 m** drift.
The reported analytical minimum was .239761; 274/274 independent body-containment
rows passed with maximum center error .192470 m. However, minimum independent
clearance was only **.026425 m**, just .006425 m above the unchanged .02 m gate.
Positive clearance is not rear-pass acceptance, and this tight result is not
recommended for the final demo. The curve/execution and stricter timed/location
gates all failed. Actual transverse walking was 21.8667..26.25 s; the robot
crossed the observed walking plane at31.2907 s, **5.0407 s after** that window.
Actor trace data is evaluation only, never a runtime maneuver input.

There were 274 committed fields (215 while translating), 86 accepted visual
instances and **zero fresh complete four-factor assessments** among 18,017 object
assessments. Adult mass/body-contact and unknown-factor priors remain explicit.
Independent audits reproduced 293 motion fits and 47 policy records, retained
raw physical velocities, verified source/identity/field clocks and reproduced
the original zero-row execution metric. All original reports remain unchanged.

Original failed report SHA256:
`002e6bb27ca683477cc172fb06785a3077e543cc731400fc755273d4c4877985`.
The launcher returned zero with the expected saved gate-failure RuntimeError;
this is a failed native experiment. No mirrored person-relative wheel trial
was enabled. Tests remain 2,754 units / 75 focused / Ruff / compilation passed
at the frozen functional source, not a passing live rear-pass claim.

Artifacts under `outputs/paper_flux/`:

- `native-person-rear-motion-left-v4/` retains original reports, sensor/actor
  traces, independent source/curve/fit/policy audits, the exact unavailable
  snapshot replay and the search/timing summaries.
- `native-person-rear-motion-left-v4-trajectory.png` distinguishes the proposed
  purple curve from the actual blue path. A proposed curve is not execution.
- `native-person-rear-motion-left-v4/human-navigation.mp4` is the native video;
  all 355 frames are readable, with distinct start/middle/end frames checked.
- `person-relative-native-comparison-v2.png` retains the first two failed
  timing-fixture paths. These are not matched causal comparisons.

The bounded recommendation is the previously accepted stop/go configuration
for perpendicular crossings, with normal fixed-lambda guidance for parallel
departure. Further pronounced rear passing needs an explicitly scoped earlier
predictive-planning or more open encounter experiment. Do not keep tightening
the pedestrian trigger, weaken geometry/uncertainty or relabel failures to obtain
a curve. The primary comparison, main/long-service/PR117 holds, prior/oracle
provenance, RGB-clock approximation, paced-not-real-time and renderer limitations
remain unchanged. PR181 stays a development draft.
