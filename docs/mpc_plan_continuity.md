# MPC continuity at a moving-human encounter

The computed-CRA baseline arrived with a 0.8833 s actual translation hold.
This branch diagnoses its remaining MPC interruptions without changing CRA,
the obstacle-unaware nominal, lambda 1.25, body/footprint margins, six 0.2 s
prediction intervals, command limits, source clocks or exact acceptance.

## Numerical proposal recovery

The existing QP path discarded the last iterate when OSQP exhausted its 1,000
iteration budget. In the archived source-14.2 request, that discarded finite
iterate can support subsequent exact restoration: the original stops; the
prototype returns (0.249937 m/s, 0.392771 rad/s) with full current-field horizon
revalidation and an unchanged analytical-filter command (margin 0.145733).
This is a same-input numerical result, not a native smoothness result.

The separate `--mpc-unfinished-qp-proposal` option makes those iterates available
only as numerical proposals. Primal/dual infeasible, nonconvex, interrupted,
missing, wrongly shaped and nonfinite outputs remain rejected. Finite commands
are clipped to the existing actuator domain, then follow the original exact
DTCBF, dense geometry, restoration and safe-objective line-search checks.
Current-pose revalidation, field/candidate/execution expiry and analytical CBF
remain mandatory. A failed solver status never authorizes wheel execution.
Original solver status, residuals, iteration count and proposed first command
are reported separately from final trajectory acceptance. No convergence or
global optimality claim follows. Legacy/default behavior remains disabled.

The QP and nonlinear per-call budgets remain 1,000 and 20. An unfinished
proposal can permit subsequent original bounded SQP iterations instead of an
early abort; total solver work need not remain identical.

## Diagnostic window

`--planner-replay-source-window START END` retains every planner request in a
bounded original field-source window (maximum five seconds/64 requests).
Original input states, fields and SHA-256 checks remain in trusted local
archives; files never enter control. Recording is included in worker time.
The source-13.5..17.5 capture at 740352b followed an independent and its own
60-frame stationary gate. It intentionally ended at 11 navigation seconds;
its duration-limit report is failed, not arrival acceptance. All runtime
source hashes remained unchanged.

The window separates numerical failure from geometry: source 15.3's nonlinear
candidate has near-zero DTCBF margin but future occupied configurations. Neither
the QP prototype nor a 1,125-candidate offline pool recovered that request.
This does not prove infeasibility and does not justify relaxing geometry or
tolerances. The candidate's live test must establish whether earlier valid
motion avoids entering that difficult state. Retaining nonlinear iterates did
not recover the first archived failure and is not integrated.

## Validation status

Focused regression, paired integrated replay and full suite precede source
freeze. New independent and own 60-frame stationary gates precede matched
navigation. The original 0.5 s maximum actual translation hold and at least
50% reduction versus the original 2.3833 s baseline remain the smoothing target.
Actual arrival, nearby walking curve, independent clearance/containment,
current kinetic data and braking checks remain mandatory.

Human mass/contact/body and unavailable potential/thermal inputs retain their
declared priors. This remains a paced, prior-assisted guidance-tracking ablation,
not complete fresh four-factor perception, real-time control, primary-comparison
superiority or exact Safe-SAGE paper equivalence. All main/PI/long-service and
PR #117 holds remain until their own requirements are met.

## Native outcome at 6ef5448

The independent stationary gate passed 60 frames in 220.99 wall seconds. The
matched full trial passed its own 60-frame gate, reached all three guides/final
goal, and stopped. Runtime hashes and Git source stayed unchanged throughout.
The 11-second diagnostic also retained its original failed duration-limit
report; it is never included as navigation acceptance.

| Metric | Previous computed CRA | QP proposal option |
|---|---:|---:|
| Native navigation acceptance | Pass | Pass |
| Navigation simulation time | 19.6500 s | 21.0833 s |
| Longest actual nearby translation hold | 0.8833 s | 2.3500 s |
| Longest nearby translation below 0.05 m/s | 0.9333 s | 2.3833 s |
| Nearby walking curve | 0.2372 m | 0.2455 m |
| Minimum independent clearance | 0.4408 m | 0.4295 m |
| Stopping drift | 0.002417 m | 0.002452 m |

**Smoothness failed and regressed in this trial. Do not promote the option as a
smoothing fix or replace the accepted previous demo.** It remains disabled by
default and PR #199 stays draft. This single development run does not establish
that the option always worsens motion, either.

The run recorded 204 live fields (186 while translating), 21 supported moving
kinetic rows and a minimum analytical margin of 0.130698. Applied human CRA was
0.558060..0.683594. No quiet-person preparation or future-human geometry option
was enabled. There were zero generic fresh visual instances/complete fresh
four-factor updates; the separately validated human priors/current-motion
path remains explicit.

Seven worker results included unfinished QP proposals, four were accepted by
the worker-result gate. Original optimizer-request-clock joining identifies
five issued translating command rows from those requests; this does not mean
every proposed iterate was executed. Linear command total variation increased
from 2.4021 to 2.8950 m/s over the respective routes; maximum linear command
step increased from 0.2000 to 0.2498 m/s. The option is not smoother by those
descriptive command measures. These are not calibrated physical jerk metrics.

The source-15.3 rejected-geometry audit identifies 215 padded dense queries in
the human body's configuration-space occupancy (owner 3000000000, applied risk
0.683594), despite clear unperturbed endpoints. This localizes one failure to
the existing position/yaw robustness checks, not missing human tracking or a
proof that the margins should be reduced. End-point/dense distance-constraint
rescue prototypes and a finite 1,125-command search did not recover the exact
request under the original acceptance gates. These remain offline, unintegrated
negative results and do not prove infeasibility. Future work should address
joint geometry/CBF trajectory feasibility before any further wheel retry.

Source validation: 3,216 full unit tests in 113.35 s; 141 focused tests in
18.92 s; Ruff, compilation and diff checks passed. Six integrated OFF/ON paired
replays retain default full-command parity, original field bytes and epochs;
two early failures recover valid moving horizons, two valid controls remain
unchanged, and two later geometry failures remain rejected.

Artifacts are under `outputs/mpc_plan_continuity/`:

- `motion-candidate-v1/continuity-comparison.json/png`: original smoothness
  scoring, matched-configuration audit and comparison plot.
- `motion-candidate-v1/command-smoothness.json`: issued-command variation and
  original request-clock join.
- `motion-candidate-v1/navigation-only-v1.mp4`: unchanged 10 fps, omitting only
  6.9 seconds of zero-command setup; all navigation pauses and stopping remain.
- `integrated-replay-v1.json`, `rejected-geometry-owners.json` and the bounded
  `motion-diagnostic-v1/planner-replays/window-*` archives preserve diagnosis.

Original candidate report SHA256:
`8d815096c2a1166ee17c3178d0ab910430f7e1c390d9534f0c07bcf27fbd8803`.
All prior reports, the accepted previous demo and release holds remain intact.

## Offline pause diagnosis after the regression

The user authorized diagnosis only. No runtime/controller source, simulation
configuration, checkpoint, field, CRA score or physical margin was modified;
no new Isaac or wheel trial was run. All numerical searches use SHA-256-checked
local snapshots from the failed-smoothness candidate above, with their original
epochs. Diagnostic trajectories are **not runtime references**. A future online
method must generate its own candidates from the current observations and field.

The diagnosis identifies three distinct stages, all in original simulation
source time (navigation command logs use an 8.2 s offset):

| Saved field time | Finding under the original checks | Consequence |
|---|---|---|
| 15.1 s | The guidance ablation selects a turn-first reference and returns approximately `(0.00017 m/s, -0.52009 rad/s)`. A constant moving arc passes the complete 1.2 s geometry/DTCBF and analytical first-command checks. Supplying that arc only as an offline numerical initial guess leaves the original turn-first result unchanged. | The auxiliary reference selection can suppress translation even when a moving horizon exists. It replaces the initial guess with its selected reference. This is separate from solver convergence. |
| 15.3 s | The native planner fails. A bounded offline search finds a varying slow sequence passing every original full-horizon check, with minimum DTCBF margin 0.002140 and unchanged analytical first command. The unmodified planner, diagnostically initialized with this sequence, returns a checked command of approximately `(0.04872 m/s, -0.07367 rad/s)`. A second request at the same field epoch also recovers. | There is a numerical search/initialization miss on these particular frozen inputs. The witness is not a new command policy or a live smoothness result. |
| 15.4, 16.2 and 17.2 s | Every command on a 51-by-65 first-command grid fails padded geometry. Conservative reachable-box bounds additionally place one mandatory 0.1 s padded query entirely inside human-owned configuration cells for all allowed constant first commands `v in [0, 0.25]`, `omega in [-0.8, 0.8]`. The original checker confirms all 561 enclosure spot checks per bound. | Under this recorded field, kinematic model and existing checker, improving solver convergence cannot recover an accepted first command. This is not a physical collision claim, universal navigation impossibility, or justification to reduce margins. |

The 15.3 s witness uses commands approximately
`(0.04286, -0.05148)`, `(0.02957, -0.16356)`, and `(0.08039, -0.19643)`,
each for two original 0.2 s intervals. It was found in 720 evaluations, roughly
1.47 wall seconds after loading; this diagnostic search is not a real-time
algorithm. The original current-state full revalidator and analytical filter
both accept it. It also passes at the second actual planning state for the
same 15.3 s field. It fails the next 15.4 s field at that field's recorded robot
state. These independent snapshot tests are not a counterfactual sensor replay
or proof of feasibility across the intervening motion.

An additional audit checks the auxiliary reference generator's forecast
analytical half-spaces at all six command starts. The 15.1 s moving arc passes
all six (minimum 0.131307). The 15.3 s slow witness fails the later forecast
half-spaces (minimum -0.024566), despite passing full MPC DTCBF/geometry and the
current analytical filter. It therefore demonstrates an MPC initialization
miss but is not a ready-to-use checked guidance reference. Future analytical
interventions and changing fields must remain part of validation.

A smaller deterministic audit of 165 constant arcs at each of five recorded
states reinforces the earlier reference-selection finding. At 15.1 s, 49 arcs
pass full geometry/DTCBF, all six forecast reference half-spaces, and the current
analytical filter. One uses **0.20 m/s and -0.25 rad/s**, with minimum DTCBF
margin 0.050688 and minimum forecast half-space margin 0.320223. It uses the
existing nominal forward speed, not a higher robot limit. At 14.8, 14.9 and
15.2 s there are respectively 53, 52 and 46 passing arcs; at 15.3 s this finite
constant-arc set has none. The audit takes roughly 0.22..0.28 wall seconds per
loaded request. This identifies an online-shaped candidate family to test,
not a real-time implementation or a closed-loop benefit. Its nominal-cost
ranking is diagnostic only; it does not silently replace the current guidance
tracking objective or establish which arc should be executed.

A separate 57,960-evaluation search requiring at least 0.06 m/s throughout
the horizon found no passing sequence. Its best candidate still violated the
unchanged DTCBF tolerance. Preserve this negative result: neither it nor the
finite first-command grid proves general infeasibility. The low-speed witness
also does not establish useful physical translation or satisfaction of the
actual-speed smoothness gate.

The later reachable-box certificate concerns the yaw-22.5-degree configuration
slice selected by an existing uncertainty query. For example, at 15.4 s, the
0.1 s query uses 55% of the original position/yaw padding, and its reachable
box covers rows 64..65/column 66, wholly owned by the human envelope. The robot's
unperturbed heading is about 9.34 degrees. This distinguishes the nominal pose
from its padded, discretized configurations; it is not proof that the
uncertainty bounds or yaw discretization are wrong.

Clock alignment was independently checked against all 1,265 recorded issued
commands and their next-physics-step truth entries. In source time 15.3..17.9 s,
142 commands wait for a fresh valid plan, two record new-field invalidation,
two record a stale-field stop, and ten have no stop reason. The actual 2.35 s
translation hold remains unchanged. Fresh positive human support continues,
and applied human CRA stays 0.68359375 throughout the audited 14.7..17.5 s
body observations. A passing-side transition is also recorded; this audit does
not isolate its causal contribution. Human center truth is used only for an
independent recorded error audit, never for the feasibility search. The human
was already walking by source 11.3 s, before all these diagnosed states.

The next bounded implementation should investigate online generation and
selection of jointly geometry/CBF-checked moving references **before** a
turn-first reference becomes the strong tracking target. Existing guidance
search is only attempted when no basic reference is found, so finding any
turn-first reference currently prevents that broader search. Candidate search
and smooth command preference must never authorize retaining a plan rejected
by a new field. Addressing the later geometry-blocked state requires avoiding
entry into it using current evidence; any predictive geometry or uncertainty
policy change remains a separately justified research decision. A cross-plan
command smoothness penalty is secondary and cannot resolve an empty accepted
first-command set. No offline path should be inserted into live navigation.

Validation for this diagnostic: 180 relevant configuration-space, paper-filter,
unicycle MPC and guidance tests passed in 19.08 s. Thirteen recorded states each
used 3,315 command samples and 23 original-checker parity comparisons. Three
reachable-box cases each verified 561 exact-query enclosure samples. Field-byte
immutability assertions and original snapshot hashes passed. Ruff passes for
the diagnostic scripts and the runtime/test tree. The first exploratory audit
was corrected to invoke the original dense checker directly when a different
endpoint already fails; a search setup assertion was also corrected for the
helper's included time-zero state. Only corrected complete outputs support the
conclusions above; neither correction changed runtime or a native report.

Artifacts in `outputs/mpc_pause_diagnosis/`:

- `pause-diagnosis-v2.png`: recorded speed and a later bounded geometry blockage.
- `first-interval-candidate-v2/`, `first-interval-transition-v1/`: finite-grid
  and original-checker parity results (the incomplete `v1` is retained).
- `joint-candidate-15_3-v1.json`: fully checked slow moving witness.
- `joint-candidate-15_3-min06-v1.json`: negative faster-horizon search.
- `witness-recheck-v1.json`: original solver with diagnostic initial guesses
  and rejection at the next recorded field/state.
- `witness-guidance-v1.json`: extra forecast half-space checks, including the
  slow witness's later failures.
- `constant-guided-arcs-v1.json`: 165-arc audits and the 0.20 m/s moving candidate.
- `interval-bounds-v1/`: original first-command domain enclosure certificates.
- `timeline-v1.json`: source-clock alignment, stop reasons and observed bodies.
- `diagnostic-manifest.json`: source/report hashes and reproducibility helpers.
