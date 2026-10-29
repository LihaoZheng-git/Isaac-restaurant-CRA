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
