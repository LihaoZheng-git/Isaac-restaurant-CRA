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
