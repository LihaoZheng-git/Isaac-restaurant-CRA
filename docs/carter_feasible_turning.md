# Carter turning feasibility without early preparation

The user's latest decision excludes quiet-person early preparation from the
selected restaurant demonstration. Normal obstacle-unaware waypoint motion,
CRA field policy and online MPC/analytical intervention remain. A stationary
person remains physical occupancy; excluding preparation never removes them.
Observed approaching motion can support the separately labeled passing
guidance. The existing explicit guidance-tracking objective remains an ablation,
not unmodified Safe-SAGE or the primary comparison.

The accepted earlier route-guide run at `3fb7e2e` reached all three guides and
the final goal, with a 0.222896 m nearby bend and a 2.383333 s translation hold.
It recorded 27 fresh `unsafe_configuration_boundary` planner rejections. That
status comes from the **initial** configuration query, before optimization;
it does not establish that different turn seeds at that already blocked state
could pass. The first low-progress guidance reference preceded the boundary
rejection. Earlier five-second replay buckets overwrote the exact failing input.

First preserve the first low-progress and initial-geometry rejected planner
requests, using the existing opt-in trusted local replay path. Export stays
inside worker timing. Snapshots retain original source epochs, are never
loaded by runtime control, and do not change commands or safety checks.
Reproduce with early/rear preparation and predictive redesign disabled, keeping
the original actor route/onset, CRA/factor/body policies, lambda 1.25, nominal
0.20 m/s, MPC 0.25 m/s, solver budgets and all original checks. This diagnostic
does not by itself constitute a fix or a new accepted native trial.

Any implementation must address a demonstrated planning/geometry cause, retain
all occupancy and margins, and pass exact DTCBF, dense configuration checks,
per-step analytical filtering, original freshness/identity guards, independent
clearance/containment, arrival and stopping. Numerical completeness is not
fresh four-factor perception. Revalidate stationary before new navigation.
Keep the separate original smoothness target (actual nearby hold at most 0.5 s
and at least 50% reduction) and preserve every failed report. No primary,
main, PI-demo, long-service or PR117 hold is released by development alone.

## Reproduction and first bounded candidate

At `d892784`, the fresh original human/perception component and independent
60-frame stationary gate passed. The unchanged no-preparation motion invocation
reached all three goals in 21.333333 simulation seconds, clearance 0.352025 m,
stopping drift 0.002406 m and minimum analytical margin 0.080218. It remains
**FAILED overall**: four supported moving-kinetic rows versus the required five.
Its actual translation hold was 2.383333 s, reproducing the original pause.
The saved report is never retroactively passed:
`135b0b8fa9bd3d9eb5f4bcf1d1aed72bc8366e3af4ea23bde6f4151e476ed4c6`.

The preserved first low-progress request at source14.2 had a free starting
configuration but no accepted horizon in the tested bounded search, including
a maximum-speed0.25 m/s probe. The first initial-geometry rejection at source15.3
was occupied by the human body prior (ID3000000000). Its interpolated h was
positive; the mandatory occupied-cell rejection correctly overruled that value.
There is no additional free-cell boundary ring: the Dirichlet mask includes
occupied configuration cells and the outer grid edges.

An offline **body-only component** retained the exact radius0.65 m and footprint
with physical margin0.05 m. The recorded starting pose had continuous
rectangle/circle margin0.199792 m. Its active-yaw raster cell was occupied at
resolution0.10 m and free at0.05 m. This shows numerical conservatism in that
component, not that every physical obstacle is clear or that a navigation
trajectory exists. Original rasterization conservatively accounts for both
obstacle/query cell extents and the yaw slices; none of those checks is bypassed.
See the immutable `turning-rejection-audit-v1.json` in the diagnostic directory.

The next candidate exposes `--grid-resolution .05` (legacy default remains.1),
with unchanged bounds, inflation, footprint, body radius, lambda1.25, nominal.20,
MPC maximum.25 and all physical/source/arrival/stopping checks. The complete
field retains every observed return and unassessed occupancy. The actual
configuration is saved separately and in the final report. More computation is
expected; freshness and optimizer-age limits remain unchanged.

The live CLI also separates the existing explicitly approved three-second
adult-mass/current-body retention option from early preparation. It still
requires explicit body identity and all original provenance/expiry checks;
its implementation is unchanged. The next trial enables it to address the
reproduced evidence gap, not to lower CRA risk or change physical measurements.
Grid refinement plus mass retention is a declared **two-option** candidate,
not a single-variable causal comparison. New no-preparation mass component,
fine-grid independent and own60 gates precede its wheels. Native outcome is
pending; early/rear preparation and predictive geometry/redesign stay disabled.

## Finer-grid result and numerical rescue follow-up

At `a1e5c11`, the fresh no-preparation component passed, including85 retained
mass updates from26 original accepted priors. The independent and own60-frame
fine-grid gates passed. Native `motion-mass-grid005-v2` **PASSED navigation and
evidence**:3/3 goals in21.216667 simulation seconds,22 supported moving-kinetic
rows,0.214108 m nearby bend,0.371309 m physical clearance,0.002367 m stop drift,
0.322381 minimum analytical margin and211 fields. Source stayed clean through
shutdown. Report SHA:
`c7a4d90ac0848bae496bd7c9a4bbc9cc202d4e98e7ba918f806320b39e5f2e2b`.

It **FAILED smoothing**:2.466667 s actual hold, versus original2.383333 s.
Grid refinement did not fix smoothness. Preserve that failure and its plot;
neither the component counterfactual nor successful arrival overrides it.

The first low-progress native input at source14.4 is still initially free.
Its original planner returned a stop. A broader OFFLINE six-dimensional search
found a moving horizon under unchanged exact DTCBF/dense/current-analytical
checks; this establishes a search limitation on this specific input, not
global feasibility or an executable stored route. A fixed1,050-candidate
three-phase actuator pool found no exact-safe candidate, but its closest
infeasible seed let **one original20-iteration nonlinear rescue** find a fully
safe moving horizon. No solver budget, constraint, uncertainty prior or cost
was changed. The first command changed from(0,0) to(0.167417,-0.006957), and
the actual analytical filter left it unchanged (margin0.174595). Full-horizon
minimum DTCBF margin was0.000014241; revalidation passed with unchanged field
bytes and original timestamps. This is a design regression, not a native result.
Original trusted input SHA:
`29d31598651d08321c3d8a80a52ea8e2d32f1f791786b17a5bd209573e1e4935`.

The optional `--mpc-tapered-rescue-seed` initializes the EXISTING nonlinear
rescue after the original SQP restoration fails. Its data-independent pool
scales only the existing actuator bounds and horizon: three phases with
forward speed tapered toward braking and small turns of both signs. Candidates
must satisfy the original current analytical halfspace; ranking uses existing
exact constraint violation then the unchanged objective. Endpoint-only ranking
cannot override dense geometry. A selected infeasible seed is only a numerical
starting point, never permission to execute. One original-budget rescue and
all existing final/current-field/analytical/source/physical checks remain.
The option defaults off for every legacy caller. Early preparation stays off.

Reproduce the archived numeric counterexample (explicit trusted LOCAL pickle
only; matching repository/environment required):

```powershell
python scripts/evaluation/replay_carter_rescue_seed.py `
  --replay-directory outputs/carter_feasible_turning/motion-mass-grid005-v2/planner-replays `
  --output outputs/carter_feasible_turning/integrated-tapered-replay-v1.json `
  --trusted-local-replay
```

The command refuses existing output and hash mismatches. Default full-command
parity and the enabled exact revalidation are recorded separately. The third
native candidate adds ONLY this flag to the fine-grid/mass-retention invocation,
with a source audit and new independent/own60 gates; the unchanged mass/actor/
perception component can be reused only by exact fingerprint proof. Outcome
is pending. All original smoothness, evidence, curve, arrival/stopping and
primary/main/PI/long-service/PR117 holds remain.

## Tapered-seed result and forecast-Jacobian consistency

At `faf5aa7`, independent and own60 gates passed. Native
`motion-tapered-seed-v3` **PASSED navigation/evidence**:3/3 guides/final arrival
in21.116667 s,21 supported moving-kinetic rows,0.237685 m nearby curve,
0.427766 m clearance,0.002453 m stop drift,0.350702 minimum analytical margin
and207 fields. **Smoothing FAILED**:2.383333 s hold, essentially unchanged from
the original baseline. Five selected numerical-seed attempts produced zero
accepted plans. The offline moving witness does not establish a live benefit.
Preserve original report SHA:
`503c99cfacda36d999ff604802936a51f784db5239c790560542bcd1263f243b`.
The external read-only replay observer retained the first failed attempt at
source14.2, SHA`a708b3d8cabef1f76ae97de565d2324e253eb1a65de7ff2c5b1c081c5915f01a`;
it captured no successful selected seed. It never changed control or source time.

A separate numerical audit found a concrete derivative inconsistency. Existing
constraint VALUES are `h(q) + t * scaled_ht(q)`, with the scaled temporal term
sampled at each predicted state. Restoration and nonlinear rescue previously
used only `gradient(h)` for their Jacobian. On the archived source14.4 example,
the maximum discrepancy against finite differences was0.046266 (maximum
numerical Jacobian entry0.144141). This is not proof of a sole stall cause.

The default-off `--mpc-consistent-temporal-jacobian` adds the missing
`t * gradient(scaled_ht)` analytically to restoration/nonlinear Jacobians.
It differentiates the existing spatial/yaw interpolants, guidance-vector norm,
spatial-gradient norm, regularizer, clamp branch and raw temporal field. It
changes NO barrier values, source epochs, cost, guidance vector, uncertainty
policy, actuator bounds or acceptance thresholds. Grid/yaw/clamp knots can
remain nondifferentiable; derivatives follow the active branch. There are no
finite-difference gradients in runtime (finite differences are tests only).

On the same archived source14.4 input, the analytic complete Jacobian matches
finite differences within1.06e-9 and restores a moving horizon with the same
solver budgets, without a numerical seed. A different source14.2 example still
failed offline, with and without seeding; retain both failures. Numerical
consistency is not global convergence or native smoothness acceptance.
The next separate native trial adds only the Jacobian flag to the third
variant after new independent/own60 gates. Outcome remains pending and every
original gate/hold remains. The replay CLI accepts `--variant temporal_jacobian`
for an unchanged recorded baseline and preserves its other numerical options.
