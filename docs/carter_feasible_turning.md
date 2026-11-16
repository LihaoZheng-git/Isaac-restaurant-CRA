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
fine-grid independent and own60 gates precede its wheels. Its completed outcome
is recorded below; early/rear preparation and predictive geometry/redesign stay disabled.

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
perception component can be reused only by exact fingerprint proof. Its completed
outcome follows below. All original smoothness, evidence, curve, arrival/stopping and
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
variant after new independent/own60 gates. Its completed outcome follows below;
every original gate/hold remains. The replay CLI accepts `--variant temporal_jacobian`
for an unchanged recorded baseline and preserves its other numerical options.

## Completed fourth trial: navigation passes, smoothness still fails

The independent and own60-frame stationary gates passed at frozen source
`2c8443683107b4a4a55e00674e4ae2e0c3cf60e5`. The fourth invocation changes only
the Jacobian option versus the third; the actor/perception/mass component was
reused after exact unchanged-source fingerprint checks. Source remained clean
through native shutdown. All four new native reports and the earlier accepted
baseline retain their original hashes.

| Variant | Native navigation/evidence | Actual nearby hold | Nearby bend | Minimum physical clearance |
| --- | --- | --- | --- | --- |
| Earlier accepted no-preparation baseline | Pass | 2.383333 s | 0.222896 m | 0.357913 m |
| Original reproduction | **Fail: 4 kinetic rows < 5**; final arrival recorded | 2.383333 s | Separate recorded curve gate passed | 0.352025 m |
| Fine grid + approved mass retention | Pass | 2.466667 s | 0.214108 m | 0.371309 m |
| Add numerical rescue initialization | Pass | 2.383333 s | 0.237685 m | 0.427766 m |
| Add complete forecast Jacobian | Pass | **2.500000 s** | 0.239824 m | 0.429644 m |

The fourth run progressed three route guides and reached the mandatory final
goal in21.233333 simulation seconds. Intermediate guides are route references,
not three independently mandatory exact checkpoints. It recorded205 live
fields,183 while translating,19 supported moving-kinetic rows, minimum
analytical margin0.255855 and stopping drift0.002434 m. Native report SHA:
`0916a177c14a19f8af5624f383ac8aeba02af9d270fa4f81deff442bf0ad2ac1`.
Launcher exit0, no traceback, and native `passed=true` were independently checked.

The separate smoothing gate **fails** both requirements: hold at most0.5 s and
at least50% reduction from the original2.383333 s. Actual root speed stayed
below0.02 m/s for2.5 s (source15.533333--18.033333), about4.9% longer than the
original; the below0.05 m/s interval was2.55 s. A safe moving plan on one request
does not establish smoother closed-loop navigation.

No quiet-person preparation was used. Maximum route-relative lateral excursion
before actual human walking was0.000433 m. Walking began at source11.3; the
robot first exceeded2 cm of lateral excursion at source12.683333. These are
evaluation-only root-truth measurements, not controller inputs. The largest
nearby walking bend occurred with simultaneous center separation1.765568 m;
center separation is distinct from the physical-clearance metric.

This remains a prior-assisted, separately reported guidance-tracking/human-flux
experiment with lambda1.25, original actuator bounds, body prior0.65 m,
physical margin0.05 m, QP1000/nonlinear20 and original source/current-field/
exact DTCBF/dense/analytical/physical/arrival/stopping checks. The generic
perception counter records zero accepted current instances during motion and
zero fresh complete four-factor updates; the separate human specialist,
retained adult-mass prior and current motion have their own recorded audits.
Successful navigation is not fresh four-factor perception, real time, primary
comparison superiority or exact Safe-SAGE equivalence.

## Remaining planning limitation

The read-only export observer captured a successful native restoration at
source14.2 (archive SHA
`714cb1714fe01dd488822ad37d5fbbc76033969f72506176413d217c8ab21f28`).
An exact same-input offline toggle reproduces the native full command sequence
with the corrected Jacobian: first command(0.244067,0.390931), full-horizon
revalidation passed and the actual analytical filter leaves it unchanged
(margin0.133251). The minimum DTCBF margin is-3.11604e-7, within the original
unchanged1e-6 numerical tolerance. Disabling only the correction returns a stop.
Field bytes, original epochs and solver budgets remain unchanged. This isolates
a numerical benefit on this input, not a whole-encounter smoothness benefit.

At the later source15.3 request, restoration still fails full acceptance even
though its endpoint DTCBF margins alone are within tolerance. Dense geometry
remains mandatory. A separate900-candidate two-phase actuator search found225
geometry-free candidates but no horizon passing all original checks. That finite
search does not prove that every acceptable trajectory is impossible.

A diagnostic shortened the checked prediction from1.2 s to1.0/0.8/0.6 s on that
same archived request. The1.2/1.0 s variants failed;0.8 s found a moving command
(0.2,-0.2). **This is not integrated or native validation**: shorter prediction
reduces the future coverage being checked, so it cannot satisfy the original
1.2 s gate by substitution. No CLI range, acceptance threshold or prediction
policy was changed for this diagnostic.

The evidence points to remaining planning/forecast limitations beyond the
corrected derivative. The current future values are a first-order field-time
forecast; they are not a time-indexed field with validated future human motion.
A further prediction redesign must use observed approaching motion, preserve
the exclusion of quiet-person preparation, justify start/stop uncertainty and
association/fragment coverage, and pass new stationary and live validation.
The existing extra predictive-envelope ablation previously failed live coverage
and fragment continuity; it cannot be enabled on this result. No claim that a
new predictor will eliminate every safety yield is justified.

Validation for the runtime source: **3,147 unit tests passed in107.45 s**, including
11 new analytic/finite-difference tests; whole-repository Ruff, compilation,
both CLI help paths and diff checks passed. No dependencies were upgraded.
The numerical changes remain default-off in draft stacked PR196; no main,
primary, PI-demo, long-service or PR117 hold is released.

Local native artifacts (original reports unchanged):

- [Complete native report](../outputs/carter_feasible_turning/motion-temporal-jacobian-v4/report.json)
- [Independent source/metric audit](../outputs/carter_feasible_turning/motion-temporal-jacobian-v4/final-audit-v1.json)
- [Actual-speed comparison](../outputs/carter_feasible_turning/motion-temporal-jacobian-v4/comparison-v3.png)
- [Synchronized trajectory and distance](../outputs/carter_feasible_turning/motion-temporal-jacobian-v4/trajectory-timing-v4.png)
- [Native nearby-curve frame](../outputs/carter_feasible_turning/motion-temporal-jacobian-v4/native-nearby-curve.png)
- [Navigation clip](../outputs/carter_feasible_turning/motion-temporal-jacobian-v4/navigation-only-v1.mp4): only the initial6.9 s zero-command setup prefix is omitted; all navigation pauses and final stopping remain at the original10 fps. Paced simulation is not real-time operation.
