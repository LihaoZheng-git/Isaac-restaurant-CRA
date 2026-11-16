# Safe-SAGE controller realization audit

The accepted next step is to audit the existing field/MPC/analytical-CBF
mechanism before adding another guidance-tracking objective. This branch does
not add that objective, change the obstacle-unaware waypoints or retune CRA.

## Fixed field, virtual motion only

`scripts/evaluation/audit_guidance_control.py` compares directly commanded
planar `(vx,vy,omega)` and unicycle `(v,omega)` controllers in the same field.
The disk occupancy and phi .65 are authored **ORACLE** inputs. The measured
robot rectangle is shifted to the .10 m look-ahead reference; lambda is 1.25.
This has no Isaac dependency, no sensor estimation and no wheel commands.
It is a controller isolation diagnostic, not a reproduction of all paper settings.
The virtual step is .05 s with no added future-configuration padding; the
native demo uses physics substeps and its existing .025 m/.21 rad padding.
Both full configurations are recorded; virtual success is not native acceptance.

The same obstacle-unaware waypoint horizon is mapped into each command space.
Records include guidance, the planar constraint normal, its `B(theta)^T` pullback,
nominal/MPC/filtered commands, exact DTCBF margins, horizon costs and solve times.
Full predicted horizons are retained. Field hashes identify matching inputs;
manifests record the revision, source hashes and dirty status. Four seconds
without translation **or rotation** terminates a stalled diagnostic as a failure.
Arrival, executed geometry and both MPC/CBF margins are required for a pass.

Exploratory runs before the source commit are retained under
`outputs/guidance_control_audit/`. They are not frozen experimental collection:

* `initial-v1`: the planar controller initially curved left, then stopped; the
  unicycle stopped earlier at x .90 m. Both failed arrival with valid margins.
* `reference4000-v2`: increasing the planar QP budget to 4000 still stalled;
  its final predicted horizon was stationary. More iterations alone did not fix it.
* `early-seed-v3` and `mirror-v4`: using the existing unicycle progress-seed
  trigger at .90 rather than .50 completed the left/right virtual routes in
  20.85/21.10 s, with maximum lateral deviations +.773/- .735 m and valid margins.

The existing search runs when the first planned forward speed falls below a
fraction of nominal. With .50, a slowed straight horizon can remain above the
trigger until the robot is too close for the bounded turn seeds. With .90, it
can find a cheaper, exact-safe turning horizon earlier. The search still accepts
only strict improvements to the original objective. The seed family, actuator
limits, exact DTCBF, dense geometry checks and analytical CBF remain unchanged.
This bounded result does not establish global convergence or prove that all
previous stalls had this cause. The planar diagnostic remains a failed control.
The existing feasibility-pruning opt-in now also covers the progress-seed
pool: an already failed exact endpoint DTCBF can skip its expensive dense-mask
query. Every candidate selected for execution still passes the full checks.
A regression compares the exact winning horizon with pruning off/on.

Run the diagnostic using the project environment:

```text
python scripts/evaluation/audit_guidance_control.py --output outputs/guidance_control_audit/FRESH_DIRECTORY --kind unicycle --progress-seed-fraction .9
```

## Explicit live recipe

The live demo exposes `--progress-seed-fraction .9`; its legacy default remains
.5. This parameter is recorded alongside the full MPC configuration. Use the
same declared option for fixed-class and CRA comparisons. It is an online
unicycle search adaptation, not obstacle-aware nominal planning, enforced
passing side or exact Safe-SAGE equivalence. No primary baseline is silently
replaced by this recipe.

Native validation must progress through a new parked component, static wheel
navigation and crossing trials. Keep the old PR #137 failures and unresolved
track-loss stops. Do not extend human mass expiry, accept truncated SAM masks,
remove unresolved occupancy or relax field/optimizer freshness to pass. A
virtual arrival does not establish moving-camera perception, fresh four-factor
coverage, native stopping or restaurant crossing acceptance. Main and the long
service demonstration remain on hold.

## Native development checks at f39395d

`native-parked-v1` passed the 60-frame fresh-person/mass-prior/contact/current-
motion component with zero wheel commands. Potential and thermal remained
labeled priors. `native-static-v1` then passed actual service-route arrival in
20.50 simulated seconds, 1.219426 m minimum signed monitor clearance and
.002268 m stopping drift. These are prior-assisted paced Isaac checks, not
complete fresh four-factor perception.

`native-crossing-v1` retained the original .5 m/s left-to-right, 13 s actor delay
and +15 degree camera. It failed at 15.15 navigation seconds on unresolved track
484 (last observed 22.5 source seconds, missing .7 s, robot distance 1.108 m).
Only one policy-selection row survived; the moving arc was .013946 m and human
validation had zero supported moving-kinetic rows. Planner computations reached
11.83 wall seconds; perception reached 10.40 s including a 7.19 s field stage,
followed by unchanged stale/source-gap rejections. Keep all raw records and the
failed independent validators. This does not establish a native bypass.

The launcher returned zero after a separate final-report JSON failure: a
failed MPC result correctly contained a NaN unavailable margin, which the
strict serializer rejected inside `plan_events`. The canonical `report.json`
is absent; do not reconstruct missing stopping/clearance metrics or mark the
trial passed. The reporting fix stores a null margin with an explicit
unavailable status and preserves failure/fallback flags, without changing the
controller result. Subsequent runs use fresh directories.

The subsequent performance correction removes repeated dense-mask work from
the nonlinear rescue's internal smooth constraint/Jacobian callbacks, which
never used that discrete mask. The final optimizer proposal still receives a
full exact/DTCBF/dense validation before acceptance, even when the optimizer
reports failure. A regression deliberately proposes an endpoint-feasible
inter-sample collision and requires rejection. No safety threshold, objective,
iteration budget, evidence clock or occupancy-loss rule changes.

Software validation after those corrections: 2,072 unit tests passed; repository
Ruff passed. The static trial recorded 204 live fields, 203 while translating.
These results do not supersede the failed late-crossing gate.

The exact repeat `native-crossing-v2` at e534647 also failed: robot termination
at 17.5333 s on unresolved fragment 357 (original source 18.0 s, missing 7.6 s,
distance 1.19995 m). It retained 162 live fields/156 while translating, .173236 m
minimum signed monitor clearance and .004167 m stopping drift. Nineteen
side-selection rows produced only .081634 m left excursion; human validation
still had zero supported moving-kinetic rows. Planner/perception calls still
exceeded 10 wall seconds; removing redundant rescue masks did not resolve
native latency. The strict report now saved successfully, including one
explicitly unavailable failed-plan margin. The subsequent expected wheel-gate
exception remains a failure even though the batch launcher returned zero.

The next bounded continuity step investigates fresh positive returns covering
a former fragment despite a farther cluster centroid, and early rejection of
already-colliding recovery candidates. It must preserve unobserved occupancy,
factor expiry, primary lambda and exact acceptance; no new guidance cost.
