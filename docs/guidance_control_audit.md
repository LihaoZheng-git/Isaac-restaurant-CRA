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
