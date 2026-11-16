# Checked waiting-field transitions

The user approved diagnosing the late-crossing analytical failure and testing a
bounded correction, with fixed-lambda primary and CRA estimation unchanged.
This branch adds an **opt-in, ORACLE evaluation integration** of a reusable
stopped-horizon field check. It is not native wheel/perception acceptance.

## Failure reproduced

The original guide-mode right-mirror case at onset11.5 s still reproduces the
saved `analytical_emergency` failure with .465790 m minimum independent
clearance. Navigation time and field epoch differ by1 s in this ORACLE fixture;
capture timing selects diagnostics only and never supplies controller inputs.

The trace shows that an intentional wait was **not accepted** before the
failure. When the selected perpendicular human switched to applied lambda0,
the new safety field and its original finite-difference temporal term made the
complete zero-command horizon fail. Legacy code correctly rejected that wait
plan but had already committed the new field, so the remaining analytical
response could become infeasible in the forward-only actuator box.

At source13.0 / navigation12.0 s, the sampled h was .537979, scaled temporal
term -.541902 and stopped constraint right-hand side +.003923. The zero
horizon's minimum exact DTCBF margin was -.109091. At source13.4 / nav12.4 s,
the failed execution state had mapped coefficients approximately
(-1.495584, .062387). The original bounds v in[0,.25] and omega in[-.8,.8]
provide at most .049910 left-hand side, below the required .057783. No command
inside that box could satisfy that constraint. Positive geometric clearance
does not make this an accepted analytical result.

The h/temporal change includes geometry motion and the field-policy switch;
the trace alone does not isolate their contributions. The new guard records
candidate and selected constraints at the same state/geometry/source pair,
making the policy alternative explicit rather than zeroing temporal evidence.
At the first rejected transition, source13.0, this same-input comparison is:

| Current field candidate | h | Scaled temporal term | Stopped constraint RHS |
| --- | --- | --- | --- |
| Proposed lambda0 wait field |.537979 |-.541902 |+.003923 |
| Rebuilt lambda1.25 normal field |.691048 |-.186068 |-.504979 |

Both use the same original previous accepted source12.8. The alternative
changes the guidance policy, not the actor timing, geometry, derivative clocks
or physical risk. It is not an excuse to omit the original transition term.

## Bounded correction

`robot/safety/checked_wait_transition.py` assesses a proposed wait field before
publication. It requires a unique current crossing, the original bounded
selection clock, a matching current tracker epoch and sufficient consistent
positive track support. It tests a complete stopped horizon using:

- unchanged exact temporal DTCBF and dense geometry checks;
- the forecast analytical inequalities at every horizon interval;
- an unchanged, non-emergency first stopped command from the analytical filter.

The helper returns a decision/audit only. It cannot publish a field or command.
The ORACLE caller commits the zero-bias wait candidate only when all checks pass.
If it fails, the caller rebuilds a **current** normal-guidance field against
the same previous **accepted** field and temporal estimate. The rejected
candidate never becomes the derivative baseline. Raw occupancy, physical risk,
normal flux, track/selection timestamps and source frame stay unchanged.
There is no derivative reset, missing-input substitution or clock renewal.

The normal alternative is not presumed safe: original MPC, full exact/dense
acceptance and per-physics-step analytical filtering still determine execution.
If it fails, the trial remains failed. A rejected wait is neither an accepted
intentional yield nor successful braking. The zero-command virtual model does
not validate physical deceleration or indefinite future waiting.
Each rejected candidate costs at most one additional current field solve in
this bounded integration; these virtual timings do not establish live latency.

This is a separate control-policy ablation. Applied lambda may be1.25 during a
perpendicular encounter when lambda0 waiting is unavailable; that behavior is
explicitly reported. It does not silently replace the earlier strict
crossing-only-zero variant or alter the fixed-lambda primary.

The existing motion-trailing reference supplies early curves when its current
motion/lead/turn gates qualify. Its original1.2 m outgoing release prevents
holding a maneuver indefinitely after the human clears. The correction adds no
new retained identity, actor-phase input, physical evidence or obstacle-aware
nominal path. Guide mode and final-goal acceptance remain separately labeled.

## Controlled validation

The first isolated guard probe reaches the previously failing right-mirror
goal in26.6 s with .330843 m minimum clearance and analytical acceptance.
It deliberately retains the old wait policy's late-release inefficiency; the
hybrid comparison separately includes the existing outgoing-release logic.
This probe is not a curved-pass result or proof across encounter timings.

`scripts/evaluation/compare_checked_crossing.py` compares the primary fixed
field, original uncorrected wait variant and guarded hybrid. All use the same
original anchors2.2/3.1/3.9, guide-mode nominal, geometry/risk/flux, actuator
bounds, N6/.2 s, QP1000/rescue20, temporal smoothing and60 Hz analytical checks.
Cases vary exogenous actor onset8/10/11.5 s, both transverse directions and
crossing-to-parallel departure; the11.5 s set also includes both parallel
directions. These timings are fixture/evaluation inputs only. Track velocity,
identity, geometry and risk are ORACLE values, not camera/LiDAR estimates.

All33 matched virtual trials completed:

| Controller variant | Navigation/clearance/analytical passes | Original late right-mirror result |
| --- | --- | --- |
| Fixed-lambda primary |11/11 |Arrival22.6 s |
| Original uncorrected wait |10/11 |Retained analytical failure |
| Guarded hybrid |11/11 |Arrival23.2 s |

The guarded hybrid's three onset8 s cases retain the exact saved early curves
(.302250/.302931/.302250 m during walking; arrivals19.8 s). All three onset10 s
cases arrive in22.0/22.0/21.6 s; the onset11.5 s transverse/mixed cases arrive
in23.4/23.2/23.0 s. The parallel cases arrive in19.4/20.0 s, retain lambda1.25
and have no intentional yields. Guarded minimum independent clearance across
all11 cases is .330843 m. The fixed primary also passes11/11; no superiority,
speed or causal CRA advantage is established by these single-run fixtures.

The late right mirror rejects13 proposed wait fields and accepts seven checked
yield fields. Its accepted virtual zero-command window is nav14.6–16.0 s;
normal guidance resumes at16.0 s under the existing outgoing-release gate.
Unavailable waiting earlier is recorded as normal-guidance safety response,
not an accepted intentional yield. Primary arrival is .6 s faster here.

The separate post-hoc audit passed every source-hash/record check. All15
available unchanged historical comparisons preserve exact state arrays and
acceptance flags, including the old analytical failure. Accepted virtual yield
plans were also checked against actual zero commands/unchanged interval states.
These record audits are not physical braking or native command-lease audits.

Arrival/safety acceptance and visible curved-passing acceptance remain separate;
late checked waiting or normal-guidance response need not invent a curve.
The full model has no physical inertia, sensor occlusion or asynchronous delay.
No causal CRA advantage, universal safety, global convergence, exact paper
equivalence, physical braking or real-time claim follows from these cases.

Regression validation: **2,606 unit tests passed in139.46 s**; the focused
guard/controller group passed63 tests in6.86 s. Ruff over `robot`, `tests` and
`scripts`, Python compilation and diff checks passed. No dependencies or
environment settings changed, and no new Isaac process/wheel trial was run.

## Reproduction and retained artifacts

Use a fresh output directory:

```text
python scripts/evaluation/compare_checked_crossing.py --output outputs/checked-crossing --onset 11.5
```

Legacy reproduction remains available with `--variant uncorrected_wait`; the
new guard is disabled by default in the shared benchmark and old demos.
`compare_checked_rear_pass.py --checked-wait-transition` also explicitly opts
into the guard without changing the primary controller.

Local ignored artifacts under `outputs/paper_flux`:

- `late-crossing-wait-diagnosis-v1/`: exact original reproduction and full
  constraint audit, with trusted internally generated snapshot pickles only.
- `late-crossing-checked-field-first-v1/`: failed development invocation from
  a tracker timestamp attribute mismatch, before any guard outcome or wheels.
- `late-crossing-checked-field-first-v2/`: first bounded positive guard probe.
- `checked-crossing-{early8,middle10,late115}-v1/`: final source-hashed matrices.
- `checked-crossing-audit-v1.json`: complete post-hoc records and15 exact
  historical regressions; `audit_checked_crossing_v1.py` is its local recipe.
- `late-crossing-checked-wait-v1.png`: recorded late goal progress, analytical
  margin and applied-lambda comparison; local plotting recipe is retained.

All previous failures, native factor priors, zero-complete-fresh-four-factor
finding and main/long-service/PR117 holds remain. ORACLE success alone does not
establish native acceptance; run the integration's own stationary gate before
wheel trials and retain its full motion checks.
