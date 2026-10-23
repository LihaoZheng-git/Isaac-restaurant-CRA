# Rear-pass attempt and selected stop-and-go fallback

Follow-up: the user subsequently authorized further diagnosis. See
`rear_pass_feasibility.md` for a conditional virtual result under a separately
declared fixture and the pending waypoint-semantics decision. All original
failed results below remain unchanged.

The user authorized step4 as a bounded attempt at a curved pass behind a crossing
pedestrian, with the validated stop-and-go strategy retained if it did not work.
The outcome is **rear-pass acceptance failed; retain stop-and-go**. No new Isaac
wheel trial was launched, and no experimental controller was integrated into
runtime. The existing native acceptance remains the step1–3 result documented in
`encounter_guidance_switching.md`.

## What was tested

Five experimental families ran both transverse directions and the crossing-then-
parallel case, for **15 ORACLE virtual trials**. They reuse the existing moving-
disk benchmark, original robot waypoints, human11.5s exogenous onset/.5m/s motion,
N6/.2s prediction, QP1000/rescue20, exact temporal DTCBF/dense geometry and 60Hz
analytical checks. Authored identity, velocity, geometry and risk enter this
virtual fixture; none of these trials is sensor-derived perception or physical
wheel/braking validation. No packages, physical factors, body priors, source-age
limits or safety thresholds were changed.

All five families use applied field lambda1.25, the corrected nominal robot
route-segment context and current crossing eligibility. They are separate from
the accepted crossing-only lambda0 wait-and-go variant. The trials compare
complete experimental reference/control variants, not a causal lambda effect.

- **Existing tangent:** the current selected-obstacle tangent and the original
  rear-pass-or-yield preference, with weight100 and the explicit .25 minimum
  reference-speed fraction.
- **Motion-trailing reference, .4/.7m:** a virtual local reference goal offset
  opposite the current transverse track velocity. The configured waypoint
  nominal is unchanged. This is a candidate for the separate tracking objective;
  inherited rollout projection and full exact checks remain mandatory.
- **Direct checked reference, .7/1.0m:** a diagnostic selection rule may choose
  the complete reference itself after repeating full exact checks, requiring
  positive trailing displacement and verifying that its first command passes
  the analytical filter unchanged. Every subsequent virtual physics command is
  still filtered. This changes control selection rather than merely tuning the
  original MPC objective; it is explicitly **unintegrated** and is not a primary
  Safe-SAGE comparison or a native controller feature.

The reference offsets are configurable engineering hypotheses in the local
experiment, not measured human geometry, a larger safety envelope or a change
to CRA phi. Each probe's source and full trace remain under ignored outputs.

## Independent acceptance and results

The new evaluation-only `summarize_oracle_rear_pass` preserves the established
criteria: original route arrival/safety, positive independent clearance,
nonnegative analytical margins, at least.20m trailing excursion **during the
nearby transverse walking window**, and crossing the original walking plane
behind the actor by at least.10m. It correctly aligns recorded robot interval-
endpoint states with the authored actor at that endpoint. It rejects missing
ticks, nonfinite data, wrong fixture scope and native/sensor records. Actor
phase/route is scoring truth only. The evaluator never supplies a field or
command and does not modify the original trial's arrival result.

| Experimental family | Route arrivals | Maximum absolute lateral deviation, any time | Maximum trailing arc during crossing | Curved rear-pass acceptance |
| --- | --- | --- | --- | --- |
| Existing tangent | 3/3 | .0410m | .0167m | 0/3 |
| Motion-trailing .4m | 3/3 | .0852m | .0211m | 0/3 |
| Motion-trailing .7m | 3/3 | .1079m | .0183m | 0/3 |
| Direct checked .7m | 3/3 | .1532m | .0173m | 0/3 |
| Direct checked1.0m | 3/3 | .1975m | .0129m | 0/3 |

All15 original route checks and analytical/clearance checks passed, with minimum
monitored virtual clearance.398415m. Crossing-plane behind checks also passed.
**All15 curved-strategy checks failed** on the walking-window arc. Larger bends
occurred later, after the perpendicular encounter cleared. An earlier progress
update's 8–11cm figures described any-time deviation, not an accepted curved
pass; the independent timing audit is the authoritative strategy result.

This bounded attempt does not establish global infeasibility or a single root
cause for every earlier native stall. It shows that these current short-horizon
references did not produce the desired timely curved motion. Increasing offsets
or selecting complete checked references did not solve that acceptance problem.
The.20m criterion was not lowered, late motion was not moved into the walking
window, and positive clearance/arrival was not relabeled as rear-pass success.

## Selected practical behavior

Keep the already validated **encounter-dependent stop-and-go configuration**:
only a current qualified perpendicular human may get applied field lambda0 and
an exact/analytical checked wait. Parallel/departing/stationary/uncertain humans
and unrelated occupancy keep lambda1.25. Original source clocks and all geometry,
MPC, CBF, track-loss, arrival and stopping checks remain. The saved native
configuration is `encounter-native-encounter-v3/invocation.json`; this attempt
does not silently change the default fixed-lambda primary comparison.

There is no new native video because these candidates did not pass the virtual
gate for enabling new wheels. Human mass/contact and unavailable potential/
thermal in the accepted native fallback retain their explicit priors; it had
zero complete fresh four-factor assessments. Keep main/long-service/PR117 holds.
A broader time-aware path-reference approach remains future research, with
separate scope and validation rather than a claimed completed step4.

Local artifacts:

- `outputs/paper_flux/rear-pass-existing-segment-v1/`
- `outputs/paper_flux/rear-pass-motion-offset-{0.4,0.7}-v1/`
- `outputs/paper_flux/rear-pass-motion-direct-0.7-v1/`
- `outputs/paper_flux/rear-pass-motion-direct-1.0-v2/`
- `outputs/paper_flux/probe_rear_pass_{existing,motion,motion_direct}_v1.py` and
  `probe_rear_pass_motion_direct_v2.py`: unintegrated zero-wheel probe sources.
- `outputs/paper_flux/audit_rear_pass_attempts_v1.py` and
  `rear-pass-attempts-v1-summary.json`: independent assessments, original result
  SHA256s and failed checks. Original trial files remain unchanged.
- `outputs/paper_flux/rear-pass-development-v1.png`: independently checked
  trajectory/timing plot; it is a virtual result, not a new Isaac recording.

Only the independent evaluation helper, its regression tests and project
guidance are integrated by this branch. Experimental references/selection rules
stay outside runtime.

Validation:102 focused control/evaluation tests and all2,537 unit tests passed
(full suite85.52s); Ruff over robot/tests/scripts, compilation and diff checks
passed. The full suite used a fresh repository-local temporary directory. No
environment or dependency changes were made.
