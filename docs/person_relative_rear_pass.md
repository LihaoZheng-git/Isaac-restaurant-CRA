# Person-relative rear-pass development

This is a separate, disabled-by-default online maneuver experiment on top of
the accepted rear-reference development branch. The user approved implementing
a person-relative arc, and on 2026-10-25 approved adjusting only pedestrian
timing/route for its next fixture, keeping robot limits and all safety gates.
Main, long-service and PR117 holds remain unchanged.

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
