# Human-body contact prior

The user approved treating a freshly observed human body separately from a
generic object outline for contact. The previous carried-server run's three
group phis of 1.0 remain valid historical results: its whole-body silhouette
proxy saturated. Elongation, clothing contours and articulated outlines are
not measurements of physical sharpness. This new policy is an explicit model
change, not a revised interpretation of that earlier experiment.

## Score and meaning

The recommended default is **0.10**, a configurable low, nonzero normalized
blunt-body contact prior. It is an engineering assumption for the selected
adult restaurant demonstrations, not a trained estimate, injury probability,
measured material property or safety certificate. Zero would assert an absence
of contact hazard; the generic missing-evidence prior remains .25. Choosing .10
keeps an explicit residual body-contact term below that unspecified-object prior.
The exact value is uncalibrated and should be included in later sensitivity
analysis, for example .05/.10/.20. No measured human hardness or sharpness is
invented and no physical-attribute field is filled with zero.

Mass and relative speed continue to determine kinetic evidence. A sufficiently
large kinetic factor can still saturate phi despite the low body-contact prior.
All human, prop and unmatched LiDAR occupancy remains in the field. Contact
score is not collision permission; nominal control, MPC, analytical CBF, geometry,
source clocks and fixed lambda 1.25 remain unchanged.

## Selection and priority

`robot/perception/human_body_contact.py` reuses the existing fresh person/SAM
ownership gate from the human-mass adapter. Only the gate is reused; none of
its mass values enters the contact score. Same RGB frame/time/resolution,
accepted same-track SAM ownership, an untruncated resolved mask, adequate
confidence, unique person overlap and the existing freshness checks are required.
An authored role, asset name, class string alone, stale identity or retained mass
cannot select this prior. The policy also requires the original current camera
epoch and track. A missed person assessment reverts to the unchanged generic
evidence/prior policy instead of renewing the human prior.

`robot/risk/object_risk_policy.py` applies the following order:

1. Preserve accepted physical contact or a directly supplied specialist model.
2. For a fresh person whose remaining model is a whole-body silhouette or missing
   contact evidence, select the declared .10 blunt-body policy prior.
3. Assess any separately accepted exposed-edge cue with its original .25
   uncertainty policy. Keep it if its conservative contact score exceeds the
   body prior; missing part interaction is not interpreted as safe.
4. Keep the carried object's independent contact estimate and all other factors.
   Confirmed groups still use the maximum of the individual aggregate phis.

The body prior is represented with **zero measurement confidence** and an
explicit `prior` source/status. Only that person's contact fallback becomes .10,
and an unknown contact estimate resolves to that constant through the existing
assessor. Confidence is not increased to make a desired score. Aggregation,
other priors and the uncertainty rule for accepted part cues are unchanged.
The raw outline/missing-contact diagnostic, original contact score and original
aggregate phi remain in every affected record. A bare body prior cannot suppress
a stronger accepted part cue. A knife or other object does not inherit a body
prior merely by being carried or near a person.

This intentionally does not solve hidden sharp objects, unusual protective
equipment, arbitrary clothing/accessories or crowded/ambiguous people. These
are outside the selected adult-body prior scope. A visual edge proxy remains
different from physical sharpness or hardness.

## Runtime opt-in and checks

Core object-map callers pass `HumanBodyContactPriorConfig(score=.10)` as
`human_body_contact_config`. Legacy callers omit it and retain their original
outline policy. The zero-wheel native sensor diagnostic exposes an explicit
`--human-body-contact-prior .10` flag. It validates the value before launching
Isaac; the legacy experiment is reproduced by omitting that flag.

```powershell
$env:PYTHONEXE='"C:/Users/tayli/anaconda3/envs/env_60_isaacsim_yolo/python.exe"'
& C:/isaacsim/python.bat -u scripts/demos/restaurant_human_locomotion_isaac.py `
  --scene scenes/generated/restaurant_service_isaac_server_v2.usda `
  --output outputs/restaurant_service/NEW-UNUSED-DIRECTORY `
  --layout-survey outputs/restaurant_service/layout-monitor-v1/geometry-survey.json `
  --sensor-diagnostic --carry-bowl --carried-object-groups `
  --continuous-crossing --continuous-upper-continuation --carry-facing-crossing `
  --speed .5 --human-cluster-radius .18 --robot-view carried-diagonal `
  --camera-upward-degrees 15 --sensor-simulation-rate .05 `
  --human-body-contact-prior .10
```

`human-body-contact-validation.json` independently audits fresh positive human
ownership, .10 policy values with zero measurement confidence, absence of
nonhuman assignments, unchanged noisy-OR arithmetic and preserved raw diagnostics.
The normal native actor, attachment, human-sensor and carried-group gates remain
separate and mandatory for a full component pass. Inspect saved flags and error
logs because the Isaac launcher can return zero after an exception. No new
robot-wheel experiment is part of this branch.

Unit coverage includes unsupported/stale/ambiguous person input, prior provenance,
configuration, unchanged object scores/occupancy/other factors, raw saturation
retention, independent hazardous-part precedence, kinetic saturation, group max
and report-audit failures. Executed sensor/regression results are recorded below.

## Executed results

`outputs/restaurant_service/human-body-contact-v1/`, functional source `87d059c`,
passed the native actor, attachment, human-sensor, carried-group and new body
contact gates with the command above. Robot wheel commands were always zero and
scene source hashes stayed unchanged. The run used paced simulation, declared
oracle ego/extrinsic references and the existing RGB read-time clock approximation;
it does not establish real-time control or independent camera acquisition timing.

- 167 fields; 108 fresh body-prior rows, including 92 during actor movement.
- All selected body contact scores were .10 with explicit prior provenance and
  zero measurement confidence. No nonhuman body-prior assignment or arithmetic
  mismatch was found. Fifteen of these frames retained an original raw contact
  score of 1.0 in their diagnostics.
- 26 separate person/bowl assessments and three confirmed groups at source times
  7.5/7.6/7.7 s. Maximum bowl-centroid error was .159418 m; group completions were
  .166667–.183333 simulation seconds old, below the unchanged .35 s gate.
- At 7.6 s, raw body contact/phi were 1.0/1.0. The new body contact was .10,
  person phi .608678, bowl phi .821154 and group phi .821154. The other two group
  phis were .822696 and .822236. These are same-frame policy results, not matched
  before/after robot trajectories. Bowl mass and scores were not supplied or
  overridden. The representative figure is `human-body-contact-preview.png`.

An explicitly offline single-frame sensitivity check (`contact-prior-sensitivity.json`)
at body priors .05/.10/.20 gave person phis .586938/.608678/.652158. Group phi
remained .821154 because the independently assessed bowl dominated. This small
sanity check does not calibrate the default, establish ordering across arbitrary
people or demonstrate a navigation benefit.

The complete unit regression passed **2,019 tests in 85.12 s**
(`outputs/human_body_contact/full-unit-v1.log`); repository Ruff and whitespace
checks passed. No packages, models, weights or data were installed or changed.
The previous contact model and its saturated results remain reproducible.

Carrying association is still brief and view dependent, and potential/thermal
inputs remain priors where unavailable. This branch adds no grouped wheel
acceptance and does not clear PR #117's main hold. The next experiment discussion
should keep the chosen body-contact model fixed across matched controller
baselines, with any contact-prior sensitivity and risk-dependent lambda reported
as separate ablations.
