# Short crate placement with the full restaurant controller

The user requests a bounded placement test, with the same 1.2-second MPC
horizon and guidance objective used by the full demo. A slight detour is
sufficient; do not change controller, estimation or uncertainty policy to
obtain a larger bend. If a bug or recurring planning blocker appears, preserve
the result and report it for discussion instead of fixing it in this task.

Use the accepted V5 scene unchanged: crate centre (17.50,3.00), bin centre
(21.20,3.20). The crate is 10 cm farther from the y=4.05 route than the
interrupted V6 clearance placement. Accepted V5 arrived with this layout, but
its crate-local deviation was small; a new short-test bend is not guaranteed.

The separate recipe is
`configs/restaurant_service_crate_short_full_controller.json`. It copies
`restaurant_service_risk_calibration_witnesses.json`; only the long-parallel
command's maximum duration changes from 300 to 45 seconds and the test selector
`--service-crate-short-probe` is added. All other prerequisite commands and
scene paths remain identical. The selector changes the robot start to
(15.65,4.05,0) and gives a straight four-metre obstacle-unaware nominal goal.
It does not supply obstacle geometry, object identity or risk to control.

Keep the full N6/dt0.2/guidance100 settings, lambda1.25, B1.7 mapping, calibrated
factor scoring/palette, model cadence, property ownership, rolling field,
stationary furniture and unknown-obstacle policies, surface witnesses, robot
footprint, pose padding and physical limits. The old isolated N6 static recipe
uses different memory rules and is not used as a substitute. This short route
has a different observation history from the long mission; matching settings
does not make it full-route acceptance.

The remote human and its runtime processing remain present. The short route
does not reach the human trigger or validate a walking encounter. Preserve
raw human audits and mark those encounter results unexercised; do not relabel
them as passed. Keep the existing independent human component, near-view and
idle prerequisites, plus a fresh independent short stationary60 and motion's
own60. Robot arrival, stopping, field/controller checks and passive contact/
clearance measurements remain in force. No authored-world motion veto is added.

Run one driven trial at this placement after its prerequisites. Native output
root is `F:/CRA-experiments/service-crate-short-full-controller-20261108`;
audit/preview output is `outputs/service-crate-short-full-controller/`. Keep
sources frozen throughout the gates and post-run source/configuration checks.
Report actual alongside deviation, full chassis passage, final error, stopping
drift, pauses, contact coverage and retained contact/phi evidence. Preserve
every frame if producing a clip. A stalled or near-straight result is reported
as observed rather than repaired through score or controller changes.

The accepted V5 retained evidence already has the desired ordering: bin raw
contact cue0.139254/calibrated contact0.050713/median phi0.293853 (75 rows),
crate cue0.436105/contact0.254809/phi0.449108 (308 rows). These are historical
observations, not supplied scores for the new trial. The mass and motion/context
also differ, so the aggregate phi difference is not exclusively due to contact.
Potential/thermal priors, rough mass, intermittent ownership, simulated ego
localization and paced execution remain explicit.

Implementation validation passes: 273 focused controller, geometry, launcher,
service and harness regression tests in 27.45 seconds, plus repository Ruff and
diff checks. The agent's 47 harness/static-controller checks and syntax checks
also pass. Independent review confirms exact parsed-argument parity and that
the crate stationary gate cannot be overwritten by later acceptance checks.
Native validation was pending at the implementation snapshot; the completed
result is recorded below.
Preserve all old short clips, completed V5, both interrupted V6 attempts and
main/primary/PR117 holds. Do not begin another full-demo run in this bounded task.

## Completed native result (2026-11-08)

Frozen runtime revision `d261e53a7193309fa979b1026ec125838eede2fd` passed the
four fresh external prerequisites, motion's own60 and the short native run.
The saved MPC and guidance dictionaries exactly match accepted V5; source,
recipe, scene, perception and uncertainty-policy checks pass. The native output
is `F:/CRA-experiments/service-crate-short-full-controller-20261108/motion-v1`.

| Measurement | Recorded result |
| --- | --- |
| Four-metre goal | 1/1, reached in 24.20 simulation seconds |
| Final error / unchanged tolerance | 0.096956 m / 0.10 m |
| Braking drift | 0.002728 m |
| Minimum sensor-field CBF margin | 0.130150 |
| Independent evaluation clearance | 0.305773 m |
| Entire padded chassis beyond crate | First recorded at 18.00 navigation seconds |
| Crate PhysX reports | Observer available; zero recorded crate contacts |
| Committed fields / while translating | 172 / 166 |
| Retained crate evidence | 91 fields; not continuous retention |

The path is almost straight. Its maximum lateral offset is **2.3052 cm toward
the crate**, after the robot centre has passed the crate centre. While the robot
centre is alongside the crate box, its offset reaches 1.0289 cm toward the
crate. The maximum away-side offset over the complete path is only 0.3198 cm.
This is successful passage, **not a demonstrated away-side detour**. Brief zero
commands remain in the recorded command trace; this is not a pause-free run.
One initial camera/LiDAR epoch rejection is preserved. No runtime algorithm,
scores or acceptance thresholds were changed to improve the outcome.

The retained crate cue is 0.359947, median calibrated contact 0.187421,
median phi 0.396152 (range 0.391830--0.428496), and rough mass 2.363315 kg.
There are no retained bin rows in this short run, so it does not provide a new
paired comparison. The previously accepted V5 observations above support
lower bin contact and aggregate phi; their different mass and motion/context
prevent attributing the entire phi difference to contact alone.

The original human stayed held at its remote start, with zero measured drift,
minimum robot distance 21.85 m and no runtime human evidence. Its encounter
is explicitly **unexercised**, not passed. Different initial sensor history
also means this short run does not repair or replace either interrupted V6
full mission. Zero recorded PhysX contacts is limited to the available
observer coverage, not a universal collision guarantee.

Post-run source/property/current-point/calibrated-score audits pass with no
reported errors. Independent review agrees with the signed trajectory and
scope interpretation. The separate preview retains all 322 native frames at
10 fps, including pauses, with a status banner only; full decode and frame/
epoch checks pass. Key workspace artifacts are:

- `outputs/service-crate-short-full-controller/audits-v1/summary.json`
- `outputs/service-crate-short-full-controller/audits-v1/policy-contract.json`
- `outputs/service-crate-short-full-controller/audits-v1/trajectory.png`
- `outputs/service-crate-short-full-controller/preview-v1/crate-short-pass-share.mp4`
- `outputs/service-crate-short-full-controller/preview-v1/validation.json`

The bounded placement trial is closed. Keep this as a reliable-passage result
with minimal path change; do not label it the requested visible detour or
promote it to a new full demo. No further controller repair or placement sweep
was performed. Draft PR253 and all main/primary/PR117 holds remain.
