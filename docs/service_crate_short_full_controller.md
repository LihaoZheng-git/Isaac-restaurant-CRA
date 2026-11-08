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
Native validation is pending at this protocol snapshot.
Preserve all old short clips, completed V5, both interrupted V6 attempts and
main/primary/PR117 holds. Do not begin another full-demo run in this bounded task.
