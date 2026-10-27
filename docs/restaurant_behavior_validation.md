# Short restaurant behavior validation

The accepted final behavior design is curved avoidance for parallel walking and
checked stop-and-go for perpendicular crossing. Rear passing remains experimental;
none of its failed reports or main/long-service/PR117 holds is changed here.
These short tests precede the separate PI demo design, not a long service route.

The crossing case repeats the accepted left-to-right continuing-away10s fixture,
with current checked-wait admission and strict three checkpoints. Only the
selected eligible crossing may receive applied lambda0, after full stopped-
horizon/source/current-pose admission; physical CRA stays lambda1.25. Rejected
waits rebuild normal fields and cannot count as intentional yields.

The separate oncoming parallel fixture places the free robot at(28.8,7.0),
heading north, with obstacle-unaware local checkpoints(1,0),(2,0),(3.5,0).
The native adult starts at(29.8,10.5), moves to(29.8,7.0) at.5m/s after an
explicit3s navigation delay, and receives normal field lambda1.25 throughout.
No guidance-tracking cost or rear-reference planning is enabled in the first
trial. Authored positions/speed/delay belong only to actor rendering/evaluation;
they never supply runtime factors, masks, motion, geometry or desired phi.
Original restaurant/robot assets remain unchanged. The endpoint is not a claim
that the human stops rigidly; all-phase physical clearance still applies.

The original-layout proposal survey gives1.405545m minimum robot-route clearance
and.302145m human-route clearance; it excludes added prop bounds and is NOT
acceptance. The native setup and continuous independent monitor include all
actual prop bounds and remain mandatory abort/evaluation information only.

Before motion, require a parked parallel-human/sensor component and independent
and own60-frame zero-wheel controller gates. Keep the existing.40+.25m human
envelope, adult mass70[45,100]kg and blunt-contact.1 priors; unsupported potential/
thermal remain labeled.25 priors. The wider.30m prior was a separate rear trial,
not a changed default. No domain/quality/identity/source/freshness or stopping
threshold is relaxed. Preserve the original exact DTCBF/dense geometry and
per-step analytical checks, nominal/MPC speed limits, N6/.2s execution interval,
QP1000/rescue20, .5s candidate limit and .025 simulation pacing. No real-time,
complete fresh four-factor or causal CRA-superiority claim follows.

Parallel acceptance requires actual articulated oncoming walking, independent
grounding/collider/layout checks, supported current human motion/kinetic inputs,
at least three distinct current observed oncoming source epochs, normal
lambda1.25 on all policies, >=.20m lateral excursion while the walking actor is
within2m, actual final goal rejoining, clearance, containment and stopping.
A late curve, restored lambda, positive clearance or proposed reference alone
cannot pass. Crossing retains the existing intentional-wait/execution/source
policy gates. Every failure remains a separate original report.

Source validation passed2,803 unit tests in100.73s and69 final focused tests
in5.41s, including explicit parallel/default actor dispatch and incompatible-
protocol rejection. Ruff, compilation, CLI help and whitespace checks passed.
Native acceptance is pending. Exact frozen invocations and actual
videos will be saved under `outputs/restaurant_behavior_validation/`; ignored
local records are not automatically bundled into a PR or shared externally.
