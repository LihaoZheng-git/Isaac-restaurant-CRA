# Crossing maneuver development

The user selected two strategies: a checked curved pass behind the pedestrian,
or waiting for the crossing to clear before continuing the waypoint route.
The separate `--crossing-strategy` ablation leaves the obstacle-unaware nominal,
physical CRA policies, current positive occupancy and original source clocks intact.
Default `legacy` preserves the existing controller.

`rear-pass-or-yield` uses the existing selected-observed-obstacle reference and
its original eight-second policy lifetime. When no complete checked reference
exists, or the optimized first turn opposes that reference, it tries a zero
horizon. This horizon must pass the original temporal DTCBF, dense configuration
geometry and analytical zero-command checks. An unsafe wait retains the original
safety response. This is a bounded preference, not enforced side or convergence.
The explicit reduced-speed reference pool is available independently.

`yield-then-go` tries that same checked wait while a current crossing selection
exists. No selection, failed identity, or expired original policy delegates to
the normal safety controller. Selection is not renewed to prolong a maneuver.
The optional `--zero-tangential-bias-ablation` requires this strategy and zero
extra guidance-tracking weight. It sets applied **field** lambda to zero for
track and fallback policies while keeping normal repulsion, occupancy, physical
CRA scores and their original lambda 1.25. It is separately reported from the
fixed-lambda primary comparison; no safety constraint is disabled.

Evaluation preserves the original closest-encounter report. A new strategy report
measures trailing position at the robot's first crossing of the pedestrian's
walking plane. This avoids calling a correct wait a failed pass merely because
closest approach occurred before walking finished. Actor truth is scoring only.
The rear-pass strategy still requires a 0.20 m trailing arc during walking.
Wait-and-go makes no curved-bypass claim and does not require simultaneous robot
translation, but retains supported observed human velocity, actual arrival,
clearance, braking, current fields, identity and body-prior containment gates.
Old failed experiments remain failed under their original acceptance.

Initial validation: 36 focused tests and all 2,447 unit tests passed; Ruff passed.
The trusted original 19.7 s input reproduces the old right-turn fallback, replaces
it with a fully checked wait when no useful reference exists, and preserves the
quarter-speed checked left reference when available. Exact/analytical checks pass
with original epochs. This is numerical replay, not arrival acceptance. Each
candidate's own stationary/native gates remain mandatory. No main or long-service
acceptance follows.
