# Closer-crate straight-route test

The user requested a simple test with the crate nearer the unchanged straight
waypoint direction, retaining the confirmed better-evidence update rule. This
separate default-off `--crate-route-probe` uses the same robot start
(15.65, 4.05), yaw zero, and a 4 m straight nominal goal. The prior 40 cm approach
stopped before passing the crate; it cannot answer the new bypass question.

The original licensed crate moves 45 cm toward the route, from world center
(17.5, 3.0) to (17.5, 3.45). The bin is inactive in this crate-only layer.
The original mesh/scale, camera, mass/contact estimators, confirmed refresh,
live kinetic/interaction, priors, lambda, MPC/analytical CBF, physical limits,
clearance, arrival and stopping checks remain unchanged. No authored detour,
supplied phi or moving-human guidance preference is introduced. Original V3
and short-test outputs remain untouched.

Require a fresh source-matched independent60 stationary gate, then own60,
before the bounded motion trial. Report actual lateral deviation from the
straight nominal, passage beyond the crate, arrival/stopping/clearance, fresh
and retained evidence, and accepted/rejected refresh attempts. A successful
curve demonstrates this controller/scene combination, not isolated CRA benefit
over fixed-risk control. Preserve failures and default-off/draft/main holds.

Native validation pending. Outputs: `outputs/crate-route-probe/`.
