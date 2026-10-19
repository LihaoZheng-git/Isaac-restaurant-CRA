# Bounded Laplace-guidance heading ablation

The user approved this separate controller variant after the primary native
8 s crossing arrived/passed behind the person but failed the 20 cm visible-arc
criterion, and the 13 s repeat stopped on unresolved occupancy. Those failures
remain in `docs/observed_human_guidance.md`. The primary objective is unchanged.

The ablation adds a nonnegative heading cost to the existing MPC objective:

`J = J_primary + beta * sum(a[k] * wrap(yaw[k+1] - reference[k])**2)`.

For one synchronous solve, the current field is sampled along the unchanged
obstacle-unaware nominal rollout at its look-ahead states. Each valid guidance
direction supplies a preferred heading, capped to +/-45 degrees around the
nominal rollout's terminal heading. This terminal heading approximates the
nominal direction of progress; it is not an obstacle-aware waypoint or a hard
constraint on actual heading. `a = clip(1 - h / activation_h, 0, 1)**2`.
Unavailable/out-of-domain/unsafe reference stages, negligible guidance and
nonpositive nominal forward speed omit only this auxiliary preference. Their
physical evidence/occupancy and exact safety checks are never removed or
interpreted as low risk. The development setting is beta=100, activation_h=1,
heading bias=45 degrees, fixed lambda=1.25.

Two bounded guidance-following rollout seeds (turn caps .4/.8 rad/s, clipped
to actual actuator bounds) can initialize the optimizer. They retain the nominal
forward-speed sequence and original stop/rotation commands. A seed is used only
after complete exact DTCBF/dense geometry/temporal validation and lower evaluated
cost than the current safe incumbent. No feasible incumbent means an unsafe
nominal has no admissible cost to beat. The same heading term appears in QP,
rescue, recovery ranking, gradients and exact objective comparisons. Cheap
recovery lower bounds may omit the added nonnegative state cost. The per-plan
reference/cost arrays are owned and immutable; no shared planning context or
source clock is renewed.

Zero weight delegates directly to the original controller, including its result
type. The primary gets no extra sampling, seed, cost or shortcut change. Runtime
flags are `--guidance-tracking-weight`, `--guidance-activation-h` and
`--guidance-max-heading-bias-degrees`; all defaults retain the primary. The
geometric CBF baseline rejects the MPC-only opt-in. Reports and per-plan
diagnostics name the variant; the primary comparison aggregator rejects enabled
ablations. A later matched ablation comparison must apply the same controller
configuration to fixed-policy Safe-SAGE and CRA Safe-SAGE before attributing an
effect to CRA.

Development is zero-wheel and uses an authored disk/phi, not perception. The
audit's `--track-nominal-states` matches the native path-cost mode; the earlier
audit without this path cost is a different controller configuration. With
path tracking, the unchanged primary stalled on both sides at 9.95 s. Initial
command-projection, fixed angular-rate and uncapped-heading candidates also
failed; all raw trials under `outputs/guidance_tracking_ablation` remain.
Those failed alternatives are not integrated modes.

The terminal-direction heading variant with validated seeds reached both goals
in the dirty-tree development fixture (31.0/34.45 s, .936351/-.963203 m
excursions). Its ORACLE geometry, .05 virtual integration step and lack of native
future-configuration padding remain explicit. Frozen-source and native results
must be recorded separately; development arrival is not a moving-person,
sensor-derived four-factor, real-time or universal-convergence acceptance.

Unit checks cover zero-weight identity, frame rotations, both steering signs,
unchanged nominal inputs, zero guidance/far field/zero-speed behavior, unsafe
initial stops, rejected seeds, bounded references, immutable cost data, finite-
difference gradients/QP agreement and exclusion from primary comparisons.
Native testing must retain the 60-frame stationary gate and all original mask,
provenance, continuity, field/candidate age, execution-lease and CBF limits.
This objective does not fix close-view human truncation or unresolved fragments.

The final implementation passed 2,127 unit tests and repository Ruff. The
development beta=10 seeded heading variant still stalled; beta=100 is the fixed
candidate for the next frozen fixture/native collection, not an automatically
retuned runtime value. Frozen results must use the committed source hash.
