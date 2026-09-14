# Physical evidence to CRA

`robot.risk.adapt_physical_factors` converts a `PhysicalAttributes` record
to a `PhysicalRiskProfile` and immutable per-factor diagnostics. It does not
run perception, build a field, or command the robot. Existing control and
oracle providers are unchanged.

Potential consequence is `mass * gravity * fall_height`; kinetic consequence
is `0.5 * mass * relative_speed**2`. Nonnegative input bounds propagate by
interval multiplication. Severity is `min(1, energy / configured_scale)`.
Both scales default to 100 J, gravity to 9.81 m/s². These are configurable,
uncalibrated engineering hypotheses; energy is not itself validated danger.
The upper energy and interaction bounds are used by default; nominal values
can be selected with `use_upper_bounds=False`.

Potential requires a separate dimensionless `potential_interaction_likelihood`
estimate describing harmful-fall opportunity. The caller must justify support,
fall path, and accessibility evidence. The adapter does not automatically use
instability, camera exposure, route crossing, object height, or center height.
It uses only explicit `geometry.fall_height_m`. The current LiDAR geometry
adapter deliberately leaves fall height unavailable and requires a support/
stability estimator. Thus live LiDAR geometry alone cannot enable potential
risk estimation. Future fall-height evidence still needs source scrutiny; an
object envelope is not a measured CoM, support surface, or fall trajectory.

Kinetic interaction uses `motion.route_crossing_likelihood`, a heuristic rather
than a collision probability. The existing relative-speed estimate references
waypoint-aligned nominal robot velocity; it is not necessarily measured actual
robot velocity. This adapter does not correct that reference or infer speed
from world-frame velocity components.

Contributing inputs must be available and no older than the configured age
(default 1 s) at the supplied evaluation timestamp. Future, stale, missing, or
disabled prior inputs make that factor unknown, with rejection reasons. The
clock must share the evidence time base. Zero-valued measurements remain valid
zero consequence evidence. Non-finite energy arithmetic fails to unknown.

Confidence is the minimum contributing confidence, capped at 0.5 for priors
and 0 for fallbacks by default. `allow_priors=False` rejects priors. Semantic
detector confidence does not replace physical confidence. Derived energies
are labeled inferred, and diagnostics retain every original input status,
source, interval, confidence, timestamp, and evidence statement. Input status
labels are producer assertions, not independently verified observations.

Contact and thermal are explicitly unknown, even if their attribute groups
contain values. The existing CRA assessor subsequently applies its conservative
uncertainty handling. Its default fallback risk of 1 for unknown contact and
thermal saturates overall `phi` at 1. This is expected and must not be hidden
by declaring those hazards absent or silently lowering fallback risks. Inspect
potential/kinetic factor diagnostics for incremental validation.

Run `python scripts/demos/physical_factor_adapter_demo.py` from the repository
root. This stationary diagnostic uses synthetic contract fixtures only, not
live sensing, a trained model, simulator truth, or wheel motion. It exercises
prior mass, missing fall height, missing interaction, stale inputs, and fully
unavailable evidence. Calibration, live evidence assembly, sensor evaluation,
specialist contact/thermal adaptation, and closed-loop validation remain future
work.
