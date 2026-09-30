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
zero consequence evidence. Derived fall-height, contact, and mass estimates
retain their oldest contributing evidence timestamp so feature extraction
cannot make stale evidence appear fresh. Non-finite energy arithmetic fails to
unknown.

Confidence is the minimum contributing confidence, capped at 0.5 for priors
and 0 for fallbacks by default. The same caps apply to energy-based potential/
kinetic inputs and normalized contact/thermal specialist inputs.
`allow_priors=False` rejects priors. Semantic detector confidence does not
replace physical confidence. Derived energies
are labeled inferred when their inputs are sensor evidence, while any prior,
fallback, or oracle dependency is retained. Diagnostics retain every original
input status, source, interval, confidence, timestamp, and evidence statement. Input status
labels are producer assertions, not independently verified observations.

Contact requires sharpness, hardness, dangerous-part exposure, and orientation
toward the route. Thermal requires a heat-source likelihood plus a separate
route-interaction likelihood. A factor is adapted only when every required
specialist input is available and fresh. Otherwise the existing CRA assessor
applies conservative uncertainty handling; its default fallback risk of 1 can
saturate overall `phi`. This must not be hidden by declaring unsupported
hazards absent or silently lowering fallback risks. Inspect each factor's
diagnostics during incremental validation.

`SafeSAGEPerceptionRuntime.initialize_from_tracking_with_cra_resolver` and its
update counterpart let a caller construct current-frame profiles after the
runtime's cloned candidate tracker and semantic resolver have completed. The
profile resolver receives those exact tracking and label results. Tracking,
labels, physical CRA profiles, and the Safe-SAGE field commit atomically; a
resolver failure preserves the previously accepted tracker and field. The
resolver does not relax provenance or freshness requirements.

Run `python scripts/demos/physical_factor_adapter_demo.py` from the repository
root. This stationary diagnostic uses synthetic contract fixtures only, not
live sensing, a trained model, simulator truth, or wheel motion. It exercises
prior mass, missing fall height, missing interaction, stale inputs, and fully
unavailable evidence. Live sensor accuracy, calibrated factor mappings, and
closed-loop restaurant validation remain future work.
