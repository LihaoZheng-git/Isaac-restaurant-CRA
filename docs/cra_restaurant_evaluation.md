# CRA restaurant evaluation boundary

The primary comparison uses matched scenario, seed, and evidence-mode cells for
geometric velocity CBF, fixed-class Safe-SAGE, and continuous CRA Safe-SAGE.
The two Safe-SAGE baselines keep lambda fixed at 1.25. Risk-dependent lambda is
a separate ablation.

Sensor-derived wheel-motion trials require factor-specific readiness records
for potential, kinetic, contact, and thermal evidence. Each factor must have a
validated stationary pipeline, audited runtime provenance, and sensor-domain
validation appropriate to the [accepted bounded thesis scope](restaurant_evaluation.md#accepted-thesis-scope).
Generalization outside the selected restaurant cases is future work. Missing
readiness or an unsupported factor in the active experiment blocks motion;
it is never converted to zero risk.

The current primary contact scenario is an exposed blade edge because the
local PACO baseline has nonzero held-out performance only for exposed edges.
The accepted primary contact direction is a visual exposed-edge geometry proxy;
its opt-in adapter and labeled readiness path are implemented. Pointed tips,
protrusions, jagged surfaces, physical sharpness and hardness are future work.
The thermal scenario is carried hot soup with a visible-plume
cue above an open container. It does not claim that a visual plume is measured
steam or temperature.

Before any motion comparison, stationary outputs are recorded twice: a
factor-specific diagnostic for the scenario's named hazard family and a
separately labeled four-factor aggregate. The factor-specific result cannot be
presented as aggregate `phi`. Unknown factors remain unknown; if their
conservative fallbacks saturate aggregate `phi`, the saturation report names
them and preserves their rejection reasons.

The present repository is not ready for sensor-derived wheel-motion trials.
The remaining bounded tasks are a relative-heaviness check, positive fall-height
evidence, a visual contact-proxy scoring path and a working plume-estimation
method. A validated image-based plume baseline may be used without requiring
a new neural network. Oracle and prior trials may test controller plumbing, but their
results must be reported separately from sensor-derived experiments.
The result aggregator enforces that separation by emitting distinct baseline
summaries for each evidence mode.

`robot.scene.restaurant_risk_scenarios` is the reusable source of truth for
the five primary scene definitions. Actor positions use the robot's initial
frame. Procedural positions identify geometric centers; the reused chair uses
an explicit rendered-bounds floor anchor. This prevents a floor elevation from
being interpreted as an object center. Every authored semantic role,
trajectory, support height, edge cue, and plume/container cue is oracle scoring
metadata only. The crossing-person capsule and procedural edge/plume objects
are visual and LiDAR test proxies; they do not establish real person detection,
physical sharpness, temperature, or sensor-domain factor accuracy. The
evaluation scenario list is generated from this catalog so scene IDs, factor
families, and descriptions cannot drift.

`robot.scene.restaurant_scenario_authoring` places these definitions only in
the anonymous USD session layer. It rotates robot-frame positions and configured
velocities into the world frame, but never enables actor or robot motion. Z
coordinates remain world heights; the robot chassis elevation is not added to
support, landing, or fall heights. The stationary Isaac diagnostic
`scripts/demos/restaurant_scenario_isaac_stationary_demo.py` checks composition,
rendered bounds, root-layer preservation, and robot drift while issuing only
zero wheel commands. Its coauthored fixtures validate placement only and are
not valid perception samples.

`robot.evaluation.restaurant_evidence_records` is the durable boundary for the
stationary benchmark. Each record identifies the scenario, seed, capture, code
revision, and optional SHA-256 fingerprints for configuration, weights,
datasets, scenes, or code artifacts. Runtime physical inputs retain measured,
inferred, prior, oracle, fallback, or unavailable provenance.

Factor-specific records contain only the scenario's named factor and omit the
aggregate. Four-factor records store aggregate `phi` separately and mark it
reportable only when all required evidence passes the matching evidence-mode
readiness check. Unknown fallback factors and saturation remain visible even
when the aggregate is unreportable. Authored scoring truth is stored in a
separate `oracle_scoring` section with `available_to_runtime=false`; undeclared
fields are rejected, so object height cannot be inserted as fall height.
