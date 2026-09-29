# CRA restaurant evaluation boundary

The primary comparison uses matched scenario, seed, and evidence-mode cells for
geometric velocity CBF, fixed-class Safe-SAGE, and continuous CRA Safe-SAGE.
The two Safe-SAGE baselines keep lambda fixed at 1.25. Risk-dependent lambda is
a separate ablation.

Sensor-derived wheel-motion trials require factor-specific readiness records
for potential, kinetic, contact, and thermal evidence. Each factor must have a
validated stationary pipeline, audited runtime provenance, and sensor-domain
accuracy evidence. Missing readiness or an unsupported factor blocks motion;
it is never converted to zero risk.

The current primary contact scenario is an exposed blade edge because the
local PACO baseline has nonzero held-out performance only for exposed edges.
Pointed tips, protrusions, jagged surfaces, physical sharpness, and hardness
remain pending. The thermal scenario is carried hot soup with a visible-plume
cue above an open container. It does not claim that a visual plume is measured
steam or temperature.

Before any motion comparison, stationary outputs are recorded twice: a
factor-specific diagnostic for the scenario's named hazard family and a
separately labeled four-factor aggregate. The factor-specific result cannot be
presented as aggregate `phi`. Unknown factors remain unknown; if their
conservative fallbacks saturate aggregate `phi`, the saturation report names
them and preserves their rejection reasons.

The present repository is not ready for sensor-derived wheel-motion trials.
Visual mass lacks Isaac-domain accuracy calibration, the contact model has
limited geometry coverage, and no trained hot-soup checkpoint has been
validated. Oracle and prior trials may test controller plumbing, but their
results must be reported separately from sensor-derived experiments.
The result aggregator enforces that separation by emitting distinct baseline
summaries for each evidence mode.
