# Restaurant evaluation protocol

Primary comparisons use matched scenario, seed, geometry, nominal waypoint,
initial state, solver settings, and sensor conditions. The baselines are the
geometric velocity-aware CBF, fixed-class Safe-SAGE-style policy, and continuous
CRA Safe-SAGE policy. Primary Safe-SAGE comparisons require lambda 1.25.
Risk-dependent lambda is a separately labeled ablation.

Each trial records evidence mode (`sensor`, `prior`, or `oracle`), completion,
collision, fail-closed state, minimum clearance and CBF margin, interventions,
fallbacks, and elapsed time. Oracle/prior trials cannot be reported as sensor
performance. Repeated seeds and uncertainty/domain-shift evaluations remain
required before research conclusions.

This branch implements the protocol and validators, not completed restaurant
experiments. Actual sensor trials remain blocked on collected mass data and
verified contact/thermal model weights. Wheel motion should only begin after
stationary scenario-specific perception acceptance checks pass.
