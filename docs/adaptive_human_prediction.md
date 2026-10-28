# Bounded motion-disagreement uncertainty ablation

The user prioritized broader predictive planning before a PI demo. Early
preparation with a1.2 s fixed-growth forecast reached all guides, but the
measured nearby hold was .816667 s against the unchanged .5 s target. Its
failed native acceptance and the separate mass-continuity work stay recorded
in `predictive_parallel_redesign.md` and `human_mass_identity_retention.md`.

This separate default-off option changes only the future geometry growth
PRIOR. For each current verified human, let `v_raw` be the current LiDAR
tracker velocity and `v_fit` the existing .8 s source-window fit. The candidate
uses `growth = min(.50, .30 + norm(v_raw-v_fit))` m/s. The .30 floor and .50
ceiling are explicit engineering priors, not calibrated confidence limits or
a bound on arbitrary human acceleration. Transition disagreement increases
growth; stable motion retains a nonzero floor. Neither fitted velocity nor
the physical CRA kinetic estimator is changed.

`--human-prediction-uncertainty-floor .30` opts in; the existing explicit
`--human-prediction-uncertainty-rate .50` supplies the ceiling. Omitted floor
keeps fixed growth, including all legacy defaults. Source fit, body radius,
current occupancy, clocks, horizon bounds and every exact endpoint/dense/QP/
nonlinear rescue/analytical/physical/arrival/stopping check remain mandatory.
MPC's objective and robot limits are unchanged. Current source-derived rates
stay fixed during one query horizon; future actor truth never supplies them.
The forecast audit independently recomputes the rate from original velocities
and rejects missing or inconsistent adaptive diagnostic fields.

DEVELOPMENT replay covered all5,920 available queries across five recorded
runs with zero misses; minimum union-containment margin was .0238051 m. These
are reused design inputs on the same bounded actor scenario, not independent
calibration or native acceptance. Fixed .45 also covered these datasets but
was not selected as an explanation of transition uncertainty. Preserve
`outputs/restaurant_predictive_redesign/adaptive-growth-development-v1.json`
and `adaptive-growth-short-moving-development-v1.json` as offline hypotheses.

The implemented fixed/default mode reproduced both complete original native
command horizons exactly (maximum difference0) on two trusted saved inputs,
with original fields/epochs unchanged. See
`outputs/restaurant_adaptive_human_prediction/fixed-default-parity-v1.json`.
Source tests cover rate bounds, unchanged source/fit/body, state gradients,
hard QP and nonlinear rescue Jacobians, and diagnostic conformance. New native
forecast plus mass-provenance/component and independent+own60 gates must pass
before a combined wheel trial. Keep at least five supported moving-kinetic
rows, actual arrival, nearby curve and .5 s/50% hold targets. No primary/main/
long-service/PI/PR117 hold is released; native validation remains pending.

Source freeze passed3,097 full unit tests in104.13 s,53 focused tests in7.31 s,
Ruff, compilation, both CLI help paths and whitespace checks. Default fixed
command parity on the two original native inputs was exact (maximum error0).
