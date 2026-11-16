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

At50e5b1f the fresh component PASSED108 sources/909 future queries/zero misses,
minimum union margin .0485359 m,88 retained mass updates/27 original priors,
and all original component gates. Independent+own60 parked gates passed too.

The combined live run PASSED all native navigation/evidence gates:3/3 goals in
22.016667 simulation seconds, .662147 m nearby walking bend, .915047 m minimum
physical clearance, .002680 m stopping drift and .125814 minimum analytical
margin. It recorded219 fields/205 while translating,23 supported moving-kinetic
rows (19 simultaneous), and219 forecast sources/1,965 queries/zero misses
(minimum union margin .0234560 m). The combined provenance audit passed89
retained updates/62 original mass priors; preparation109 used sources/four
original selections and current body/normal-selection audits passed. Preserve
`motion-adaptive030-cap050-horizon12-v1/report.json`, SHA256
`4cb4f466a3b911cbd8e6c005352b4237a6bc254a250bb4c4e02f26c82afde578`.

The SEPARATE smoothing target still FAILED: .816667 s actual nearby hold,
.833333 s slow interval, despite65.73% reduction from the original2.383333 s
hold. This equals the previous shorter-horizon hold, so do not attribute a
further smoothness benefit to adaptive growth. Native success and smoothing
failure are distinct outcomes. All-track numeric completeness is not complete
fresh four-factor vision or calibrated mass/heat/sharpness.

Original event evidence shows several exact-restoration attempts at essentially
zero future margin, some failed searches, five field-invalidated plans and
four stale-field ticks in the pause window. The passing preference was present
during most of the pause. Earlier1.6 m lateral qualification churn temporarily
disabled the preference before it; these observations do not establish one
sole cause. Detailed failed revalidation inputs were not preserved by the
existing last-in-bucket snapshots. Keep `pause-events-companion-v1.json` and
`guidance-gap-development-v1.json` separate from the immutable native report.
An earlier avoidance-preference continuity variant may be evaluated while
retaining every hard current/future field constraint; do not simply remove
checks or reinterpret the unchanged .5 s target as passed.
