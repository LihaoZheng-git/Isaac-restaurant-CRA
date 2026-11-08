# CRA boundary-response experiment

The user approved testing stronger CRA-to-field boundary mapping while keeping
the calibrated normalized scores and colors. This is a separate, default-off
control-policy experiment, not a revision to the estimated physical evidence.
All earlier V3/V4/V5 recordings and main/primary/PR117 holds remain.

## Mapping and scope

The original mapping is `b=-(0.5+1.2*phi)`. The opt-in mapping is
`b=-(0.5+(B-0.5)*phi)`, where `B` is the configured unit-risk magnitude.
The baseline has `B=1.7`; initial diagnostic candidates are 2.3, 2.9 and 4.1.
The zero-risk endpoint stays 0.5. Normalized CRA phi, factor scores, priors,
evidence clocks, palette, lambda=1.25, geometry, nominal waypoints, MPC/CBF
objectives and wheel limits do not change.

Only the existing calibrated verified/retained-object admission path uses the
alternative mapping. Ordinary unassociated obstacles retain their old policy.
Previously stored geometry can retain its actual earlier applied policy; this
does not turn an unknown surface into a new physical assessment. Mixed-cell
ownership remains risk-first, so a higher-phi unknown surface may own overlap
against a lower-phi calibrated object even when the latter has stronger flux.
No ownership selection is changed to obtain a preferred trajectory.

`--restaurant-boundary-unit-flux` defaults to 1.7 and requires explicit
`--restaurant-calibration kinetic` (or equal). The alternative endpoint is
finite and at least 1.7 for this strengthening experiment. Native use is limited
to the isolated prop-route probes. Human fragment/part/furniture policy paths
still enforce the original shared mapping, and stronger mapping is not enabled
in the full V3 human demo. Pure object-policy tests do not validate that mission.

The paper defines semantic boundary modulation separately from normalized CRA
scores; 1.7 is an experimental setting, not a numerical ceiling imposed by our
PDE solver. See [Safe-SAGE](https://arxiv.org/html/2603.05497v1), section III-C.
This project retains its planar/unicycle adaptation; no exact equivalence or
universal reachability claim follows from changing the endpoint.

## Validation protocol

The archived field probe uses the accepted calibrated V3 sensor/geometry export
with unchanged point placements, IDs, risks and recorded robot pose. Each
variant receives the same declared straight probe command, not a claim of
replaying the actual native nominal or issued command.
It compares boundary/PDE and static analytical response. It is not exact MPC
warm-start replay or a prediction of a new closed-loop path. Old geometry-prior
policies remain archived rather than pretending to reconstruct counterfactual
memory history. Baseline field parity is checked where a planner array exists.

Separate native recipes are `restaurant_boundary_response_baseline.json`
(B=1.7) and `restaurant_boundary_response.json` (initial B=2.9). They derive
without other command changes from the existing matched-bin and 5 cm-setback
crate passes: same world start (15.65,4.05,0), prop center (17.5,3.40), straight
4 m route, N12 sensor MPC, guidance tracking off, 45 s bound and 0.025 pacing.
This controller is distinct from the full V3 N6/guidance100 variant.

Each setting requires fresh independent60 and its own60 parked validation
before motion. Compare calibrated baseline and candidate on the same scene;
the old uncalibrated runs are context only. Record waypoint arrival, actual
deviation away beside the object, delay/stops, padded-box clearance, legacy
circle metric, passive PhysX contacts, and qualified-object applied flux/coverage.
The legacy authored-world monitor remains evaluation-only. A larger bend without
arrival is not a successful improvement. No authored detour or risk override
will be added to force visible motion.

## Archived diagnostic result

Eight saved fields were reconstructed at B=1.7/2.3/2.9/4.1. The one available
saved planner field (7334) matches the reconstructed baseline bit-for-bit in h,
forcing, guidance and occupied geometry. All 24 bin/crate static analytical
probes still admit the declared straight command (0.20 m/s,0 rad/s). The scalar
field changes (maximum absolute h change up to 0.136); this is not a no-op or a
1.7 clipping effect. The analytical margins do not indicate more intervention.

A separate cold N12 primary-objective probe at crate frame3122 keeps the same
predicted straight plan at B=1.7 and2.9. A synthetic pose shifted30cm toward
the prop also produces identical recovered plans, first input(0.05,-0.4).
These probes use limits from the saved N6 controller but explicitly replace
the horizon with12, omit the guidance objective/history, and set the temporal
derivative tozero. They do not reproduce native warm-start MPC. Results are in
`outputs/boundary-response/probe-v2` and `probe-cold-v2`; both source hashes
remained unchanged. Earlier nullable-metadata/result-attribute helper failures
remain saved separately.

There is no offline evidence here that stronger b yields a larger curve.
The next bounded native comparison tests B=1.7 against2.9 in the closer-crate
scene where the original isolated controller actually bent. Native results
remain pending at this protocol snapshot; no full-demo promotion is implied.
