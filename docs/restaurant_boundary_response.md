# CRA boundary-response experiment

The user approved testing stronger CRA-to-field boundary mapping while keeping
the calibrated normalized scores and colors. This is a separate, default-off
control-policy experiment, not a revision to the estimated physical evidence.
All earlier V3/V4/V5 recordings and main/primary/PR117 holds remain.

## Mapping and scope

The original mapping is `b=-(0.5+1.2*phi)`. The opt-in mapping is
`b=-(0.5+(B-0.5)*phi)`, where `B` is the configured unit-risk magnitude.
The baseline has `B=1.7`; initial diagnostic candidates are 2.3, 2.9 and 4.1.
The zero-risk endpoint stays 0.5. The scoring function and palette are unchanged;
identical evidence produces identical normalized CRA phi and factor scores.
Priors, evidence clocks, lambda=1.25, geometry, nominal waypoints, MPC/CBF
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
A structural review explains part of this response: the fixed tangential
Dirichlet ring is unchanged when only normal boundary magnitude changes. At
saved frame 3122 the four interpolation corners around the robot are on that
ring, with guidance `(1.25,0)` at both endpoints. At frame 4628 the local free
component is also isolated from the changed normal boundary by fixed cells.
The Poisson field still changes: local h increases by about .015--.020 in these
views. With unchanged local guidance, a larger h does not demand a stronger
analytical correction. This is an archived field diagnostic, not proof that
every native command has the same cause, and not evidence of a 1.7 clamp.
See `outputs/boundary-response/dirichlet-shielding-review.json`.

## Native matched-scene result

The four-stage pair completed at frozen revision
`838c3ec8ff3185f310011f1613bbdcf63cfb264b`. Both independent60 parked checks,
both motion runs' own60 checks, and both native arrival/stopping reports passed.
Every launcher reports unchanged source, exit 0 and no traceback. Native files
are preserved under `F:/CRA-experiments/boundary-response-20261107/`.

| Recorded measure | Baseline B=1.7 | Stronger B=2.9 |
| --- | ---: | ---: |
| Waypoint arrival | 1/1 | 1/1 |
| Navigation simulation time | 29.3833 s | 26.9500 s |
| Maximum lateral offset beside crate | 8.6657 cm | 8.2217 cm |
| Final goal error (10 cm tolerance) | 9.6643 cm | 9.6017 cm |
| Minimum padded chassis / crate-box clearance | 3.7116 cm | 3.3499 cm |
| Braking drift | 2.6615 mm | 2.6024 mm |
| Minimum sensor-field CBF margin | .119365 | .127872 |
| Recorded crate PhysX contact points | 0 | 0 |
| Qualified retained crate fields | 78/282 | 81/264 |
| Qualified crate median phi | .519938 | .506020 |
| Qualified crate median boundary b | -1.123925 | -1.714449 |

The candidate changes applied strength but does **not** give a larger curve in
this pair. It finishes 2.4333 simulated seconds sooner, with a 4.4404 mm smaller
bend and 3.6171 mm smaller padded-box clearance. This is one live trial per
setting, with fresh sensor/model observations in each run, not a statistically
established speedup or a risk-only causal comparison. The scoring algorithm,
palette and control configuration are unchanged except for B; measured inputs
are not identical. Retained mass/contact cue are 3.133366 kg/.531206 in the
baseline and 3.478502 kg/.513915 in the candidate. Mass is uncalibrated.

Independent source/point and individual-to-applied boundary audits pass for
both runs. The stronger run has 401 explicitly admitted boundary-override rows;
ordinary policies retain the original map and every audited lambda remains 1.25.
Both full paths pass the padded chassis beyond the crate and finish inside the
existing goal tolerance. Passive PhysX reporting is installed without error,
records floor contacts as a positive control in every phase, and records no
nonfloor environment headers or crate contact points.

The historical enclosing-circle metric still **fails** in both runs: minimum
-4.0924 cm/-4.6778 cm, respectively. That old 2 cm benchmark is preserved as an
evaluation-only result; the new native passes do not erase it. Padded-box
clearance and contact observations are distinct evidence, not a redefinition
of that circle metric. Neither record establishes collision freedom between
samples or universal safety.

Continuous crate retention also **fails** its separate all-fields check.
The 78/81 qualified fields have zero source/point errors, and every selected
point is within the original 3 cm mesh-box allowance, but the crate record is
intermittent and is absent through much of the bend. Only 13 qualified rows in
each run meet the old 20 cm whole-center diagnostic. Potential and thermal remain
priors; unavailable speed still uses the existing kinetic fallback. Do not
attribute the entire native curve to continuously available fresh crate CRA.

Post-run outputs are `outputs/boundary-response/native-pair-audit-v1/`:
`summary.json`, per-run clearance/retention/boundary audits, and
`paired-crate-boundary.png`. The figure uses exact recorded times and poses
through arrival and braking; the lateral detail panel explicitly expands its
vertical scale. The crate outline is evaluation-only, transformed by the actual
anchor. Source hashes are retained in the audits.

## Decision and checks

Keep the stronger mapping opt-in and the existing full demo unchanged. Do not
promote B=2.9 or launch additional full-demo sweeps on the assumption that a
larger value guarantees a larger detour. The next bounded investigation should
trace the local field/constraint response on identical sensor inputs, including
the fixed tangential interface and calibrated-versus-ordinary cell ownership,
before changing how risk activates intervention. Changing guidance coupling,
lambda, clearance geometry or the MPC objective would be a separately reported
method change. A route that already satisfies the filter need not be modified.

Validation: 4210 unit tests passed in 247.45 s; the final 83 focused tests passed
in 5.84 s (overlap with that suite), Ruff over robot/scripts/tests and diff checks
passed. Initial focused failures exposed the original fixed-map guard and the
CLI alias destination; both were corrected before the frozen native runs.
The post-run audits are read-only. All 28 files in the current calibrated V5
presentation manifest remain byte-identical, including both videos. Earlier
demos and failures are preserved; draft PR248 and main/primary/PR117 holds stay.
