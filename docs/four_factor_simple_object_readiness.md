# Four-factor simple-object readiness

The bounded sphere test now computes all four factor inputs from observed
geometry, motion and RGB cues plus explicitly admitted density/floor priors.
No factor score or CRA phi is supplied by the fixture. This is a **prior-assisted
demonstration**, not a complete sensor-only mass/support estimator.

| Factor/input | Current method | Important limit |
|---|---|---|
| Rough mass | Fit an associated LiDAR spherical surface; multiply inferred volume by an explicitly configured effective-density interval | Mass stays PRIOR; contents, material and solid/hollow construction are not observed. Existing learned mass coverage is limited, not a general-object model. |
| Kinetic | Confirmed centroid tracking; relative velocity against waypoint nominal robot motion; bounded kinetic energy and route overlap | A static object still has nonzero relative approach speed. Dynamic-object accuracy was not established by these static trials. |
| Potential | Prefer the existing observed LiDAR support/landing method. Optional ground mode fits a nearby floor plane and checks inferred sphere-bottom consistency, then applies an explicit local continuity PRIOR | Unseen support remains assumed. Fall height is support-to-landing drop, never object/center height. Tipping and hidden edges are outside this prior. |
| Contact | Continuous visible-silhouette geometry proxy, retaining stronger accepted specialist part cues | A rounded outline can give a low visible cue; hidden geometry, physical sharpness and hardness stay unknown. |
| Thermal | Three fresh stationary object-associated RGB frames; continuous plume strength, extent and persistence | Validated no-plume views can give low cue evidence. No temperature/coldness/steam claim; clean reference and bounded negative-validation domain required. |

The adapter retains exact input bounds, sources, timestamps and statuses. CRA's
existing conservative uncertainty uplift and aggregation remain unchanged.
Computed phi sets Safe-SAGE boundary flux; lambda remains **1.25**. Waypoint
nominal remains obstacle-unaware; online MPC and analytical CBF own intervention.

## New ground-support method

`robot/perception/ground_support.py` accepts calibrated Z-up LiDAR scene points,
an accepted fitted sphere, a separately timestamped point cloud, and an explicit
scoped continuity PRIOR. It excludes the object's fitted footprint from floor
candidates so object returns cannot establish their own supporting plane.

The lowest local band must contain enough returns with two-dimensional spread,
a near-horizontal planar fit and bounded extrapolation distance. The fitted
bottom interval must be consistent with that plane. The bottom only checks the
ground-contact hypothesis; it does not provide the fall distance.

Under the stated continuity assumption, support and local landing follow the
observed plane. Their bounded difference gives nominal local drop 0 with a
nonzero upper allowance. Both extrapolated surfaces and derived fall height stay
PRIOR. The harmful-fall interaction is also an explicit PRIOR: an unmeasured
opportunity conservatively bounded by 1, with interval [0, 1]. It is not silently
replaced by route crossing or set to zero.

Defaults: 2.0 m floor search radius, 1.5 m maximum extrapolation, at least 20
floor returns, 0.015 m maximum plane RMS, 0.10 maximum slope, 0.06 m maximum
nominal bottom/plane mismatch and 0.15 m maximum combined contact uncertainty.
Plane/drop allowances are engineering bounds, not calibrated confidence
intervals. All method settings are configurable. Missing priors, stale/future
evidence, elevated/buried spheres, poor fits and excessive extrapolation remain
unknown. The strict `lidar_fall_surfaces.py` implementation is unchanged and has
precedence when it supplies observed support/landing evidence.

## Stationary combined validation, 2026-10-08

Fresh captures use the actual mounted Pandar and Hawk RGB, existing object and
contact models, same-track association, and the already implemented estimators.
Fixture position selects the target and authored radius scores geometry offline;
neither authored radius nor authored support height supplies an estimator input.
Pose/extrinsic calibration uses simulation-oracle references, as in earlier runs.

| Stationary case | Small sphere | Larger sphere |
|---|---:|---:|
| Authored radius, offline fixture label | 0.18 m | 0.25 m |
| Target LiDAR returns | 192 | 753 |
| Rough mass, PRIOR | 0.523 kg | 1.443 kg |
| Mass interval, PRIOR | [0.066, 3.123] kg | [0.178, 8.772] kg |
| Local fall interval, PRIOR | [0, 0.0764] m | [0, 0.0771] m |
| Visible-outline contact cue | 0.0914 | 0.1056 |
| Visible-plume cue | 0.000 | 0.000 |
| Conservative CRA phi | 0.957558 | 0.970768 |
| Safe-SAGE normal flux magnitude | 1.649070 | 1.664921 |
| Fixed lambda | 1.25 | 1.25 |
| All four factors available / PRIOR readiness | passed | passed |
| Complete sensor-only readiness | **false** | **false** |

Both use the same configured effective density 25 [5, 100] kg/m3 with confidence
0.35. This is an explicit rough sphere-family assumption, not an estimated
material density or a universal ball density. The nearest observed floor returns
are 1.166 and 1.189 m from the fitted centers; 505 and 396 returns support the
respective planes. This is exactly why floor continuity remains a prior.
Robot drift is 1.44 mm in both runs; scene files are unchanged.

Artifacts are retained in `outputs/ground_support/live_small`, `live_medium`,
and `summary.json`. No authored factor profile is used. All four-factor rejection
lists are empty in the accepted prior-assisted mode; strict sensor-only readiness
still rejects prior mass/support/interaction. No missing factor was changed to 0.

Both phi values exceed the diagnostic saturation threshold 0.95, despite low
visible cues. Their uncertainty uplift dominates: small-sphere pre-uplift
potential/kinetic risks are 0.0234/0.00189, compared with conservative risks
0.6582/0.6507. Thermal is 0.013125 before uplift and 0.457219 after it. The
larger sphere raises mass-dependent risks and phi, but geometry and fit confidence
also differ; this is not an isolated mass ablation or a proof of clearance ordering.
Confidence, priors, energy scales and aggregation were not tuned to lower phi.

## Physical wheel pass-by validation

After both stationary gates passed, two fresh captures computed their own
four-factor profiles and supplied the resulting phi unchanged to the existing
controller. All-four-factor PRIOR readiness was required before wheels could
move. No supplied/oracle factor profile, controller retuning, deadline extension,
goal-tolerance relaxation or dependency change was used.

| Actual wheel run | Small sphere | Larger sphere |
|---|---:|---:|
| Fresh computed phi | 0.957502 | 0.969787 |
| Goal reached / termination | passed / goal_reached | passed / goal_reached |
| Simulated time to goal | 25.4 s | 28.4 s |
| Final goal error | 0.0931 m | 0.0844 m |
| Minimum monitored footprint clearance | 0.1538 m | 0.2047 m |
| Minimum MPC margin | 0.0403 | 0.0307 |
| Minimum analytical CBF margin | 0.2765 | 0.2703 |
| MPC / CBF interventions | 112 / 0 | 125 / 0 |
| 0.5 s stopping check | passed | passed |
| All-factor prior-assisted gate | passed | passed |
| Complete sensor-only readiness | false | false |

Artifacts: `outputs/ground_support/motion_small`, `motion_medium`,
`motion_summary.json` and `pass_by_comparison.png`. Report SHA-256 hashes:

- Small: `b19c78fd26ce20d89be1efd225f3e40d648c492afd8f5604783b0e98ee2d7e07`.
- Larger: `363bf0056b2a62ca70da4d698bd5203256606739e1ccb5ad6c4278e2512a8af7`.

The fields and initial physical/cue assessments are frozen during these static
passes; this is not moving-camera plume estimation or continuous perception
refresh while navigating. Both observed passes go left although the configured
right-side preference is soft guidance. Mean/max MPC wall times are
0.235/0.906 s and 0.302/1.324 s, exceeding the 0.20 s interval in some cases;
real-time feasibility remains unestablished. Both scene-file hashes are unchanged.
Different radii change geometry as well as CRA evidence, so the clearance
difference does not isolate the causal effect of phi.

## Acceptance and remaining scope

The existing demo adds `--ground-continuity-prior` and
`--require-four-factor-readiness`. Ground-prior mode always requires all four;
the latter flag also enables that requirement in strict sensor mode. A failed
factor blocks motion even if the uncertainty-aware controller alone could run. It keeps
sensor-only and prior-assisted readiness as separate report fields. Use
`--object-plume-policy --sphere-density-prior-kg-m3 25 5 100` for this bounded
fixture; `--ground-prior-maximum-extrapolation-m` controls the optional ground
assumption's distance bound. Omit ground opt-in to retain strict unknown support.
`--oracle-comparison` is separate and cannot satisfy the new observed-data gate.

Validation: **1,440 unit tests passed** in 56.59 s; repository Ruff and whitespace
checks passed. New tests cover ground rejection/provenance/time contracts and a
combined image/geometry/motion-to-four-factors-to-phi-to-Safe-SAGE path. Larger
fitted size, higher approach velocity, a pointed mask and stronger plume pixels
each increase their corresponding risk and aggregate phi in independent
synthetic controls. Removing density or floor continuity preserves unknown
factors and phi 1.0. Synthetic masks/bindings are fixture priors, not evidence of
real detector accuracy; the Isaac captures above exercise actual association.

The selected sphere cases passed the bounded prior-assisted simple pass-by test.
They do not establish all-object perception, moving-camera plume estimation,
calibrated hazard probabilities, universal safety, real-time execution or complete
restaurant readiness. A genuinely low aggregate phi would require a separately
justified confidence/uncertainty-policy study; it must not be obtained by silently
weakening the current primary policy.
