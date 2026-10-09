# Paired physical-factor object tests

The later [paired ground/raised sphere navigation test](paired_potential_navigation_test.md)
adds a complete four-factor raised-object gate and support-inclusive wheel trials.
Its explicit matched-geometry policy replay and planar support-dominance limitation
are separate from the potential-only component results below.

These tests extend the [four-factor sphere gate](four_factor_simple_object_readiness.md)
before restaurant enrichment. Factor values and CRA phi are computed from the
estimator outputs, never supplied as experimental profiles. Lambda stays 1.25;
waypoint nominal remains obstacle-unaware. All motion requires stationary
four-factor readiness first. Mass and unseen ground continuity remain explicit
PRIORS, so these are prior-assisted demonstrations, not sensor-only results.

## Thermal comparison

Branch: `codex/cra-thermal-paired-test`. Run the existing simple-object demo with
`--object-radius 0.25 --object-plume-policy --sphere-density-prior-kg-m3 25 5 100
--ground-continuity-prior --require-four-factor-readiness`, changing only
`--plume-control none|intermittent|persistent` and the output directory. Existing
local YOLO, contact and mass weights/environment are unchanged. Add `--motion`
only after the stationary gates pass.

The same actual Hawk RGB object detector and Pandar LiDAR association identify
the sphere. A clean initial RGB reference precedes three fresh stationary
observations. Session-layer puffs change the rendered RGB; they never supply a
thermal score, mask, temperature or class label. They are opaque stylized visual
proxies, not a fluid/steam simulation. LiDAR occupancy is captured before the
puffs appear. Source/air-region association and the unchanged RGB plume backend
compute continuous strength, extent and persistence.

The sphere isolates the thermal intervention from changes in body geometry.
The separate bowl diagnostic retains configured source-region and binding
PRIORS; it is a component test, not a robot detector/navigation result. No bowl
score is transferred to the sphere.

All three stationary conditions passed four-factor PRIOR readiness; sensor-only
readiness remained false. No unknown-factor fallback was used.

| Rendered control | Inferred plume cue | Computed conservative phi |
|---|---:|---:|
| None | 0.000000 | 0.970619 |
| Intermittent, revised | 0.370235 | 0.980808 |
| Persistent | 0.417656 | 0.982479 |

The revised intermittent window still contained weak foreground residuals in
its nominally hidden middle frame (all three frames were positive). Its score
ordering therefore validates combined rendered area/extent/temporal evidence,
not clean on/off persistence timing. RGB timestamps are simulation read times;
the wrapper does not expose independent render-frame acquisition timestamps.
Six physics/render updates separate reads, but hardware-level freshness is not
established. This is a simulation diagnostic limitation.

The separate bowl component passed: cues 0.000000 (none), 0.352861
(intermittent), 0.762454 (dense) and 0.000000 (unrelated region). Unvalidated
reference and moving-camera controls remained unavailable, as required.

Physical-wheel endpoint trials, each freshly assessed and gated:

| Control | Computed phi | Goal time | Minimum footprint clearance | Minimum MPC / CBF margin | Stop |
|---|---:|---:|---:|---:|---|
| No plume | 0.970792 | 29.0 s | 0.20137 m | 0.02293 / 0.26483 | Passed |
| Persistent plume | 0.983087 | 27.6 s | 0.20402 m | 0.02701 / 0.27161 | Passed |

Both completed the waypoint and stopped with positive independent disk-to-full-
robot-footprint clearance. Both actually passed on the left despite soft right-
side guidance. Mean/max MPC wall times were 0.2094/0.7882 s and 0.1986/0.7947 s
against a 0.20 s planning interval. This is not a real-time deadline result.
Scene files were unchanged. The field/assessment is frozen during each static
pass; this does not validate moving-camera plume tracking.

The first intermittent fixture used counts `(1, 0, 1)` and returned cue 0 with
valid observation gates. Its low-contrast small puffs were missed against the
bright window background. This negative result is preserved in
`outputs/thermal_pair/intermittent`. The revised, more resolved fixture uses
`(3, 0, 3)`; the persistent fixture uses `(4, 4, 4)`. No estimator threshold,
confidence, uncertainty uplift, aggregation or controller parameter was changed.
This demonstrates a visibility/contrast limitation: a low cue is not proof that
plume or heat is absent.

The small endpoint clearance difference is one run per condition, with small
sensor/fit variation. It does not establish a causal or monotonic phi-to-clearance
relationship. Factor-specific diagnostics expose the plume response even while
uncertainty keeps aggregate phi high. Physical temperature, aerosol identity,
spill severity and carried state remain unavailable.

Validation: 1,441 unit tests passed; full Ruff check, whitespace and demo syntax
checks passed. Outputs, raw RGB windows, captures, trajectories and model hashes
remain local under `outputs/thermal_pair/`; they are not committed datasets.

Report SHA-256 values are retained in `outputs/thermal_pair/summary.json`.
The actual RGB/results figure is `outputs/thermal_pair/comparison.png`.
Endpoint report SHA-256: no plume
`13ce0b886bde4abe032d00f44c9affd3b1e929837b7a5bd6457b115c3388a15d`;
persistent plume
`9518c2303e363a579b9c7a6b5e6f2a4e39bf62b505d797c3ac128d097ba93c35`.

## Contact comparison

Branch: `codex/cra-contact-paired-test`. The same 0.25 m sphere has either a
rounded outline or an attached visible/collidable cone (0.22 m length, 0.055 m
base radius). Controls change geometry, never the contact score. Add
`--contact-control round|protrusion --seed-color-outline --spherical-body-prior`
to the common stationary command above; leave plume control at `none`.

The first unrefined capture exposed two useful failures: YOLO segmented the
round body but omitted the tip, giving a misleadingly low outline cue (~0.10),
and the full track correctly failed spherical extrapolation. Mass/potential
remained unavailable, phi stayed 1 and motion was blocked. Outputs are preserved
under `outputs/contact_pair/protrusion`.

`robot/perception/seed_color_outline.py` now provides an explicit opt-in for a
validated, separated, mostly uniform-color object. Its observed detector mask
seeds a local connected RGB/HSV component. Circular hue handling, minimum seed
color coverage, bounded area growth and ROI-boundary checks reject unsupported
inputs. Authored masks/colors and class names never define the recovered tip.
The unchanged silhouette estimator scores the refined visible outline and
retains stronger accepted specialist cues. This is a classical RGB refinement,
not a new pretrained general contact model. Gray/multicolor objects, touching
similar colors, hidden points and missed occlusions remain limitations.

The full track's spherical fit remains the preferred path. The optional body
hypothesis is attempted only after it fails and a valid refined outline exists.
The original detector body mask, with the existing six-pixel fusion tolerance,
must retain at least 75% of associated returns and pass all unchanged sphere-fit
gates. The return fraction is **not a bound on excluded mass**. Effective density
then explicitly means whole-assembly mass per fitted body volume, including
unmodeled small appendages; mass remains PRIOR. No authored radius, tip volume
or accurate material mass is supplied to that estimate. Without this scoped
prior, missing mass/support stays unknown.

An exact-mask clipping attempt failed the unchanged ground-contact uncertainty
gate; it remains in `refined_v2_protrusion`. Aligning selection with the existing
six-pixel camera/LiDAR tolerance recovered curved boundary returns. No sphere
residual, floor-consistency, confidence, energy scale or CRA threshold changed.
Full track occupancy, including the tip, always remains in the safety field.

| Stationary control | Visible contact cue | Computed conservative phi | Four-factor PRIOR readiness |
|---|---:|---:|---|
| Rounded | 0.100726 | 0.971371 | Passed |
| Protrusion | 0.812879 | 0.998337 | Passed |

Sensor-only readiness remains false. Physical edge radius, sharpness, hardness,
material injury consequence and hidden-surface safety remain unavailable.
Independent navigation scoring covers the union of the sphere and a disk
enclosing the entire cone projection. Neither authored scoring disk enters the
controller. Geometry changes confound any causal interpretation of clearance.

Both physical-wheel trials also completed their waypoint and stopping gates:

| Control | Fresh computed phi | Goal time | Minimum footprint clearance | Minimum MPC / CBF margin |
|---|---:|---:|---:|---:|
| Rounded | 0.969922 | 29.4 s | 0.05222 m | 0.03197 / 0.15591 |
| Protrusion | 0.997355 | 29.6 s | 0.20102 m | 0.02874 / 0.27324 |

Both passed left. Scene files were unchanged. Mean/max MPC wall times were
0.2343/1.0441 s and 0.2255/0.7991 s, so real-time deadlines are not established.
These single runs show computed evidence reaching the controller; they do not
establish monotonic clearance or isolate risk from geometry/fit variation.
Reports: `outputs/contact_pair/motion_round/report.json` SHA-256
`9bc2ee62f3dbe6c1d190fd4415704b5fbf7c41ea0d8f71d406f908ff5fad2336`;
`motion_protrusion/report.json` SHA-256
`2c3eb810316a2710ed536039a662be13c50a0eb468c0dd40555bfea143375df2`.
Actual RGB outlines and paths: `outputs/contact_pair/comparison.png`.

Validation: 1,457 unit tests passed, including 16 new RGB refinement
cases; full Ruff, whitespace and demo syntax checks passed. Stationary results
are under `outputs/contact_pair/refined_v3_round` and `refined_v3_protrusion`.

## Potential support-drop comparison

Branch: `codex/cra-potential-paired-test`. This is a **stationary potential-only
component diagnostic**, not complete four-factor raised-object perception or
navigation. The existing live Pandar surface method estimates support and a
lower candidate landing band. The 0.14 x 0.14 x 0.34 m object's own height stays
constant while the authored support drop changes. No object/center height is
substituted for fall height, and no wheel motion is commanded.

The diagnostic now exposes bounded fixture distance/width/lateral placement and
landing search range through environment settings. Reproduce the accepted pair
with `ISAAC_SIM_HEADLESS=1`, `CRA_FALL_ISOLATED_SCENE=1`,
`CRA_FALL_FIXTURE_DISTANCE_M=1.3`, `CRA_FALL_SUPPORT_WIDTH_M=0.7`,
`CRA_FALL_OBJECT_LATERAL_M=0.25`, `CRA_FALL_LANDING_MAXIMUM_OFFSET_M=2.0`,
`CRA_FALL_FACTOR_MASS_PRIOR_KG=1,0.5,2`; set `CRA_FALL_SUPPORT_DROP_M` to 0.3 or
0.4 and a fresh `CRA_LIDAR_DIAGNOSTIC_DIR`. Run
`scripts/demos/lidar_fall_height_isaac_diagnostic.py` with the existing Isaac
Python launcher. The potential energy scale remains 100 J; the optional
`CRA_FALL_FACTOR_ENERGY_SCALE_J` setting is recorded in adapter diagnostics.

| Authored drop, scoring only | Inferred drop | Absolute error | Potential factor, before uncertainty uplift | Conservative potential factor |
|---|---:|---:|---:|---:|
| 0.30 m | 0.301437 m | 0.001437 m | 0.074838 | 0.537419 |
| 0.40 m | 0.371669 m | 0.028331 m | 0.088617 | 0.544309 |

Both engineering intervals covered the reference drop and passed stationary
checks. These are single captures, not calibrated accuracy statistics.
Surface bands can mix nearby top/edge/underside returns; the 0.40 m case
underestimated by 2.8 cm. Interval/tolerance acceptance does not identify an
exact support face or prove a falling trajectory. Mass
1 kg [0.5, 2] and harmful-fall opportunity 1 [0, 1] are explicit matched PRIORS,
not mass-network predictions or measured probabilities. Other factor inputs
stay unavailable, aggregate phi remains 1, and no complete-perception or motion
gate is claimed. The factor-specific result shows continuous potential ordering
without silently zeroing missing factors.

Visibility failures are preserved under `outputs/potential_pair/`: original
2.0 m-distance broad-platform cases at drops 0.4/0.6/0.8 failed the current
support-localization gate. Closer broad-platform cases localized support but
lacked accepted landing evidence. A smaller platform at 0.2 m still lacked
adequate support. For the smaller 0.35 m platform, the nearest visible floor was
1.626 m from the object, beyond the configured 1.5 m search; replay at 2.0 m
localized a 0.351 m drop. Fresh matched 0.3/0.4 m captures then passed at that
explicit 2.0 m setting. No surface acceptance threshold or confidence changed;
the production default surface configuration remains unchanged.

The farther band is observed candidate landing evidence. Immediate unseen edge
continuity, exact falling trajectory and impact location are not established.
This fixture's target location and support-relative height filter are configured
selection inputs. The old `oracle_role=offline_scoring_only` label was therefore
incorrect and has been corrected. Actual support/landing heights still come
from live LiDAR inference. A historical 0.8 m report predates the current planar
support checks and does not count as current validation; its failure to reproduce
is retained rather than silently accepting the older result.

Fixture edits stay in the session layer and the saved scene files are unchanged.
Runtime physics setup changes transient root-layer settings; the audit now
distinguishes those from fixture authoring and saved-file changes. Before a
raised-object wheel experiment, validate camera instance separation, all four
object factors and complete platform occupancy in its own stationary gate.

Validation: 79 relevant fall-height/adapter/factor-specific regression tests,
full Ruff, whitespace and syntax checks passed. The earlier full suite passed
1,457 tests. Actual point-cloud figure: `outputs/potential_pair/comparison.png`.
Report SHA-256 values: `final_drop_0.3/report.json`
`f0729ccb3b997379ea563e691872f6869f012f11a4c66d2897c90db8b01defac`;
`final_drop_0.4/report.json`
`3c55b372a37fe4e1af7896f0b5daf13339943d7ff86645d648c19c7b498be4a3`.
