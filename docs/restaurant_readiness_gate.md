# Bounded CRA readiness before restaurant enrichment

The accepted target is three gates: separated multi-object factor/phi evaluation,
moving-object kinetic evaluation, and repeated combined live navigation. A gate
does not require every input to be measured: unavailable physical inputs keep
their diagnostics and receive explicitly configured numeric factor priors.
These experiments retain fixed lambda 1.25 and obstacle-unaware waypoint nominal.
Restaurant assets are not saved or enriched by these session-only fixtures.

## Separated multi-object evidence

`cra_live_object_isaac_demo.py --fixture-layout three
--persistent-object-risk-map --plume-control persistent` renders three matte
spheres: a grounded object, a sphere on a .30 m support, and a ground object
with visible opaque puffs. Fixture poses, sizes and support height are authored
oracle **evaluation/rendering** metadata, never factor inputs. The estimators
still receive only copied RGB/LiDAR, simulation pose/extrinsic references, and
explicit domain/interaction priors. No weights or packages are downloaded.

The opt-in 3D connected components preserve original return indices; world-frame
velocity and track association remain planar. This separates the visible raised
sphere from its support in the bounded scene, not general 3D object discovery.
All support, background, unclustered and rejected-mask occupancy stays in the
field. Inference remains category-free through audited SAM and the existing
RGB/LiDAR mass network; exact kilograms are not an acceptance target.

The separate camera visibility opt-ins use observed nearby projected depths and
a first mask depth layer separated by .5 m, with at least eight foreground hits.
They are explicit sparse/separated-depth association assumptions, not complete
occlusion reconstruction. Nearby competing returns still count as foreign;
SAM quality, image-boundary, coverage and ambiguity thresholds are unchanged.
Farther returns excluded from camera association remain in physical occupancy.
Mask-based depth layers must not be described as directly measured visibility.

The support estimator's optional equal-height landing mode requires all candidate
lower landing directions to agree in height. It may return an inferred vertical
drop with an **unknown impact direction/site**, separately from the legacy strict
directional mode. The spherical bottom is only a support-consistency gate.
The 2.5 m landing-search extent and worst-case harmful-fall interaction are
configured priors; object/center height never supplies fall height. Grounded
objects retain the earlier observed-plane/local-continuity prior path.

`scripts/evaluation/cra_object_map_report.py --require-fixture-evidence` matches
recorded assessments to independent oracle fixture centers for reporting only.
Ambiguous matches fail rather than selecting a favorable score. It requires
accepted mass/contact observations for each sphere, grounded low drops, a
positive support-to-landing drop for the raised sphere, and a positive plume cue.
It recomputes .10/.25/.50 prior sensitivity **offline**, preserving the recorded
factor estimates. These sweeps never control wheels or claim calibrated priors.

Earlier failed diagnostics remain in `outputs/cra_readiness/`: v1 had partial
plume-object occlusion and planar support/object mixing; v2 showed a low-contrast
plume miss and ambiguous landing directions; v3 aborted on a new diagnostic
type error (fixed with tests); v4 measured .283 m raised drop and .534 plume cue
but rejected the ground association; v5 accepted ground/plume evidence but
rejected the overlapping raised source. None establishes successful combined
perception. Visible puffs are a proxy fixture, not steam physics or temperature.

The final `multi-stationary-v6` zero-wheel fixture gate **passed** all seven
independent checks over 18 fresh fields. Ground/plume spheres each had 18 accepted
visual updates; the raised sphere had 14. Latest inferred mass outputs were
.421/.127/.280 kg (rough model outputs, not mass truth). Ground/plume local drops
were 0 with explicit continuity priors; raised support-to-landing drop was .25046 m.
The ground plume cue was 0; the visible-puff cue was .53137; raised thermal stayed
unavailable and used its .25 numeric prior.

| Fixture | Thermal/contact/potential/kinetic control scores | Computed phi | Offline phi with .10/.25/.50 priors |
| --- | --- | --- | --- |
| Ground sphere | .1171/.1627/.1999/.1876 | .5195 | .3049/.5195/.7715 |
| Raised sphere | .2500/.0852/.2253/.1875 | .5682 | .3272/.5682/.8436 |
| Visible-puff sphere | .1586/.0858/.1952/.1875 | .4970 | .2660/.4970/.7605 |

All factors include their recorded uncertainty/interaction policy. The positive
plume cue raises its thermal factor relative to the observed no-plume ground
source, but aggregate ordering also includes different visible geometry, mass,
route context and missing factors; these are **not** controlled thermal/potential
causal comparisons. Priors noticeably affect phi, so .25 remains an explicit
uncalibrated engineering setting, not an optimized probability.

Validation: full unit suite 1,687 passed; final affected regression 84 passed;
repository-wide Ruff and diff checks passed. Artifacts and prior failed attempts
are retained under `outputs/cra_readiness/`; `object-validation.json` records
per-factor provenance, association failures and the offline sweeps. Reproduce
with 18 stationary frames and `--save-sensor-snapshots`, then run the report's
`--require-fixture-evidence`. No wheel motion occurred in this gate.

## Remaining gates

The moving-object test must keep velocity current, distinguish sensor estimates
from independently authored motion truth, invalidate moved support priors, and
report stale occupancy. A stationary development gate precedes any robot wheel
motion. Static occupancy memory is not appropriate for moving actors.

Combined navigation must use each obstacle's computed policy with all occupancy,
fresh exact MPC/DTCBF and analytical CBF checks, actual waypoint arrival, clearance
and safe stopping. Record simulation pacing, latency, field/evidence/prior counts,
identity failures and safety stops. Slowed simulation is not real-time operation.
Repeated combined acceptance is required before claiming restaurant readiness.

Mass calibration, physical sharpness/hardness, temperature, moving-view plume
re-estimation, crowded identity crossings, hidden heat/support changes and causal
CRA superiority remain outside these bounded component gates.
