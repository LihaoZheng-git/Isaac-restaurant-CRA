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

`--kinetic-diagnostic` animates the sphere only after the zero-wheel acquisition
gate and freezes accepted intrinsic cues. It records a 12 s stopped/slow/stopped/
fast-reverse/stopped profile. The .12/.24 m/s scripted velocities are independent
oracle evaluation truth; the tracker/CRA receives actual simulated RGB/LiDAR and
ego pose references, not actor truth. .4 s intervals around phase changes are
excluded explicitly to avoid claiming instantaneous accumulated-scan/filter
response. Per-return ego deskew still does not correct actor motion.

The first `kinetic-stationary-v1` run passed speed/energy ordering: stopped median
.0173 m/s; .12 m/s truth produced .0980 m/s; .24 m/s truth produced .1814 m/s.
Median vector error was .0286/.0659 m/s for slow/fast phases. It has measurable
underestimation, not exact velocity calibration. The repeat gate additionally
requires current returns to own the computed policy, robot stationarity, scene
invariants and bounded old-occupancy age.

The moving-actor opt-in may translate earlier accepted instance bounds by the
**observed track centroid displacement**, after the original identity/competitor
and property-persistence checks. It is an unchanged-shape, planar-translation
association PRIOR, not a mask measurement, actor truth or rotation/occlusion
solution. Original bounds/property source times are retained; unmatched returns
remain separately occupied with priors. Legacy static bounds stay the default.
Moved support is invalidated independently. Positive occupancy has an explicit
.15 s expiry in this diagnostic and never establishes free space or preserves
moving actors indefinitely. Unit tests verify original epochs, translated
ownership and expiry; no confidence/energy/phi threshold is tuned to the fixture.

The final `kinetic-stationary-v2` gate **passed** all eight checks, including
policy ownership, scene invariants and robot stationarity. Steady-phase usable
counts were 12/22/12/22/12. Median relative speeds were .0174/.0955/.0161/.1805/
.0159 m/s; slow/fast median vector errors were .0283/.0686 m/s. Energy severity
ordered .000622 < .001384 < .002553 for initial stopped/slow/fast phases. These
engineering upper-bound severities are small under the unchanged 100 J scale;
aggregate phi also responds to current route interaction, support invalidation,
confidence decay and factor priors, not velocity alone.

Maximum robot displacement was .01618 m and yaw .000293 rad; zero wheel commands
were issued. Maximum retained-occupancy source age was .10 s under the .15 s limit.
Full regression: 1,691 passed, with repository-wide Ruff and diff checks passed.
The sensor plot is `outputs/cra_readiness/kinetic-stationary-v2/kinetic-validation.png`.
This is a stationary-robot dynamic-actor gate, not yet simultaneous robot/actor
navigation or a general moving-human model.

Combined navigation must use each obstacle's computed policy with all occupancy,
fresh exact MPC/DTCBF and analytical CBF checks, actual waypoint arrival, clearance
and safe stopping. Record simulation pacing, latency, field/evidence/prior counts,
identity failures and safety stops. Slowed simulation is not real-time operation.
Repeated combined acceptance is required before claiming restaurant readiness.

The first `combined-live-v1` three-hazard navigation trial **failed** its 100 s
simulation budget: final local state (1.67695, -.34533, .27263), 184 committed
fields/168 translating updates, minimum independent fixture clearance .08894 m,
minimum analytical CBF margin .17076, stop drift .000938 m. Pacing was .1
simulation s/wall s; wall duration after stationarity was 1,233 s. Clearance and
safe stopping do not replace waypoint arrival. Latency/temporal-gap rejections
and safe stops remain recorded; this is not a successful pass-by.

The live demo now exposes a separately recorded frozen-property vision retry
interval (legacy 1 s; bounded experiment 5 s). It never slows current geometry,
motion, route interaction or numeric scoring, renews sources or relaxes freshness.
Stage timers separate perception/ownership/occupancy from field construction.
An offline representative **uniform-prior** sensor-occupancy profile identified
Python configuration-space propagation and connected-component work; it is not
a four-factor or navigation result.

The opt-in vectorized configuration expansion preserves the exact legacy
risk/absolute-flux/known-ID/first-source overlap priority, all occupancy, metadata
and counters. Vectorized equal-policy connected-region grouping preserves
eight-neighbor geometry and deterministic seed labels; nearest-distance tie
normal sums are aggregated exactly. Tests compare every yaw's metadata and
complete guidance/h_full values, gradients and forcing to the legacy path
(1e-12 tolerance). No phi rounding, discarded geometry, PDE tolerance, safety
constraint, nominal or lambda changes are involved. Motion acceptance of these
optimizations is still a separate gate.

`combined-live-v2` also **failed** the 100 s budget with 5 s vision retries:
216 fields/187 translating, final (1.61156, -.37671, .27181), clearance .14714 m,
CBF margin .14131, stop drift .000905 m, 1,264 wall seconds. Faster field
construction was then verified locally (1,699 unit tests and Ruff passed).
The optimized `.05`-paced `combined-live-v3` reached a similar tight-layout
standoff; it was explicitly aborted, with 280-field checkpoints preserved at
source time 38.1, local state (1.67886, -.36601, .23151). It has **no completed
arrival or safe-stop report** and is not acceptance. These mixed-risk tight
layouts expose a bounded unicycle-navigation/liveness limitation; numerical
field equivalence is not global convergence.

The separate selected readiness layout moves the visible-plume sphere from
local x=2.2 to x=3.0 and uses waypoint (3.8, 0), preserving sphere/support sizes
and all estimator inputs/gates. This is a different, explicitly separated
scenario, not a successful replay of the failed tight layout. A 20 s no-progress
watchdog can only fail a run; arrival remains mandatory.

Its first stationary gate (`separated-live-v1`) correctly blocked wheels: mass,
contact and fall checks passed but the 140-degree rectified plume source was
unresolved. The opt-in calibrated 120-degree view preserves more native source
detail; the narrower view is recorded and retains unchanged confidence, source
timestamps, clean-reference/background/coverage and pixel-count gates. Native
source mask pixel counts are carried separately, and an unresolved native
source cannot pass by remap upsampling. Mass/contact/fusion remain in native
camera pixels. `separated-live-v2` passed all seven independent stationary
checks before its wheel trial; motion acceptance is still recorded separately.

The combined moving-actor gate also compares actual relative speed with the
independent actor velocity minus the source-epoch ego velocity. It requires at
least five distinct, owned, evidence-supported simultaneous-motion updates and
median world-velocity/relative-speed errors below .10 m/s, excluding .4 s phase
transitions. Actor truth never supplies estimator inputs. This remains a
separated translating-sphere diagnostic, not human/crowd acceptance.

Full regression reached 1,701 passed; the final source-deduplication change also
passed its affected kinetic tests, and Ruff/diff checks passed. Two earlier
invocations had 113 temporary-directory setup errors (Windows access denied),
not assertion failures. The successful run used a fresh workspace `--basetemp`;
no package or environment upgrade was made.

Long multi-object diagnostics can produce nearly a gigabyte of indented records.
The explicit `--stream-live-records` opt-in appends every complete record once to
`live-fields.jsonl`; `live-fields.json` is only the latest-five checkpoint in
that mode. The report names the storage contract. It requires a fresh stream,
retains every original source epoch and factor/provenance record, and changes
neither scores nor control verification. This avoids repeated full-history
serialization hitches; simulation pacing and any remaining latency are still
reported, with no real-time claim.

Mass calibration, physical sharpness/hardness, temperature, moving-view plume
re-estimation, crowded identity crossings, hidden heat/support changes and causal
CRA superiority remain outside these bounded component gates.
