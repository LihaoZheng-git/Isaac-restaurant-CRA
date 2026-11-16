# V6 service trial with additional prop clearance

This separate placement trial moves the broken crate to world (17.50, 3.10)
and the rounded bin to (21.20, 3.40). Relative to the accepted calibrated V5
layout, these are +10 cm and +20 cm in world Y toward the unchanged y=4.05
aisle route. They leave 35 cm and 25 cm more route clearance, respectively,
than the interrupted V6 closer-prop placement. Their longitudinal separation
remains 3.70 m. The native trial passed its seven parked/component gates but
stalled near the crate and was explicitly interrupted. It is not an accepted
new demo; the completed calibrated V5 remains the current full presentation.

The previous V6 attempt reached the crate approach and was interrupted after
sustained low progress. Its partial native sources, missing final report and
stopping acceptance, unfinalized videos and preservation proofs remain intact.
The previously successful approximately 20 cm bends were isolated N12 tests at
different positions. They do not validate these new positions or establish
full-service N6 success. The separate bounded diagnostic recipe remains
prepared but unlaunched; this placement does not depend on its results.

`scripts/scene_creation/build_service_pair_layout.py --layout v6-clearance`
creates additive layers over the accepted `service_pair_original_v3` scene.
Only the two prop translations change. Meshes, scales, physical properties,
furniture, actor timing and walking path remain unchanged. The scene directory
is `scenes/generated/service_pair_v6_clearance/`; original contact and bench
basenames remain for the established native CLI contract.

`configs/restaurant_service_v6_clearance_calibrated.json` derives directly from
the accepted calibrated witness recipe. All six command arrays change only
their scene paths. Preserve N6 with 0.2 s intervals, guidance tracking weight
100, fixed lambda=1.25, B=1.7 boundary mapping, calibrated CRA scores and palette,
the obstacle-unaware nominal waypoints, physical wheel limits, analytical CBF,
evidence ownership and clocks, and every uncertainty rule. The persistent
background position allowance stays at 0.30 m. No longer horizon, altered
prior, object-specific score or authored avoidance path is introduced.

Build into a fresh directory and recipe path, for example:

```powershell
python scripts/scene_creation/build_service_pair_layout.py --layout v6-clearance --output-directory scenes/generated/service_pair_v6_clearance --recipe outputs/service-v6-clearance-preparation/generated-recipe.json
```

Compare generated and tracked recipes exactly. Verify composed prop bounds,
the two translation-only overlays, actor/timing/units and original source
hashes. The builder refuses existing output paths. Keep historical parent
revision metadata as provenance; the native invocation records the actual
frozen revision, recipe and runtime hashes.

Before motion, require six fresh gates for this scene: bin stationary60,
crate stationary60, idle actor, moving human component, full-route startup60
and nearby-human60. The full mission must also pass its own stationary60.
Use fresh native outputs under
`F:/CRA-experiments/service-v6-clearance-20261108`, with the same committed
source and scene hashes throughout. Authored obstacle geometry and PhysX
contacts are passive evaluation data, not runtime controller inputs.

Then evaluate actual arrival, stopping, clearance, contact coverage and source
integrity; independently audit retained evidence, current ownership, static
geometry, motion, fragment proofs and camera/map joins. Measure prop-local
deviation against the actual nominal polyline and keep all pauses. Arrival
alone is insufficient for acceptance. A larger curve is not promised; report
what the unchanged controller produces with the extra maneuvering space.

Only after native and source-policy acceptance, create the separate package
`outputs/final demo/v6-clearance`: assessed camera at left, controller-applied
LiDAR risk map at upper right and native follow view at lower right. Preserve
native frame order, pauses, source timestamps, fixed palette and licensed
credits. Decode the complete master and share video and verify earlier demo
hashes. Never replace V5 or the interrupted V6 files with this trial.

Potential and thermal remain labeled priors; mass/contact remain rough visual
estimates with intermittent qualified association. Ego localization is
simulated and execution is paced. This placement comparison is not a risk-only
causal result, continuous fresh four-factor sensing, real-time proof or exact
Safe-SAGE equivalence. Main, primary-comparison and PR117 holds remain.

Implementation validation: 4,255 full unit tests pass in 237.23 s and 27
focused builder, preservation and CLI tests pass in 6.15 s. Repository Ruff
and the diff whitespace check pass.
Independent inspection of the actual generated layers verifies 144 unique
source hashes, exactly the two translation properties, unchanged other prop
bounds and metadata, actor state, stage timing and units. All six command arrays
differ from the accepted recipe only in scene paths. The builder-generated
tracked recipe matches its saved snapshot byte-for-byte, SHA256
`4e30ef8c63a07284e8103927e5f2a146c55e1d2b897ed8f4c0a48f7008a6d1dc`.
Proofs and logs are under `outputs/service-v6-clearance-preparation/`.
Standalone USD cannot resolve the remote Carter/human assets; these local
geometry checks do not replace native component and stationary validation.

## Native result and preservation

The run at `f46efdc0e2741984eef173ec63ce0460323751f9` passed the bin and crate
stationary60, idle actor, moving-human component, startup60 and nearby-human60
prerequisites, then its own60. One initial camera/LiDAR epoch rejection remains
recorded. Motion reached the first route guide and the crate-centre region,
then stalled. The final field is source75.5/commit75.8, local pose
(9.768353,-0.022642,-0.023692), world XY(17.568777,4.035959). The robot centre
passed the crate centre; full rear passage, the bin encounter and final arrival
were not accepted.

Source67.5 through75.5 records only 0.448 mm net movement and 3.62 mm total
drift, with every field numeric-ready. The last two saved navigation seconds
contain 121/121 zero-forward commands, all `waiting_for_fresh_valid_plan`.
The final five contain 299/301 zero-forward commands. The last command
checkpoint is elapsed navigation65.1833; source/commit timestamps are absolute
simulation time and are not interchangeable with that clock.

The process was explicitly interrupted after this sustained stall and all
native processes exited. It did not complete a 300-second timeout. There is no
final native report, launcher execution result, braking acceptance or complete
contact/arrival evaluation. All three partial MP4s lack closing metadata and
remain unplayable; none is presented as a completed demo.

`outputs/service-v6-clearance/interruption-v1.json` records this outcome.
Its preservation proof hashes 4,897 native files (4,164,726,733 bytes), checks
395 runtime files, 150 scene/source hashes, the recipe, 17 bound helpers and
six historical audit hashes, and verifies all 488 previous demo files and all
4,243 first-interruption files unchanged. The archive contains 472 complete
live-field records and no malformed rows. The corrected partial-trajectory
figure is `interruption-artifacts-v1/interrupted-trajectory-v2.png`; earlier
plots and the native archive remain intact.

## Exact diagnosis

The two saved failed requests at source66.2 and70.1 reproduce the native
failure statuses, entire returned command sequences and predicted chassis
states using their original field, temporal derivative, nominal sequence,
lookahead goal, warm start and numerical-library versions. No environment
upgrade or horizon comparison was made. Both replays fail the MPC's dense
padded geometry check. The immediate analytical CBF would preserve the nominal
approximately 0.19988 m/s command: current h is about 0.1443 and its constraint
margin about 0.3744. This distinguishes the failed plan from a CBF stop or an
independent authored-world veto.

The rejecting cells belong to persistent background regions7000001259 and
7000001290: seven and four original points from source54.1 and55.8. Both retain
the approved 0.30 m position allowance, fixed unknown phi0.68359375,
b=-1.3203125 and lambda1.25. Their original XY points lie inside the authored
crate box, used only for post-run attribution. Current returns are 0.79--1.12 m
from these retained points at the replay sources, so the existing 0.10 m
replacement-witness test cannot retire them. This does not establish physical
crate movement or missing current sensor fields.

At source66.2, the actual rectangular footprint including its original 5 cm
margin is 9.61 cm clear of the eleven retained 0.30 m point disks in these two
regions; this is not whole-scene or PhysX clearance. However, one
of the original padded hold hypotheses overlaps a disk by 2.65 mm; the nominal
horizon's worst sampled overlap is 6.79 cm. Finer yaw/raster resolution would
not remove these continuous overlaps for the unchanged hold and nominal
candidates, although it could affect optimizer behavior. Grid rasterization
also contributes: the
hold's rejecting negative-yaw slice remains 5.92 cm clear of the exact disks
when evaluated as a rectangle at that quantized yaw.

The unchanged geometry uses 5 cm cells and 16 yaw slices. The physical chassis
extents are front0.14/rear0.587351/half-width0.2466 m; the safety reference is
0.10 m forward, yielding front0.04/rear0.687351 m at that reference. The padded
queries use +/-0.025 m in the local-field x/y axes and +/-0.21 rad yaw, with
the original first-interval ramp. The continuous-geometry diagnostic evaluates
the same finite Cartesian query set; it is not exhaustive uncertainty coverage.
Holding, slow forward motion and small rotations fail the native dense check
despite positive endpoint/DTCBF margins. These finite probes do not prove that
all possible trajectories are infeasible or that N12 would fix the full route.

Exact inputs, probes, ownership, numerical provenance, hashes and the reviewed
figure are in `outputs/service-v6-clearance/interruption-diagnosis-v1/`.
`README.md`, `exact-request-probes.json`, `geometry-resolution-audit.json` and
`continuous-versus-raster-geometry.png` explain the distinction. Exactly two
original MPC requests were replayed; the geometry comparison ran no optimizer.
Native inputs remained byte-identical and the three diagnostic scripts pass
Ruff. Runtime/scene files stayed frozen through preservation and diagnosis;
this result documentation was updated afterward.

No new full-demo video is accepted. Both isolated N12 detour-and-arrival clips
remain valid within their documented limits, but do not validate this N6 full
mission. Preserve V5, both interrupted V6 attempts and all existing holds.
The next design review should address the interaction of retained background
geometry and padded full-route planning. Do not silently shrink uncertainty,
discard retained points, alter phi, relax the human-horizon contract or declare
an untested placement successful to obtain a more visible curve.
