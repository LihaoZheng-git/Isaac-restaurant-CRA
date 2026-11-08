# V6 service trial with additional prop clearance

This separate placement trial moves the broken crate to world (17.50, 3.10)
and the rounded bin to (21.20, 3.40). Relative to the accepted calibrated V5
layout, these are +10 cm and +20 cm in world Y toward the unchanged y=4.05
aisle route. They leave 35 cm and 25 cm more route clearance, respectively,
than the interrupted V6 closer-prop placement. Their longitudinal separation
remains 3.70 m. This is a new layout with no native acceptance result yet.

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

Implementation checks pass: 4,255 full unit tests in237.23s, 27 focused
builder/recipe/CLI tests in6.15s, repository Ruff and diff checks. The actual
composed-scene proof verifies144 unique parent/source hashes, exactly the
bin+.20Y/crate+.10Y translation properties, unchanged other props/actor/timing/
units, and scene-path-only changes in all six command arrays. Validation
artifacts are in`outputs/service-v6-clearance-preparation/`. These source and
scene checks are not a native placement or mission result.

Implementation validation: 27 focused builder, preservation and CLI tests pass
in 6.15 s; Ruff on the changed Python files and the diff whitespace check pass.
Independent inspection of the actual generated layers verifies 144 unique
source hashes, exactly the two translation properties, unchanged other prop
bounds and metadata, actor state, stage timing and units. All six command arrays
differ from the accepted recipe only in scene paths. The builder-generated
tracked recipe matches its saved snapshot byte-for-byte, SHA256
`4e30ef8c63a07284e8103927e5f2a146c55e1d2b897ed8f4c0a48f7008a6d1dc`.
Proofs and logs are under `outputs/service-v6-clearance-preparation/`.
Standalone USD cannot resolve the remote Carter/human assets; these local
geometry checks do not replace native component and stationary validation.
Full regression and native results are pending at this protocol snapshot.
