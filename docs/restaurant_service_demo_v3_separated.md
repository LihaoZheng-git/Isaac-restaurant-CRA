# Separated v3 contact encounters

## Current validated demo

The furniture-gap arrangement at `580b524` passed the six fresh prop/actor/sensor
prerequisites, the mission's own 60 stationary fields and the full native run.
The broken crate is first at (17.5,3.0), followed by the rounded bin at
(21.2,3.2), 3.705m apart. Unused clearing station/caddies, pot and visual steam
are inactive. Original asset meshes/materials/scales, long waypoint route,
one-way human, estimator/retention rules, fixed lambda 1.25, margins and controller
settings remain unchanged. The runtime change in this task is the bounded image
ID namespace correction documented below.

Use `configs/restaurant_service_demo_v3_open_aisle.json`. Native records are in
`outputs/service-demo-v3-open-aisle/long-motion-v1`; the presentation is
`outputs/final demo/v3-separated/restaurant-service-v3-map.mp4`.
It places perception on the left, the route map at upper right and the follow
camera at lower right. All 1,606 navigation/stopping frames plus 60 credit frames
are retained at 10 fps, 1920x1080. The map and human root trace are presentation
data only, never perception or controller inputs. An initial presentation with
clipped map prop labels is retained separately as `v3-separated-layout-v1`;
the final map margin correction changes no native frames, timestamps or scores.

Native acceptance: 8/8 guides, 159.50 simulation seconds, 1,127 live fields with
1,120 committed while translating, 0.345473 m minimum independent clearance,
0.053314 minimum CBF margin and 0.002282 m stopping drift. The nearby human curve is 0.397806 m
relative to the configured nominal polyline, with minimum human-center distance
1.417757 m. There are 27 supported simultaneous moving-kinetic rows. The observed
nearby-walking intervals contain no translation hold below 0.02 m/s or slow interval
below 0.05 m/s. Earlier route pauses remain in the video; this is not a matched
smoothing comparison or a claim that the entire mission is pause-free.

The original-property/current-point audit passes 1,417 retained rows across 15
objects. For the intended props:

| Prop | Retained-evidence rows | Source time range | Applied phi range | Original 20 cm whole-center diagnostic |
| --- | ---: | --- | --- | --- |
| Rounded bin | 79 |68.3--80.2s|0.634475--0.638061|59/79|
| Broken crate | 133 |49.8--114.6s|0.624193--0.873506|49/133|

All selected current returns satisfy the original 3 cm mesh-box allowance. The
crate updates are intermittent, and partial observed surfaces are not complete
object centers. Original mass/contact sources, values, confidence and clocks are
preserved; current LiDAR correspondence is required and kinetic is recomputed.
Maximum property ages are 12.033 s for the bin and 64.967 s for the crate, within
the existing 120 s static-property limit; confidence keeps its existing 60 s half-life.

This is a successful assessed-object navigation demo, **not a validated sharpness
ranking**. The live retained outline cues are bin 1.0 and crate 0.600162, opposite
the desired physical sharpness contrast. The crate's peak phi 0.873506 accompanies
unavailable relative speed and interaction fallback, not stronger sharpness
evidence. Keep that peak and all other rows. Rough mass is uncalibrated;
potential/thermal use labeled 0.25 priors. There are no complete fresh generic
four-factor assessments. Oracle ego reference and labeled human/geometry priors
remain explicit. The simulation is compute-paced at0.025, not real time.

Preserve one sensor-epoch rejection, 16 late human-uncertainty rows and zero native
recovery events. Post-run audits pass: 35,184 geometry references/751 archives,
1,097 ambiguous motion proofs, 4,607 fragment provenance proofs, 218 current body
joins/zero visual-part joins, and 1,150 camera masks across 73 tracks. The camera
audit checks exact field/score/mask and video timing parity. Full 3,851 unit tests
(201.61 s), repository Ruff and diff checks pass. All 17,279 files in the original
demo/asset/native preservation snapshot match their SHA256 hashes. Two completed
failed-run logs received lossless NTFS compression only; before/after content
hashes match and the storage audit is retained.

The final folder includes validation, source/configuration records, frame mapping,
asset credits, checksums and a summary of limitations. This remains one bounded
native run on a draft development branch/PR233. Main/primary/PR117 holds and all
earlier failures remain; there is no merge or universal safety claim.

## Original separated layout and preserved failures

User-approved scene revision, separate from the recorded v3 and v3 map layout.
The original licensed rounded trash can is at (22.0, 4.65), the broken crate at
(25.7, 4.9): 3.708 m between centers. They sit beside the earlier straight aisle
before the one-way human encounter. Original asset meshes/materials/scales stay.
The extra clearing station, closed/utensil caddies, SecondPot and visual steam
are inactive in the new additive layer. The restaurant furniture, serving load,
route, one-way actor, estimation policy and controller settings stay unchanged.

Recipe: `configs/restaurant_service_demo_v3_separated.json`. New scene:
`scenes/generated/restaurant_service_demo_v3_separated_contact.usda`, built with
`scripts/scene_creation/build_restaurant_service_v3.py --separated-contact`.
The original builder behavior remains its default. Explicit deactivated props
are omitted from the independent geometry monitor; missing required structure
still raises an error.

The two zero-wheel views place the robot on the existing straight route, at
(20.15, 4.05, yaw 0) and (23.85, 4.05, yaw 0). These authored setup coordinates
and the selected evaluation target do not enter perception or CRA. The CLI
rejects these pose overrides with wheels. All original image/LiDAR, source,
confidence, geometry, actor, arrival and stopping checks remain. The full-demo
commands differ from v3 only in their scene reference. Moving the crate is
not evidence that its previous initial-acquisition failure is solved.

At ff98d5c, both fresh zero-wheel prop benches passed (60 fields each; bin
28 accepted target assessments, crate 58). Median contact/phi were
0.111879/0.594112 and 0.123943/0.599631. Their ranges overlap: this is a modest
view-dependent proxy contrast, not calibrated sharpness. Potential/thermal
remain labeled 0.25 priors. Actor, component, startup60, near60 and own60 passed.

The first full mission FAILED before reaching either prop: at source ~40.2s,
image group 5000000001 collided with the retained furniture ID for raw track 1.
Five rejected field updates triggered the existing stop. Preserve
`long-motion-v1`: 243 fields/242 translating, minimum clearance 0.526364m,
stop drift 0.001250m, no final arrival acceptance. This is a cross-subsystem ID
collision, not a reason to discard occupancy or relax association checks.

The bounded correction allocates new image groups in [8B,9B), leaving furniture
at 5B+raw ID and all grouping, clocks, geometry, factors and controller gates
unchanged. Old recordings retain their original IDs. A mixed-scene regression
reproduces the old duplicate-ID rejection and verifies distinct retained/current
geometry and policy after correction. All 308 archived candidate sources
(90 accepted hypotheses) replay with identical masks, points and original clocks
under this ID rename; this is component evidence, not native acceptance.
124 focused and 3845 full unit tests (206.73s), Ruff and diff checks pass.
The first full unit invocation was interrupted after sandbox temporary-directory
setup errors; the complete pass uses a fresh workspace-local pytest base directory.
Fresh corrected native gates and full mission remain
required. Outputs: `outputs/service-demo-v3-separated/`.
Old failures, final-demo archives, source assets and main/primary/PR117 holds
remain. A revised recording should preserve the requested left perception
panel and right map/follow-camera panels, with actual evidence provenance.

## Corrected placement after the corner stall

At 7e932b7 all six fresh gates and own60 passed. Bin/crate stationary target
assessments: 48/58 of 60; median contact 0.099756/0.123964, phi
0.588570/0.599638. Full `long-motion-v2` cleared the ID failure and both props
acquired retained evidence, but stalled near world (24.997,4.216), yaw 0.644,
before the human. The final recorded five-second command window had maximum
forward command 0.001252m/s; 194/300 rows waited for a valid plan, 33 were
stale-field stops. Preserve this interrupted run; no arrival/stop acceptance.
The exact source/current-point audit passes: bin 130 retained rows over
74.3--92.8s (phi 0.609335--0.639673), crate only 10 rows over 92.5--93.8s
(phi 0.616032--0.770923). Crate support is therefore brief, not continuous
perception through the failed turn. Original whole-center checks pass 56/130
bin and 10/10 crate rows; all selected points pass the original mesh-box audit.

The first placement survey missed the guide controller's lookahead curve: it
checked straight waypoint chords only. An offline kinematic rollout of the
actual unchanged nominal controller intersects the original crate placement
(79 samples below the unchanged 2cm geometry gate). This explains a placement
conflict; it is not a claim of general controller infeasibility.

Separate `--straight-contact` scene variant uses bin(20.3,4.85) and
crate(24.0,5.1): same 3.708m separation, both 1.7m earlier and 0.2m farther
laterally. Original scene variants remain reproducible. The same nominal
rollout clears the surveyed aisle by 0.330545m, with zero samples below 2cm;
stationary setup clearances are 0.580545/0.530545m. This is an authored survey,
not sensor evidence or native acceptance. Robot route, human, all estimation,
retention, margins, lambda and control settings stay unchanged.

Recipe: `configs/restaurant_service_demo_v3_straight.json`; scenes
`restaurant_service_demo_v3_straight_contact.usda` and `_bench.usda` under
`scenes/generated/`. New outputs: `outputs/service-demo-v3-straight/`.
Require fresh prop/actor/component/startup/near/own60 and full mission gates.
3848 unit tests (198.42s), seven explicit scene/CLI checks and Ruff pass for
this variant. The initial two CLI-test failures (numeric JSON argv values)
are preserved; the recipe now uses strings consistently with existing recipes.

## Furniture-gap placement

The straight variant's bin-only gate at 32cec10 FAILED with zero wheels despite
58/60 fresh target assessments: barstool raw89 centered at (20.2703,5.0818)
is within the unchanged 0.30m competitor distance of the bin surface center
(20.1889,4.8088). Fresh mass/contact and a static record exist, but the legacy
group property-memory prerequisite correctly refuses the ambiguous neighbor.
One initial camera/LiDAR epoch rejection is also retained. Do not weaken or
remove either prerequisite, or call the failed view a passing retention test.

A separate `--open-aisle-contact` arrangement puts the crate first at(17.5,3.0)
and the bin at(21.2,3.2), 3.705m apart in surveyed furniture gaps across the
aisle. The conservative authored prop-to-background separations are >0.52m
for the crate and >0.37m for the bin (0.1--1.8m height raster; not sensing).
The actual nominal-guide rollout has minimum aisle clearance 0.355545m and
zero samples below the original 2cm gate. Stationary robot setup clearances
are 1.359822m (crate) and 0.530545m (bin). Both remain before the guide turn.
No robot/controller/estimator/memory policy or threshold changes.

Recipe `configs/restaurant_service_demo_v3_open_aisle.json`, outputs
`outputs/service-demo-v3-open-aisle/`. All earlier layouts/failures stay.
Fresh gates and full mission are required; no native acceptance yet.
3851 unit tests (201.61s), Ruff and diff checks pass for this placement recipe.
