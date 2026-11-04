# Separated v3 contact encounters

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
