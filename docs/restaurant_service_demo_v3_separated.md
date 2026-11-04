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
