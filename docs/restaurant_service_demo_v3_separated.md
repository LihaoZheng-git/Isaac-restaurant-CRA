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

Fresh validation is pending. Outputs: `outputs/service-demo-v3-separated/`.
Old failures, final-demo archives, source assets and main/primary/PR117 holds
remain. A revised recording should preserve the requested left perception
panel and right map/follow-camera panels, with actual evidence provenance.
