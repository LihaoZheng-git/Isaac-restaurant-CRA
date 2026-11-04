# Restaurant service demo v3

Separate development branch after both static-prop short tests passed at674c893.
The accepted v2 archive, source layers, licensed assets and earlier failures stay
unchanged. Draft/main/primary/PR117 holds remain.

The new additive scene copies the original Simple Trash Can and broken-crate
asset specifications, retaining mesh topology, materials and uniform scale.
The bin is placed at(28.15,4.9); the crate at(28.15,6.1), beside the clearing
area. These authored coordinates are used only for rendering and independent
evaluation. CRA still uses actual camera/LiDAR observations. The existing pot
puffs remain visual presentation; new thermal evidence is not promised.

The front Owl camera is raised .60m with its original lens/640 resolution.
`--service-static-props` enables the passing static-location evidence method in
the mixed scene. Generic visual/human assessment stays active; current person
mask hits and known tracked person IDs exclude static property acquisition/use.
Such exclusions leave ordinary occupancy and human control policy intact.
Retained mass/contact keep original clocks/expiry/confidence; kinetic and final
phi stay live. Current projected LiDAR points are explicitly labeled retained
evidence, not a fresh SAM silhouette. No saved whole-object reconstruction is
claimed.

The actor-only `--straight-service-walk` starts the native human at(29.8,23.5),
holds it at the far end, then releases one straight walk to(29.8,7) when the
robot reaches world x26. This trigger was chosen from the previous route timing
to have the human already walking when encountered. It affects only the actor;
no actor pose, route, speed or schedule supplies CRA/MPC/CBF inputs. No loop or
return walk occurs. The native actor geometry/grounding checks remain, with a
separate straight-route audit. A preliminary authored square-footprint sweep
has minimum .252m clearance; it is not native acceptance.

The robot retains the existing long service waypoints, wide corner, fixedlambda
1.25, online guidance/MPC/CBF, physical wheel limits and arrival/stopping gates.
Fresh actor-only, human sensor, near stationary and distant-start stationary
gates plus the mission's own60 are required before navigation acceptance.
Artifacts: `outputs/service-demo-v3/`; recipe:
`configs/restaurant_service_demo_v3.json`. Full native v3 validation pending.
