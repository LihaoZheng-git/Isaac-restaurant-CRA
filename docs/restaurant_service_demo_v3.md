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

First actor-only native run0540da4 completed16.364m with zero held drift,
no reverse motion and all geometry/grounding/walking checks, but FAILED the
new .25m straight-aisle check: maximum lateral gait drift .32446m. Preserve
actor-v1. The next actor-only variant uses collinear2m guides with no braking
until the final endpoint, to correct accumulated native heading drift. It keeps
the same start/goal/speed, unchanged .25m gate and all physical checks.
Artifacts: `outputs/service-demo-v3/`; recipe:
`configs/restaurant_service_demo_v3.json`. Full native v3 validation pending.

The corrected actor run5e51871 passed:16.364m straight travel, maximum lateral
deviation .06578m, zero held drift, minimum layout clearance .20277m. The fresh
human component, startup60 and near60 also passed. Full mission long-motion-v1
passed its own60 and reached both props and the human pass, but was interrupted
after a post-pass stall; no final arrival/stopping acceptance is claimed.

Two concrete integration issues are preserved in that run. Static props inherited
the mixed scene's human3s expiry/1.5s confidence half-life, instead of their
intended120s/60s. Bin64 and crate68 did acquire real evidence, but expired early.
The correction gives static props their own configurable clock and exports it;
human property configuration and source timestamps remain unchanged.

The later stall was caused by compute freshness, not disappearance of raw human
returns: ID2437 remains tracked at130.9s and later, but an inference rejection
and unpublished recovery baseline left a committed130.2->130.9s gap. This exceeds
the unchanged.6s identity continuity gate. The resulting reachable region
expanded across the local field. The separate corrected recipe uses the already
tested near-bench simulation pace.025 instead of.04; physical speeds, freshness,
identity, uncertainty, CRA, MPC/CBF and arrival/stop thresholds stay unchanged.
This is compute-paced simulation, never real-time performance. Fresh exact-source
prerequisites and a full mission remain mandatory for the corrected variant.

Corrected configuration validation:25 focused tests,3825 full unit tests in
207.40s, and Ruff/diff checks pass. Legacy native source audit passes138 retained
rows across10 objects; independent mesh-box diagnostics identify15 bin rows
(103.1--105.9s) and6 crate rows(105.7--106.7s). These are original3s-clock
diagnostics, not corrected native acceptance. Original sources, clocks, values,
decay and current selected-point joins replay without errors.
