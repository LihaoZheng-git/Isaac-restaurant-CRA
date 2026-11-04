# Restaurant service demo v3

Final recorded route completed at runtime revision `72b89ed`: 8/8 guides,
152.600 simulation seconds, minimum independent clearance 0.473051 m,
stopping drift 0.002291 m, and 1083 live fields (1081 while translating).
The shareable package is `outputs/final demo/v3/restaurant-service-v3.mp4`.
It preserves all 1537 navigation/stopping frames and adds six seconds of asset
credits. Playback is 1x simulation time; execution was paced at .025, not real time.

The original automatic report is **FAIL**, preserved unchanged. Its only failing
navigation gates came from the legacy parallel-actor region `y <= 11`, although
this explicitly selected one-way protocol starts at `y=23.5`. The separate
straight-walk check had already passed. The corrected evaluator requires the
explicit protocol, uses its authored start with the same 0.5 m region allowance,
keeps every original speed/limb/grounding/geometry check, and additionally requires
the complete one-way audit. `scripts/evaluation/reevaluate_straight_service.py`
exactly reproduces the old summaries before re-evaluating the same archived data.
The corrected post-run audit passes; this is **not a new native wheel run**.
Original reports, exception, input hashes, and corrected reports are all included.
The far-start human-mass diagnostic has no five retained human sources; it is
preserved as a non-gating diagnostic. Fresh near-human and live retention gates pass.

The nearby curve is 0.371162 m from the actual nominal polyline, at 1.426776 m
human/robot center separation. Minimum human center separation is 1.407279 m.
The longest actual translation hold while walking within 2 m is one simulation
frame (0.016667 s); the longest interval below .05 m/s is .05 s. This is one scene
variant, not a matched smoothing or causal CRA comparison. The human visual
identity expires after its original 30 seconds near mission end: 19 uncertainty
rows remain behind the robot with zero occupied cells in the current field.
No identity clock was renewed and no native part-recovery event occurred.

The bin retains original verified mass/contact in 82 current-point rows from
103.0 to116.5 s, up to13.6667 s property age; phi remains live at .62516--.63256.
All82 satisfy the separate original center/mesh-box diagnostics. The full source
audit passes1220 retained rows across13 objects with exact source/value/clock/
confidence/current-point replay. **The broken crate does not acquire a qualified
image-LiDAR group in this full mission**, so it uses ordinary obstacle policy.
An early surface reference and later newly visible surfaces prevent the initial
proof; applying the existing2.5m range limit earlier also fails in read-only replay.
No gate was relaxed. The passing16/16 bin and crate short approaches are included
as separate component videos; they do not establish full-mission crate acquisition
or a contact-risk contrast. Pot thermal response remains unvalidated.

All seven scoped archive audits pass: geometry34459 references/752 archives,
1067 motion proofs,5498 fragment provenance proofs,217 current human body joins/
one visual-part join, exact camera parity for943 masks/65 tracks, static properties,
and zero-event recovery accounting. All6692 previously protected files are unchanged.
Final validation:3838 unit tests in210.25 s,61 focused tests, Ruff and diff checks.
Draft/main/primary/PR117 holds remain. The earlier stages and failures below are
historical records, not superseded claims of full perception acceptance.

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
