# Stationary restaurant server asset

One stationary, stylized server is added in a separate layer over the existing
restaurant. Asset conversion, collision/UV integrity, source preservation and
zero-wheel Isaac rendering pass. **The server's full stationary perception
component gate does not pass; no server wheel or crossing trial was enabled.**
This is completion of the asset-addition stage, not pedestrian-navigation
acceptance or closure of draft PR #117's moving-view perception gate.

## Asset and scene contracts

The source is [Kenney Animated Characters Protagonists](https://kenney.nl/assets/animated-characters-protagonists),
CC0-1.0. The original license, body FBX, idle-animation FBX and selected skin
PNG are retained unchanged under `assets/restaurant/kenney_server/`, with their
official archive URL and SHA-256 fingerprints in `manifest.json`. Existing
Isaac 6.0.1 conversion and USD skinning bake the idle pose at t=0 into static
geometry; no installer, dependency upgrade, executable download or new model
weights were used. The apron is project-authored geometry. The 804-vertex body
retains its indexed UVs and original texture; all collision is static.

The authored 1.75 m height, role name, placement `(29.8,8.9,0)` and final -17
degree yaw are rendering/evaluation metadata. No mass, fall height, sharpness,
temperature, semantic detector class or desired phi is supplied to perception.
The lower sensor-height band does not observe the avatar's complete anatomy.
The catalog-trained mass network is **not calibrated for humans**: its final
1.59 kg output is rough network dataflow, not a credible human-mass measurement
or evidence that human heaviness ordering is adequate.

`restaurant_service_server_v2.usda` and its `_server.usda` additions layer
preserve the original base, robot and dressed restaurant bytes and clock.
The final seven-view Isaac preview passed unchanged-file/bounds, time=0 and
zero-wheel checks. The renderer may edit the in-memory root; it does not save
those edits. Raw preview: `outputs/restaurant_service/server-preview-v3/`.

## Actual stationary sensor results

Each run used 18 distinct RGB/LiDAR observations, no wheel commands, fixed
lambda 1.25 and unchanged mask-quality/coverage/foreign-return/ambiguity gates.
Oracle fixture matching identifies report rows only. The server's independent
.40 m planar match accommodates partial-body returns; it never supplies a
perception association or mask. Existing bowl matching remains .20 m in 3D.

| Run | Result and preserved limitation |
| --- | --- |
| `server-stationary-v1` | Invalid observation setup overlapped counter occupancy; excluded from perception acceptance. An independent asymmetric-footprint setup/settled-clearance guard now rejects such configurations before accepting a trial. |
| `server-stationary-v2` | Valid service-aisle setup; server uniquely matched in all 18 frames, but point prompts produced small-part/low-quality masks. No accepted visual instance. |
| `server-stationary-v3` | Explicit observed-context-box retry; 15/18 accepted server instances, but final competing masks were ambiguous. Fresh final-instance/mass gates failed; wheels blocked. |
| `server-stationary-v4` | Server faces the approaching robot; 7/18 accepted instances and final fresh mass/relative motion. The contact proxy rejects holes/interior boundaries in the actual silhouette; wheels blocked. |

The final v4 assessment has kinetic .187527, potential/contact/thermal .25
configured priors and aggregate phi .657238. Relative speed is .012874 m/s
(stationary jitter); 508 returns are owned by the final accepted mask. The
1.59 kg mass has confidence .25 with explicit category-free domain
extrapolation and visible-envelope limitations. Potential has no justified
support-to-landing drop; object height is not substituted. No valid fresh
plume assessment means a thermal prior, not an invented low-temperature claim.
Contact's unavailable physical inputs remain recorded.

The v4 final mask was replayed from its exact stored RGB and observed prompt:
model quality .853492, with the explicit contact rejection `silhouette contains
holes or unresolved interior boundaries`. This offline diagnostic is not new
sensor evidence. Neither morphology/hole filling nor lower confidence or
ambiguity thresholds was introduced to make the server pass.

The optional generic SAM retry derives a context box solely from the selected
track's projected owned returns, with configurable symmetric padding (default
one observed span) clipped to the current image. It supplies image context, not
physical dimensions. It runs only after point/component attempts fail their
mask gates, never after an ambiguous result. All original acceptance checks,
the .75 evidence-confidence cap, unknown class and unmatched occupancy remain.
Upper-point and multipoint experiments failed and were not integrated.

## Reproduction and validation

Build the dressed restaurant first as documented in
`restaurant_service_build.md`, then use fresh output paths:

```text
python scripts/scene_creation/build_restaurant_server_scene.py
--base scenes/generated/restaurant_service_v2.usda
--output scenes/generated/restaurant_service_server_new.usda
```

Use the existing Isaac environment and the stationary command from that
document, adding `--restaurant-server --restaurant-view server
--sam-observed-context-box` and selecting the new server scene. Do not add
`--motion`: the mandatory component gate currently fails. Any eventual motion
command repeats its stationary gate and also requires the real bowl gate.
No full four-factor or human-motion acceptance follows from numeric scores.
Static occupancy memory supports this stationary actor only.

The asset can be converted again to fresh paths with
`scripts/scene_creation/convert_restaurant_server_isaac.py`; see the asset
README. It requires the installed converter and matching rig joint order.
The regular USD interpreter can warn about the existing remote NovaCarter
reference; the actual Isaac preview loaded it successfully without substitution.

Regression: 1,770 unit tests passed. After the final asset-fingerprint assertion
and formatting, 25 server/SAM tests passed again; repository Ruff and
first-party diff checks passed. Original licensed line endings and serializer
asset bytes remain unchanged. Raw tests/logs are under `outputs/restaurant-*`.

Next work is bounded human-domain mass policy and contact/instance robustness
before server wheel motion, then separately controlled crossings with short
occupancy expiry and current velocity. Permanent static memory cannot support
moving people. The 30.88 m proposal, payload transport, matched final baselines
and robust moving-camera thermal perception remain outside this completed
asset-addition stage. The earlier 3.9466 m service-route arrival and the separate
fixed-camera restaurant plume replay retain their own documented scopes.
