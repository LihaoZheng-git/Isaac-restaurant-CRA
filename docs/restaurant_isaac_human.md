# Official Isaac human restaurant asset

The user's preferred restaurant human now comes from **NVIDIA Isaac Sim's
People library**, replacing the Kenney cartoon in the new primary server
scene. The older Kenney scenes and failed diagnostic records are preserved as
historical experiments; they are not overwritten.

Source: `Isaac/People/Characters/F_Business_02/F_Business_02.usd`, under the
installed Isaac 6.0 asset root. NVIDIA documents this library in its
[character configuration guide](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/action_and_event_data_generation/ext_replicator-agent/ext_isaacsim_replicator_agent_configuration.html).
The actual accessible top-level asset is 728,407 bytes, SHA-256
`a1dceaae15e59e1cda4b4697a1e233c0cc570279178a418e544289b202c78201`.
The builder checks this fingerprint before use. Geometry and textures remain
external NVIDIA references, governed by their
[additional software/materials terms](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/),
and are not vendored into Git. Texture dependencies are external and are not
individually pinned by this top-level USD fingerprint.

## Scene contract

- Native metre/Z-up geometry is retained; the computed posed height is 1.60669
  m. No manually chosen replacement human height or mass is supplied.
- The project authors a static arms-down pose in the rig's own rest coordinates,
  using the observed source topology to lower each upper arm. Geometry,
  materials, textures and original asset files are unchanged. This is not a
  retargeted idle or walking animation. The checked standing clip has an
  incompatible skeleton (zero matching joint names) and was not integrated.
- CPU skinning computes the posed bounds for grounding, render extents and a
  separate static guide-purpose collision capsule (radius .30033 m). That
  capsule is a conservative scene collision proxy, not human anatomy, a mass
  model or an estimator input. No rigid mass or physical risk labels are authored.
- A new server additions layer preserves the original restaurant, robot and
  dressed-prop layers and their simulation clock. Placement remains
  `(29.8,8.9,0)`, yaw -17 degrees. Asset names, authored pose/bounds and collision
  metadata are rendering/evaluation information only.
- Runtime vision, CRA policies, lambda, nominal/MPC/CBF control and acceptance
  gates are unchanged. The generic monitor label is now `static_staff_avatar`
  so a native NVIDIA character is not mislabeled CC0.

`restaurant_service_isaac_server_v2.usda` is the accepted replacement build.
The earlier v1 build rejected a duplicate source transform operation; the
corrected builder puts grounding in its own parent transform without replacing
the character's existing transform order.

## Validation and use

The actual seven-view Isaac preview under
`outputs/restaurant_service/isaac-server-preview-v1/` passed: source files and
prop/server bounds unchanged, timeline 0, zero wheel commands and one static
human. The preview uses real rendered NVIDIA geometry and textures, not an
image-generated person. In-memory renderer edits are allowed and are not saved.

The separate `isaac-server-stationary-v1` RGB/LiDAR run passed the existing
18-frame server component gate with zero wheel commands: 18 uniquely matched
updates, 18 accepted visual instances, one track (55), and 358 owned returns
in the final assessment. Final relative speed was .002029 m/s (stationary
jitter). Final kinetic/contact scores were .187525/.175; potential and thermal
remained explicit .25 priors, yielding phi .622961. The contact value is the
visible-silhouette proxy, not physical sharpness or injury severity.

The unchanged catalog mass network produced 1.921 kg at confidence .25. This is
**not credible human mass or validated human heaviness ordering**. The component
gate establishes accepted-mask/geometry/network dataflow, not mass accuracy.
The unsupported fall height and heat inputs remain unavailable, with labeled
numeric control priors. The strict fresh four-factor readiness gate remains
false. No wheel or walking/crossing acceptance is claimed by this stationary
component result; all original source clocks and gates were retained.

Four new original-fixture unit tests cover lowered arm endpoints, unchanged
height, time-invariant pose, source preservation, separate static collision,
no mass input, fingerprint rejection, unsupported topology and overwrite
rejection. Full regression: **1,774 unit tests passed**; repository Ruff passed.

Build and preview with the already installed Isaac environment, using fresh
output paths:

```text
C:/isaacsim/python.bat -u scripts/scene_creation/build_restaurant_isaac_server_scene.py
--base scenes/generated/restaurant_service_v2.usda
--output scenes/generated/restaurant_service_isaac_server_new.usda

C:/isaacsim/python.bat -u scripts/demos/restaurant_service_preview_isaac.py
--scene scenes/generated/restaurant_service_isaac_server_new.usda
--output outputs/restaurant_service/isaac-server-preview-new
```

The builder requires access to the audited official asset and its dependencies;
it stops on an unavailable or changed source rather than selecting another
character. No packages, weights, external retargeter, motion extension or
environment upgrade was installed. This character can serve the staff role
without claiming that its original clothing is a dedicated server uniform.

For a zero-wheel RGB/LiDAR check use the stationary server command in
`restaurant_server.md` with the new scene. A human appearance replacement does
not validate human mass, contact injury, temperature, walking/crossing behavior
or robust moving-view estimation. Preserve the sensor gate's actual outcome;
no wheel motion is authorized by asset/render acceptance alone. The original
uncalibrated catalog mass model and explicit missing-factor priors remain.

This child branch targets the live development branch. Draft PR #117's
separate main-merge hold remains unchanged.
