# Static restaurant staff avatar

Source: [Kenney Animated Characters Protagonists](https://kenney.nl/assets/animated-characters-protagonists),
licensed **CC0-1.0**, retrieved 2026-10-17. The unchanged original license and
selected FBX/PNG bytes are included; `manifest.json` records official download
URL, archive fingerprint and all four original member hashes.

The installed Isaac Sim 6.0.1 `omni.kit.asset_converter` converts the body and
idle-animation FBX. Matching joint order is required. USD skinning evaluates
the original idle pose at t=0, then exports 804 posed vertices as a static Z-up
triangle/polygon mesh with original indexed UVs and skin texture. Authored
height is 1.75 m. White apron geometry is a project addition; the source PNG is
unchanged. This is a stylized role asset, not calibrated human anatomy.

The avatar has static collision geometry, no authored rigid mass, temperature,
physical sharpness, semantic detector label or desired CRA score. Its authored
height and placement never enter perception. The current asset is stationary;
it is not an integrated walking/interaction model. No downloaded executable,
installer, model weights or dependency upgrade was used.

Rebuild to fresh paths using the existing Isaac Python environment:

```text
C:/isaacsim/python.bat -u scripts/scene_creation/convert_restaurant_server_isaac.py
--work outputs/server-conversion-new --output outputs/server-conversion-new-asset/server.usda
```

`usd/server.usda` and `server.json` are the accepted local conversion and its
fingerprint. Original source bytes are preserved by `.gitattributes`.
The source restaurant/NVIDIA robot retain their separate licenses.
