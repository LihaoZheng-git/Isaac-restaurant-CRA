# Kenney Food Kit subset

Creator: [Kenney](https://kenney.nl/assets/food-kit), Food Kit 2.0.
License: **CC0-1.0**; the original `License.txt` is included unchanged.
The official archive was retrieved on 2026-10-16. `manifest.json` records its
download URL, archive SHA-256, selected members, individual hashes and sizes.
No executable files from the archive are retained or run.

This repository includes 13 small original OBJ/MTL models and their shared
palette texture. Eleven placed instances currently dress existing restaurant
tables and the service counter. These are stylized visual assets, not calibrated
physical models. No source mass, density, temperature, sharpness, semantic risk
or CRA score is supplied to perception or control.

`usd/` contains project conversions made with the **already installed** trimesh
4.11.1, NumPy 2.3.1, Pillow 12.1.1 and USD runtime. Conversion rotates source
Y-up coordinates to Z-up, moves the actual mesh base to z=0, retains UVs and
uses the original palette through `UsdPreviewSurface`. Static triangle-mesh
collision preserves cavities and utensils rather than replacing them with a
solid convex bowl. Metric placement sizes are explicit authoring choices.
Collision geometry does not establish physical sharpness or food stability.

Reconvert and build into a **new** scene path from the repository root:

```powershell
& 'C:/Users/tayli/anaconda3/envs/env_60_isaacsim_yolo/python.exe' scripts/scene_creation/build_restaurant_service_scene.py --convert-assets --output scenes/generated/restaurant_service_new.usda
```

The script changes DLL lookup only within its own process on Windows. It does
not install or upgrade packages. `.gitattributes` preserves original source
bytes across Git checkouts so the intake audit remains reproducible.
The restaurant/Nova Carter source files and their separate NVIDIA asset terms
are unchanged; the Food Kit CC0 license does not relicense those assets.
