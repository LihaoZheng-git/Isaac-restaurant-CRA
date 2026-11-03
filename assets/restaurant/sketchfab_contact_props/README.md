# Licensed props for stationary perception tests

Original glTF files/textures were downloaded by the user from Sketchfab on
2026-11-03. Each source folder retains the supplied attribution/license text;
each manifest records the original ZIP SHA256 and extracted file hashes.

- **Blue recycle bin**, SUSUSUBE, CC BY 4.0:
  https://sketchfab.com/3d-models/blue-recycle-bin-5516523f31f343e68da70629555291de
- **Wooden Dirty Crates - Broken Whole 2 Piece Set**, Sunbox Games, CC BY 4.0:
  https://sketchfab.com/3d-models/wooden-dirty-crates-broken-whole-2-piece-set-1ebaf13ac44748baa845ab7f5cb3fc9f
- **Kitchen Cleaver**, KAMAAIL, CC BY 4.0:
  https://sketchfab.com/3d-models/kitchen-cleaver-93d395fa7cf04811a7cc56f5169c30d5
- **Broken Plate (low poly)**, Akimkhan, CC BY 4.0 (original Poly by Google
  credit on the source listing):
  https://sketchfab.com/3d-models/broken-plate-low-poly-1cb31d29ec0d4289b43f1794fa05abe1

Share these credits with any derived scene images/video. No object identity,
physical mass, material density, contact label or CRA score is supplied to the
estimator by these assets.

The installed Isaac converter produces `converted.usdc` and copied textures.
`scripts/scene_creation/convert_restaurant_contact_assets.py` reproduces that
conversion into fresh paths; it verifies original source hashes first.

Separate scene layers select the original textured 803-triangle assembled bin
(the download also has duplicate component meshes and overlapping high-poly
baking meshes), set authored bin height to 0.65 m, and rotate/ground the prop.
The crates use the same uniform 0.65 scale: intact mesh `Crate_01_GEO`, broken
mesh `Crate_02_GEO`. Original broken boards and materials are retained. These
authored dimensions and names are evaluation metadata, never perception evidence.

Original source geometry/textures are unchanged. The bin is an ordinary object
example, not a sharp-object label. A broken-crate contact contrast must be
established from accepted sensor observations, not from its name.

Cleaver and plate keep all three original meshes each. The cleaver's maximum
horizontal extent is 0.32 m, lying flat on a separate 0.45 m-high preparation
surface. The broken plate's maximum extent is 0.28 m, lying on the floor.
Uniform scaling preserves the original thin geometry; no blade/shard thickness
is enlarged to pass LiDAR gates. Support geometry is independently reported and
never included in the cleaver's authored evaluation bounds.
