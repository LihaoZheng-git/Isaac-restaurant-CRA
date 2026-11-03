# Licensed props for stationary perception tests

Original glTF files/textures were downloaded by the user from Sketchfab on
2026-11-03. Each source folder retains the supplied attribution/license text;
each manifest records the original ZIP SHA256 and extracted file hashes.

- **Blue recycle bin**, SUSUSUBE, CC BY 4.0:
  https://sketchfab.com/3d-models/blue-recycle-bin-5516523f31f343e68da70629555291de
- **Wooden Dirty Crates - Broken Whole 2 Piece Set**, Sunbox Games, CC BY 4.0:
  https://sketchfab.com/3d-models/wooden-dirty-crates-broken-whole-2-piece-set-1ebaf13ac44748baa845ab7f5cb3fc9f

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
