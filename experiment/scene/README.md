# Restaurant scene setup

The v2 restaurant uses the included USD layers, the NVIDIA AEC Restaurant Demo Pack, and the project's converted props and character rigs. The preparation step verifies the local asset fingerprints and writes a small scene directory. It runs in Python 3.12 without starting Isaac Sim.

## Prepare the assets

1. Extract NVIDIA's **AEC Restaurant Demo Pack** (`AECO_RestaurantDemoPack_NVD`, version 10012). Use the directory containing `Demos/AEC/TowerDemo/RestaurantDemopack` as `--restaurant-pack`. Retain the package licenses. [NVIDIA downloadable content packs](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html)
2. For `--live-generated`, use the existing live-stage `scenes/generated` directory or a copy of its v2 runtime assets. On a new machine, obtain the converted v1 dishware/trolleys and retargeted v5 child from the demo asset archive maintained by the project authors. Keep the exported textures alongside their USD files. The v3 character folders use the NVIDIA Rigged Characters pack (`Characters_NVD@10012.zip`) and the Isaac 6.0 construction character with its textures. [Cast sources](data/cast.json) · [Exact file list and SHA-256 hashes](data/inputs.json)
3. Keep the original NVIDIA Isaac 6.0 asset URLs accessible to the Isaac resolver during a native run. The robot, business-character rig and human motion library load through that resolver.

The live asset directory uses these paths:

```text
restaurant_live_v1/assets/          Converted dishware, trolleys, textures, and intake manifest
restaurant_live_v3/characters/      official-pack/ and official-native/
restaurant_live_v5/characters/      rocketbox/Male_Child_01.usdc and textures/
```

From the repository root, replace the example input paths with your local directories:

```powershell
python experiment/scene/prepare.py --restaurant-pack "D:/data/Isaac_restaurant" --live-generated "D:/data/restaurant-live/scenes/generated" --output experiment/scene/prepared
```

Pass `experiment/scene/prepared` to the [experiment launcher](../README.md) as its scene root. Prepared manifests reference the local input paths. If the checkout or inputs move, prepare a fresh output directory at the new location.

## Asset credits

- **Inge Dinnerware Set**, abradburywallad, and **Kitchen Asset Library Pack photoreal Vol.1**, Davilion, CC0. Selected dishware and tools were converted to USD with baked kitchen materials. [Dinnerware](https://blendswap.com/blend/30224) · [Kitchen assets](https://blendswap.com/blend/25903)
- **Plate Trolley**, FractDesign / Huguet Rémi, and **Pink RÅSKOG Utility Trolley**, W1NT3R, CC BY 4.0. Both underwent glTF-to-USD conversion. The moving plate trolley was scaled to a height of 0.90 m. [Plate trolley](https://sketchfab.com/3d-models/plate-trolley-a7bfc7169ef94faca7ba4419b7fcae98) · [Utility trolley](https://sketchfab.com/3d-models/ikea-raskog-pink-utility-trolley-32cf0f02bbdd47d491ea7287945cfaad) · [License](https://creativecommons.org/licenses/by/4.0/)
- **Microsoft Rocketbox Male Child 01**, Microsoft, MIT license. The project converted the FBX to USD and mapped the rig for native animation. Retain the original license with any copy. [Source model and license](https://github.com/microsoft/Microsoft-Rocketbox/tree/0943055db6ec570bcef9f2c8b41c9e5467c808f9/Assets/Avatars/Children/Male_Child_01)
- The repository includes the referenced Kenney food and Sketchfab prop conversions with their original credits. [Kenney](../../assets/restaurant/kenney_food_kit/README.md) · [Bin and crate](../../assets/restaurant/sketchfab_contact_props/README.md)
