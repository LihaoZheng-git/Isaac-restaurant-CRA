# Restaurant scene setup

From the repository root, run this command with Python 3.12:

```sh
python -m experiment.scene.prepare
```

The checkout includes the converted dishware, trolleys, child character, textures, and restaurant scene layers. Setup fetches the restaurant, adult characters, and their materials directly from NVIDIA. The first run downloads about 5 GB. Allow 12 GB of free space for the downloads and extracted files. Subsequent runs reuse the cache.

Setup prints the path to the prepared USD stage. You can open that file in Isaac Sim to inspect the restaurant. To run the animated delivery with the controller, follow the [experiment guide](../README.md). Isaac Sim also loads Carter, a business character, and motion clips from NVIDIA at runtime.

### Cache and existing downloads

The default cache is `~/.cache/isaac-restaurant-cra/v2`. Use `--cache-dir` to choose where the downloads are stored. Keep the cache and checkout in place while using the prepared stage.

If you already extracted the NVIDIA restaurant pack, use `--restaurant-pack` to point to the folder containing `Demos/`. If you already have the complete original live assets, pass their `scenes/generated` directory with `--live-generated`. To prepare a second copy or change locations, choose a new directory with `--output`.

### Asset credits

- Restaurant environment and adult characters: [NVIDIA Omniverse content packs](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) and [cast sources](data/cast.json).
- Converted tableware, trolleys and Microsoft Rocketbox child: [sources, modifications and licenses](../../assets/restaurant/live/README.md).
- Additional food, bin and crate models: [Kenney food kit](../../assets/restaurant/kenney_food_kit/README.md) and [Sketchfab prop credits](../../assets/restaurant/sketchfab_contact_props/README.md).
