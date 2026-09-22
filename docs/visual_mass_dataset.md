# Visual mass dataset decision

The general-object path collects RGB and object masks in controlled Isaac Sim
scenes with mass labels from an authored USD mass, a documented density/volume
calculation, or a physical scale. These are oracle training/evaluation labels;
they must never enter runtime CRA inference.

Splits use physical `object_id`, preventing multiple views of one asset from
leaking across train, validation, and test. The manifest records scene, camera,
seed, class, dimensions, mass, and mass source. Collection stores RGB and masks
losslessly. Before training, inventory multiple independent instances per
class—not merely views of one mesh—and randomize viewpoint, lighting, texture,
background, and container fill state.

No external mass model is integrated. Image2Mass has legacy Keras requirements
and no verified repository license. CORSMAL is container-specific; some related
components are noncommercial. Architecture selection waits until the dataset
inventory demonstrates enough independent objects. Real-image evaluation is
required before simulator-trained output can be called runtime evidence.

## ABO restaurant subset

The bounded real-image path uses the official Amazon Berkeley Objects (ABO)
product listings and catalog images. The current ABO download page and archive
README identify the release as CC BY 4.0. Preserve the archive license and give
the required Amazon.com and dataset-author attribution in publications and
derived-dataset documentation.

Run `scripts/data/prepare_abo_visual_mass_subset.py` against the 16 official
`listings_*.json.gz` shards and `images/metadata/images.csv.gz`. The selector:

- uses exact product-type mappings for restaurant-relevant classes;
- requires a positive catalog weight, all three catalog dimensions, and an
  indexed main image;
- converts normalized pounds and inches to kilograms and metres;
- caps each class before training and assigns splits by product object ID; and
- records the source license and provenance in every manifest row.

ABO values are external catalog metadata claims. They are offline supervision,
not scale measurements made by this project, and they remain unavailable to
the CRA runtime. The selector does not turn object height into fall height.
Raw archives, selected images, and generated manifests live under `outputs/`
and are intentionally excluded from Git.

Example:

```powershell
python scripts/data/prepare_abo_visual_mass_subset.py `
  --listings-dir outputs/visual_mass_data/abo/raw/listings/metadata `
  --image-metadata outputs/visual_mass_data/abo/raw/images.csv.gz `
  --output-dir outputs/visual_mass_data/abo/subset `
  --max-per-class 200
```

This branch prepares and inventories data only. It does not contain trained
weights or claim general-object mass inference. A later training branch will
choose a lightweight architecture after checking the selected class balance,
image availability, pretrained-weight license, and real/Isaac validation plan.
