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
