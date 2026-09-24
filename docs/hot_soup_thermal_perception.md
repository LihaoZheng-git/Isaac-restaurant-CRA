# Hot-soup thermal cue scope

The bounded thermal example is an open soup container with a visible plume.
The runtime adapter requires three fresh visual inputs: plume likelihood,
open-container likelihood, and the likelihood that the plume lies above the
container. It uses their minimum as an inferred heat-source proxy, so a strong
generic smoke detection cannot hide missing or weak container context.

The output is not temperature, burn severity, fluid temperature, or proof that
the plume is steam. A detector labeled `smoke_like` or `visible_plume` remains
unavailable in the steam field. Only a fixed-label steam detector can populate
that field, and even that remains a visual cue rather than a thermometer.
Thermal interaction with the robot route is a separate input to the physical
factor adapter. Without it, the CRA thermal factor remains unknown.

## Candidate-model audit

[D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset) supplies fire
and smoke boxes, not steam, soup, open-container, spatial, temperature, or spill
labels. [SAINet](https://github.com/AlterMundi/sai-sainet-latest) publishes an
AGPL-3.0 wildfire checkpoint, but its data preparation explicitly treats steam
as a hard negative. It is therefore not a justified direct hot-soup steam
detector. Neither model has been downloaded or claimed as integrated on this
branch.

[The Isaac Sim 5.0 Replicator Incident documentation](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/action_and_event_data_generation/tutorial_replicator_incident.html)
describes fire and smoke event generation. The installed Isaac Sim 6.0
environment does not contain the `isaacsim.replicator.incident` extension, so
the stationary diagnostic authors
a small USD visual proxy: translucent plume geometry above a bowl. This tests
scene construction for future detector input, not aerosol physics or detector
accuracy.

## Validation boundary

`hot_soup_cues_demo.py` uses synthetic prior fixtures and verifies the context
gate plus the separate CRA interaction requirement. The Isaac demo exports a
stationary visual plume scene without wheel commands. A development capture
visually showed the bowl, soup surface, and plume, but repeated automated
headless capture was not reliable in the installed Isaac 6.0 environment and
is not treated as a passing regression. Actual perception still needs
a steam/cooking-vapor dataset, restaurant-domain negative examples, trained
weights with a compatible license, an open-container detector, and spatial
association validation on real and Isaac images.

## Project-owned dataset path

`collect_hot_soup_data.py` ingests project-owned stationary images or samples
frames from a project-owned video. Every capture records its source reference,
usage rights, domain, content hash, and capture group. Collected frames remain
`pending`; an unlabeled frame is never treated as a negative example.

The annotation contract in `robot/perception/hot_soup_dataset.py` requires
explicit present/absent decisions for both `visible_hot_soup_plume` and
`open_container`. Real labels are human visual annotations. Isaac labels remain
`isaac_authored_oracle`. Hard-negative tags cover smoke, glare, haze, white
backgrounds, and empty containers. They supervise visual discrimination only;
they do not say that an object is thermally safe.

Capture groups receive deterministic train/validation/test splits so adjacent
video frames cannot cross splits. Validation also rejects the same image hash
across splits. `prepare_hot_soup_detection_dataset.py` exports validated boxes
and explicit empty negative labels to an Ultralytics-compatible detection
layout. No checkpoint is trained or integrated by this dataset branch. Before
training, the project still needs owned real hot-soup footage, completed human
annotations, a documented train/evaluation protocol, and a final license check
for the selected pretrained backbone and released derivative weights.
