# Thermal cue status

The temporal adapter aggregates fixed-label flame and steam likelihoods across
recent frames. Flame and steam remain separate; smoke detections must not be
relabeled as steam. Neither cue measures temperature. Container openness,
carried state, and spill likelihood remain independent specialist inputs.

D-Fire provides more than 21,000 images with 14,692 fire and 11,865 smoke
bounding boxes. Its official repository places the dataset collection under
CC0 1.0, while noting that the authors do not own copyright in the underlying
public-domain images. The repository links separate trained YOLOv5 models and
an MIT-licensed research-code repository. Those old-model artifacts have not
been downloaded or validated in the current Isaac/PyTorch environment. D-Fire
does not supply steam, hot-liquid, container, spill, interaction, or temperature
labels, so it can support only a flame/smoke detector candidate.

`thermal_cue_frame_from_detections` adapts a fixed-label detector frame without
changing label meaning. `fire`/`flame` confidence becomes an inferred visual
heat-source cue. Smoke confidence is retained for diagnostics but is never
relabeled as steam; the steam field remains unavailable. A missing fire box is
zero only for the supported fire channel. It says nothing about steam,
temperature, hot liquids, or object-specific interaction.

This branch adds no dataset, dependency, or weight. A 3 GB-class D-Fire download
and model training remain separate work, followed by restaurant-domain tests
against visually similar lights, reflections, cooking vapor, and smoke. Until
then, thermal remains unavailable in sensor-derived four-factor CRA runs.

The project thermal scope is now the bounded hot-soup plume example described
in `docs/hot_soup_thermal_perception.md`. The contextual adapter and synthetic
stationary diagnostic do not make the missing detector sensor-derived.
