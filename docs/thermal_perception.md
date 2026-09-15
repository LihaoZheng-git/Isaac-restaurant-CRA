# Thermal cue status

The temporal adapter aggregates fixed-label flame and steam likelihoods across
recent frames. Flame and steam remain separate; smoke detections must not be
relabeled as steam. Neither cue measures temperature. Container openness,
carried state, and spill likelihood remain independent specialist inputs.

D-Fire provides YOLO fire/smoke labels and links to trained examples, but its
repository license and candidate weight license still need a direct audit
before download or integration. It does not supply steam or hot-liquid labels.
No external dataset, dependency, or weights are added by this branch.
