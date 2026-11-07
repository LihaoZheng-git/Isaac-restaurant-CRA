# LiDAR risk-map presentation

The next restaurant presentation adds a north-up, robot-centred 2D LiDAR
risk map beside the original perception and follow-camera views. It renders
the recorded, accepted V3 bin/crate service mission. This is a new presentation
of that mission, not a new navigation trial or an estimator improvement.

## Meaning of the display

Each observed point receives its recorded final controller policy from
`diagnostics.applied_field_policies[].applied_field_risk`, after current
ownership assignment and policy adjustments. The fixed green-to-red scale is
0 to 1; it is never normalised to the frame's minimum and maximum. A numerical
risk is not a binary declaration of safety or a calibrated collision probability.

The original camera overlay reports an object's individual CRA score and its
recorded group-policy value when different. That camera value is captured
before later field-policy adjustments. The LiDAR map uses the final field policy. The two
panels therefore need not have identical colours. The map displays obstacle
policy inputs, not the solved Poisson barrier or a collision-free-space map.
Colour encodes applied phi only; boundary flux, tangential strength, and
configuration-space padding are not additional colour channels.

Solid points are the latest committed field's source observations. Their source
age is displayed; they are not represented as a new scan at every video frame.
Separate hollow geometry depicts retained occupancy and geometry priors,
including the human envelope. The recorded nominal route and travelled path
provide context. The robot marker uses recorded simulator localization.
The active field boundary is drawn explicitly. Source observations outside
that controller window are grey; their assigned policies are still preserved
in the exported arrays.

Evidence summaries distinguish a current visual assessment, retained physical
properties, and ordinary obstacle policies. Ordinary policies may include
inferred motion and configured priors. A visual assessment may still include
individual factor priors. Retained mass/contact requires the original current
point association; the visualization does not propagate properties to unmatched
fragments. Motion-dependent risk is taken from each committed source policy.

## Source and scope

Native input: `F:/CRA-experiments/service-v3-balanced-pair/original-v3/long-motion-v1`.
The recorded runtime revision is `0047241`. The previous presentation remains
in `outputs/final demo/v3-bin-crate-validated`.

That native mission reached all eight guides in 159.0333 simulation seconds,
with 0.352070 m minimum independent clearance and 0.002164 m stopping drift.
All pauses and stopping frames are retained. These are inherited results,
not new validation of a different controller.

The known limits remain: rough mass is uncalibrated, potential and thermal
often use explicit priors, and crate property association is intermittent.
In particular, no retained crate assessment qualified in the 27 fields directly
alongside the crate. Its fallback policy must remain visible in those fields.
The full mission's bin/crate local trajectories are nearly straight; the more
curved responses belong to the separately recorded closer N12 trials.

The archived map does not improve tracking, update a model, select a new path,
or provide a real-time performance claim. Main/primary and historical promotion
holds remain in effect.

## Reproduction and checks

Use the existing `env_60_isaacsim_yolo` environment. No package installation or
Isaac launch is needed. Choose fresh output paths for each command:

1. `scripts/evaluation/export_lidar_risk_map.py --source <native-directory> --output <export-directory>`
   reconstructs saved ownership, checks final per-object counts and placement
   totals, and exports point policies and geometry. It supports the recorded
   V3 recipe with instance-bounds retention disabled, and rejects unsupported
   ownership modes rather than guessing.
2. `scripts/evaluation/audit_lidar_risk_export.py --manifest <export-directory>/manifest.json --output <audit.json>`
   independently compares every current source XY, final policy value and count
   to the original records, and checks video field receipts and source hashes.
3. `scripts/evaluation/compose_restaurant_lidar_risk.py --source <native-directory> --archive "outputs/final demo/v3" --risk-manifest <export-directory>/manifest.json --output <presentation-directory>`
   composes every native navigation/stopping frame and the original credit card.
   `--preview-only` creates review images first. Repeat `--preserve` for other
   demo packages whose manifests must be verified during composition.

Video-to-field alignment follows each original camera frame's admitted sensor
field ID. Timestamp comparison alone would incorrectly select a new field
committed after recording a frame at the same simulation time. Original source
and output geometry hashes are checked before rendering.

The fixed palette, overlapping-point precedence, active-field bounds, source
clocks, late human policy changes, static fragment ownership, corrupt scores,
and missing/unsupported input cases have focused regression coverage. The full
unit suite passed 4,022 tests in 289.59 s. After the final reconstruction and
display refinements, 52 focused tests passed in 6.22 s; repository Ruff passed.

The final clean export and independent audit passed for all 1,182 original
fields: 32,721,816 current point assignments and 23,853,998 prior/retained
geometry samples. All 3,980 source-file hashes matched. The 1,670 original
video epochs used their recorded field IDs; 105 same-time later commits were
correctly withheld from earlier video frames.

Preview source frame 1354 demonstrates the requested camera-independent
display: it has no accepted image mask, while 1,203 current LiDAR returns
inside the active field still qualify for retained physical properties.
These points retain the original association and policy values.

Two preliminary extraction attempts stopped on ambiguous historical ownership
and were preserved with failure records. The final export ran cleanly from all
original sources, using the recorded all-or-none ownership counts after image
and static-property assignments. The later human mass-policy update is not
mistaken for the earlier ownership decision.

The full presentation is saved separately in `outputs/final demo/v4-lidar-risk`.
The master `restaurant-service-v4-lidar-risk.mp4` contains all 1,601 native
navigation/stopping frames plus 60 original attribution frames: 1,661 frames,
166.1 s, 1920 x 1080 at 10 fps. Full master decoding passed. The package
includes preview images, source/frame mapping, asset credits and file hashes.
The previous 428 demo/archive files passed checksum verification unchanged.
