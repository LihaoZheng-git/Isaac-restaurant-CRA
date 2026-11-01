# Robot-camera CRA presentation

`--presentation-risk-camera` is a separate, default-OFF presentation option on
`scripts/demos/cra_live_object_isaac_demo.py`. It requires the restaurant scene
and persistent all-track risk map. It does not change acceptance thresholds,
models, factors, field policy, lambda, nominal commands or controller objectives.

The worker copies every freshly accepted image-LiDAR mask together with its
original RGB frame, camera/LiDAR source IDs and timestamps, individual CRA phi,
applied object/group phi and factor modes. Unaccepted detections, geometric-only
tracks, retained bounding regions and raw person cues cannot create masks.
Accepted class-free SAM objects may still have semantic class `unknown`.

The native recorder accepts this bundle only after its field is admitted. The
additional video combines the existing follow/overview camera with the robot's
latest assessed RGB frame. Mask color uses a fixed continuous 0..1 scale: green,
yellow at .5, orange at .75 and red at 1. No per-frame normalization or
hand-selected object scores are used. Overlapping masks display the maximum
individual risk. Labels show track ID, phi and whether factor priors are used;
an applied group score is shown separately when different. Green means lower
assessed risk, not certified safety. Four configured .25 factor priors aggregate
to .68359375 and are labeled as priors, even if a mask was visually accepted.

The assessment panel updates at perception cadence, not camera/render cadence.
Between results it holds the **original assessed RGB** and displays its source
age. It never paints an old mask onto a new robot view or claims that held cues
were freshly observed. A new result with no accepted masks replaces the old
colored view with an uncolored view. Future-source or field-mismatched records
are withheld. Source age is presentation metadata and never renews evidence or
control clocks. The overview keeps native simulation timing, including pauses.

Outputs in the trial directory:

- `service-navigation.mp4`: unchanged overview/follow recording.
- `service-risk-camera.mp4`: 1440x600 overview plus assessed robot-camera panel.
- `risk-camera-epochs.json`: per-video-frame source IDs, age, masks and provenance.
- `risk-camera-sources/<camera-id>.npz` and `.json`: lossless original RGB, accepted
  masks and score/provenance metadata for each colored assessment.
- `risk-camera-latest.png`: latest colored assessment for inspection.

The separate stationary recipe is `configs/restaurant_risk_camera_v1.json`, case
`camera-preview`. It reproduces the long-route startup settings plus this
presentation flag. Use the existing frozen-source launcher with `--stage
stationary`; it requires committed sources and a new output folder. This recipe
does not authorize or report a new motion trial. The accepted archived service
demo under `outputs/final demo/v1/` remains unchanged.

Initial focused verification: 67 tests passed, including actual all-track worker
parity (no extra inference, identical risk grids), epoch/identity mismatch,
unaccepted masks, retained-frame age, clearing old masks, fixed colors, overlap,
and video pause/frame preservation. Full regression: 3375 tests passed in
119.55 seconds; repository Ruff and compile checks passed. A subsequent focused
regression adds the explicit all-prior human-contact label case. Native stationary
preview is pending at the initial implementation commit. Rendering a mask is
not proof of four fresh factors or validated moving-camera plume estimation.
