# Rounded trash can and broken crate comparison

The user selected the downloaded Simple Trash Can as a candidate rounded
restaurant prop, with the existing broken crate as the contact-risk contrast.
The vase experiments and their failed rankings remain preserved. This branch
adds a separate stationary benchmark, not a promoted estimator or navigation
demo. Main, primary and PR117 holds remain.

## Fixed setup

- Original Simple Trash Can by Andrew.Mischenko, CC BY 4.0; supplied license,
  ZIP and extracted-file hashes are in its own asset directory. The original
  668-triangle body/lid mesh and material are retained. Uniform scale authors
  a 0.50 m height and approximately 0.296 m diameter, without geometry edits.
- Existing intact and broken Sunbox Games crates retain their common 0.65
  source scale. They provide an additional within-asset control. Bin versus
  crate is a demo comparison, not a size/material-controlled physical study.
- Fresh separate scenes place each target at the same existing (28.15, 6.10)
  test location. Original Owl camera, +0.60 m level mount, robot (26.3, 6.1),
  same light/background and LiDAR. No humans or wheel commands.
- Each case collects 18 distinct sources at 640 and 1920 image width. All
  sources, including qualification/startup and unavailable-factor rows, remain.
  These resolutions do not constitute viewpoint generalization.
- Existing `--stationary-image-objects --local-contact-geometry` with fixed
  thresholds. Contact crop refinement is OFF for all cases: its earlier crate
  replay reversed the ordering. This choice is explicit before new native
  results and does not establish that refinement works on these props.

No class/name, authored size, material, mass or desired score enters perception.
Original association/quality gates, confidence, uncertainty priors, CRA mapping,
fixed lambda and raw occupancy remain. Potential/thermal may use labeled
priors, mass is uncalibrated and simulator ego pose supplies the frame reference.

## Validation

Use `scripts/scene_creation/build_restaurant_real_asset_bench.py` with the
unchanged `restaurant_service_demo_v2d.usda` base and fresh outputs for
`simple_trash_can`, `intact_crate`, `broken_crate`. Then run
`scripts/evaluation/run_restaurant_demo.py --config
configs/restaurant_rounded_trash_contact.json --case <kind>[_rgb1920]
--stage stationary --output <fresh-folder>` from the installed environment.

Evaluate component acceptance separately from risk ranking. Report all accepted
objects, usable contact cues, priors, cue/factor/phi ranges and medians. Inspect
actual masks and preserve every adverse result. A component pass is not a
sharpness or navigation pass; no score overrides, row removal or prior reduction
are allowed to manufacture the expected comparison. Native results pending.
