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
are allowed to manufacture the expected comparison.

## Original-distance outcome and closer follow-up

All six fresh 18-source runs at `f6d8f88` retain frozen source hashes and zero
wheel commands. At both resolutions the bin has only ONE accepted source and
fails its final component gate. Original frame66 has 372 raw target returns;
frame84 has 371, but their whole-cloud correspondence gap is 0.062895 m,
exceeding the original 0.06 m. A second hypothesis is enrolled. Later sources
match both references and reject as ambiguous, preventing a continuous proof.
This is association failure, not evidence that the bin is sharp or invisible.
The whole-raw-cluster diagnostic does not substitute for masked ownership proof.

The 640 crate controls both pass 16/18 accepted assessments. Broken-crate
high-resolution ownership fails entirely; the intact high-resolution control
passes. Preserve all outcomes, including the lone bin scores; they cannot
establish a bin/crate risk comparison. See `baseline-summary.json` and the six
original native folders under `outputs/rounded-trash-contact/`.

The explicit closer follow-up uses the existing stationary pose-survey option
at **1.3 m** standoff and 640-pixel RGB for ALL three targets. This changes only
the authored observation pose, not mesh size, LiDAR, quality/correspondence
thresholds, model settings, clocks, priors or scores. Use the separate
`configs/restaurant_rounded_trash_close_contact.json` recipe and fresh outputs
under `outputs/rounded-trash-contact/close/`. It is a view-dependent diagnostic,
not repair of the preserved 1.85 m association failures.

At `e0821eb`, the closer bin passes all 18-source component checks with 16
accepted/contact-supported assessments and zero wheels. Median combined visual
cue/contact/phi is 0.126832/0.088028/0.583211. Four of those updates also use the
existing exposed-edge specialist, raising contact to about 0.204--0.210 with
lower confidence; these updates remain included. Do not describe it as uniformly
low risk or silently suppress specialist evidence. Both crate components fail
unique target observation at this closer view. Therefore there is not yet a
matched bin/crate comparison; successful different-distance captures are not
substitutes. Preserve all three closer captures and their failures.

A final bounded shared-view probe uses **1.6 m**, still 640-pixel RGB, the same
three original assets and every unchanged estimator/association/CRA setting.
Use `configs/restaurant_rounded_trash_mid_contact.json`, fresh outputs under
`outputs/rounded-trash-contact/mid/`. This pose is chosen to test common
observability between the two earlier distances, not by optimizing contact
scores. At `70d0bcb`, bin and intact crate each pass with 16 target-matched,
contact-supported updates. Broken crate fails the unchanged target match:
11 runtime image-group assessments have only 20--26 returns on two fragments,
with observed centroids 0.304--0.308 m from the fixture center (limit 0.20 m).
Its broad RGB mask is visible, but the sampled geometry is biased toward an
edge. There are also renewed hypotheses and late ambiguous temporal matches.
These are not 11 accepted whole-target observations. The grouping records and
their scores are retained separately as unmatched rows, rather than erased or
counted as target successes. No matching/ownership threshold is changed.

## Complete native outcomes

Each row below contains 18 fresh sources with zero wheel commands. Contact and
phi medians cover every target-matched assessment, including the early lower-
confidence scores; they are not selected steady-state or best-frame scores.
"Other groups" are runtime image-group assessments that fail the independent
fixture-center match. They do not establish successful target observation.

| Distance / RGB | Asset | Component | Target matches | Other groups | Median contact | Median phi |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1.85 m / 640 | Rounded bin | FAIL | 1 | 0 | .143897 | .608745 |
| 1.85 m / 1920 | Rounded bin | FAIL | 1 | 0 | .148919 | .611040 |
| 1.85 m / 640 | Intact crate | PASS | 16 | 0 | .105372 | .591142 |
| 1.85 m / 1920 | Intact crate | PASS | 16 | 0 | .119002 | .597374 |
| 1.85 m / 640 | Broken crate | PASS | 16 | 0 | .107604 | .592162 |
| 1.85 m / 1920 | Broken crate | FAIL | 0 | 0 | -- | -- |
| 1.3 m / 640 | Rounded bin | PASS | 16 | 0 | .088028 | .583211 |
| 1.3 m / 640 | Intact crate | FAIL | 0 | 16 | -- | -- |
| 1.3 m / 640 | Broken crate | FAIL | 0 | 0 | -- | -- |
| 1.6 m / 640 | Rounded bin | PASS | 16 | 0 | .084192 | .581458 |
| 1.6 m / 640 | Intact crate | PASS | 16 | 0 | .099793 | .588591 |
| 1.6 m / 640 | Broken crate | FAIL | 0 | 11 | -- | -- |

At 1.6 m the bin's raw combined contact cue is .046978--.140245 (median .092558),
CRA contact is .079035--.145693, phi is .579101--.609566 and uncalibrated visual
mass is .409979--.509806 kg. No exposed-edge specialist joins occurred in these
16 bin updates. At 1.3 m, four joins produced the larger contact range
.084254--.209768. An archived-mask outline-only replay gives .048474--.128668
for that closer bin; it explains the difference from the combined runtime cue
but does not replace the runtime scores or prove those specialist responses
are false. See `close/bin-contact-source-audit.json`.

The unmatched broken-crate groups at 1.6 m have median contact/phi
.112774/.594525. These diagnostic numbers suggest the intended direction but
are NOT a valid accepted bin/broken-crate comparison. The successful 1.85 m
crate pair has only a small median contact contrast, with overlapping ranges.
The intact crate's 1.3 m unmatched group centers are .298--.302 m from truth;
its failure similarly must not be presented as a missing RGB silhouette.

All 12 launches retain their recorded initial sensor synchronization rejections
(also present in passing runs). All three 1920 runs additionally record a pose-
history rejection. Every native `source_unchanged` check passes; source hashes,
full records, failed reports, masks and videos remain in their original folders.
Median field-build times are about 1.85--1.95 s at 640 and 2.49--2.67 s at 1920.
This is paced stationary inference, not real-time navigation; the final set also
overlapped the unit regression and is not a controlled performance benchmark.

## Validation and remaining scope

- Full unit suite: **3,767 passed in 169.79 s**. Focused asset/contact/image-object/
  runner regression: **68 passed in 29.26 s**. Ruff across `robot scripts tests`
  and working-tree `git diff --check` pass. The complete branch diff passes
  `git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol
  diff --check b3f9400..HEAD`: the plain branch check flags CRLF in the new
  generated asset manifest, whose bytes remain frozen as used by native tests.
  This is a command-local Windows line-ending allowance, not a Git configuration
  change or suppression of other trailing whitespace.
- Final SHA audits preserve all **106** accepted-v2 manifest files, all **49**
  pre-existing tracked asset files, and all three new original source files.
  Runtime sources still match every native invocation. See
  `outputs/rounded-trash-contact/preservation-final.json`.
- Final reporting is in `summary.json`, `close/summary.json`, `mid/summary.json`
  and their `comparison.png` images. Unmatched group rows are explicit; the
  original baseline reports and every failure are retained.
- Bin versus intact crate at 1.6 m supports a small view-specific contact-proxy
  contrast. It does not establish calibrated sharpness, material hardness,
  uniformly low risk, complete fresh four-factor estimation or navigation.
  Potential/thermal retain labeled .25 priors; mass and partial visible extent
  are uncalibrated, and ego reference is simulator-supplied.

The rounded bin is a reasonable asset candidate. A robust paired demonstration
with the broken crate remains unvalidated: the next bounded work is to diagnose
partial-surface geometry/whole-object association, including negative controls,
without widening a gate simply to pass this fixture. No more view search, motion
trial or accepted-demo integration is part of this branch. Keep default-OFF
development scope, draft status and all main/primary/PR117 holds.
