# Category-free camera–LiDAR CRA track

## SAM instance masks (first bounded step)

`robot/perception/sam_instance_segmentation.py` refines an existing LiDAR track
with an RGB mask. The positive prompt is an observed projected return nearest
the track's image median. No class, text, color, authored mask, density or object
dimensions are supplied to SAM. Discovery and occupancy remain LiDAR-based.
The output class is `unknown`; this means category unknown, not association
unavailable. Rejected proposals remain unassessed occupancy.

The adapter reuses installed Ultralytics 8.4.104 (`SAM2Predictor`) with Meta's
official SAM 2.1 Hiera Tiny weights. No package installation or environment
upgrade was performed. Download:
<https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt>.
The local filename `sam2.1_t.pt` selects the corresponding installed builder.
SHA-256: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`;
156,008,466 bytes; 471 tensor state entries, checked before model loading.
Official [SAM 2 repository](https://github.com/facebookresearch/sam2) and
[license](https://github.com/facebookresearch/sam2/blob/main/LICENSE) identify
Apache-2.0 for the weights. The reused Ultralytics implementation retains its
own AGPL-3.0 licensing. Neither weights nor upstream source are committed here.

Configurable gates reject too few projected points, low model quality, masks
touching the image boundary, unreasonable mask area, poor LiDAR coverage,
competing-track contamination, and similarly scored non-equivalent masks.
Equivalent decoder alternatives are allowed. The default 0.75 evidence
confidence cap is a heuristic, **not** calibrated physical confidence; SAM's
predicted mask quality is recorded separately.

The simple-object demo requires explicit `--instance-mask-geometry
--sam-instance-proposal` and an existing `SAM2_CHECKPOINT`. SAM is applied to
each distinct camera image. Exact-mask return selection keeps the complement
as conservative occupancy; no mask removes an obstacle from safety processing.
The default mass class gate is unchanged in this first step.

### Stationary validation

Fresh headless Isaac capture in
`outputs/sam2_integration/segmentation_stationary/` passed stationary sensor and
CRA mapping checks, without wheel motion. SAM accepted 186 exact-mask LiDAR
returns, with model quality 0.988, encoder time 0.100 s and decoder time 0.093 s
for the recorded frame. The output remains category unknown. Raw YOLO actually
recognized this glossy ground sphere as `sports_ball` (0.867); this capture is
**not a demonstrated YOLO miss**. Its mass remained unavailable and aggregate
phi remained 1, as expected before the category-free mass step.

The visible LiDAR envelope was approximately 0.247 × 0.085 × 0.093 m for a
configured 0.36 m diameter fixture: sparse visible surface extents are not a
complete reconstruction. Fixture dimensions never enter the estimator. Target
track selection and simulator calibration/extrinsics remain explicitly bounded
demo assumptions. Timing is a recorded diagnostic, not a real-time guarantee.

Validation: 39 focused segmentation/association tests; **1,508 full unit tests**;
repository-wide Ruff passed. Unit tests cover ambiguity, boundary/area/quality
gates, competing tracks, coverage, source-image shape and checkpoint hash.

## Remaining bounded steps

1. Explicit quality-gated category-free RGB-plus-LiDAR mass inference, retaining
   the legacy class gate by default and uncertainty on domain extrapolation.
2. Stationary combined four-factor tests, including actual detector misses and
   rejected/ambiguous inputs; no density prerequisite or invented factor zeros.
3. Only after readiness: unchanged computed CRA phi in physical pass-by trials,
   with fixed lambda 1.25, obstacle-unaware nominal and online MPC/CBF safety.

This track does not claim universal segmentation, general mass accuracy,
physical sharpness, temperature measurement or universal navigation safety.
