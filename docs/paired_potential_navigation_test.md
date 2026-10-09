# Paired ground/raised sphere CRA navigation test

This extends the potential-only diagnostic in `paired_factor_object_tests.md`.
The same radius-0.18 m matte sphere is assessed independently on the floor and
on a 0.50 m square, 0.30 m high solid support. All four factor inputs come from
the existing estimators and explicitly labeled priors. No factor scores are
supplied, copied between trials, equalized, or replaced with zero.

## Evidence and controller contracts

- Actual Hawk RGB and Pandar LiDAR are captured stationary. Robot ego pose and
  calibration/extrinsics use simulation USD references; target XY is an
  explicit fixture-selection prior. These are bounded prior-assisted trials.
- YOLO is invoked, but missed both glossy raised-ball sizes. An explicit opt-in
  class-independent RGB baseline discovers a separated chromatic connected
  component from projected target-track returns. Circular overlapping hue bands
  avoid merging different colors. It uses no authored color/shape/height/mask,
  retains class `unknown`, and uses heuristic confidence 0.5. This is not a new
  trained general object detector. Gray, multicolor, ambiguous and touching
  bounded chromatic neighbors remain unsupported. Border-connected background
  is distinguished from a bounded neighboring component. Only interior gaps
  smaller than 1% of the component can be filled as external-outline inference.
- Geometry selection uses the actual RGB instance mask with zero pixel margin;
  semantic association retains its separate six-pixel tolerance. The unchanged
  sphere residual/conditioning/coverage gates determine whether volume is usable.
  Mass uses the same explicit effective-density PRIOR: 25 [5, 100] kg/m³,
  confidence 0.35. Internal contents and density are not measured.
- Raised fall height comes from observed support and candidate landing bands,
  using the previously documented experimental support margin 0.30 m, maximum
  support gap 0.20 m, overhang 0.05 m and landing range 2.0 m. Other planarity,
  ambiguity, continuation and uncertainty checks stay unchanged. A fitted sphere
  bottom only checks support consistency; it never supplies fall height.
  The farther landing band is not an exact impact location or fall trajectory.
- Ground support uses the existing observed floor-plane fit plus bounded local
  flat-floor continuity PRIOR. Both trials use separately labeled worst-case
  harmful-fall interaction with upper bound one. Route overlap is not a fall
  interaction substitute.
- Kinetic uses tracked physical motion and nominal robot-relative approach;
  a stationary object still has relative approach speed near 0.20 m/s. Contact
  is the visible silhouette geometry proxy, not physical sharpness/hardness.
  Thermal is a validated stationary no-visible-plume RGB observation, not coldness.
- Instance geometry is separated from occupancy. Every remaining detected
  obstacle return, including support and environment, remains in a separate
  unknown-risk occupancy group with unchanged CRA fallback phi=1. Overlapping
  configuration cells conservatively arbitrate risk; this can dominate the
  ball's computed phi. Per-yaw risk-cell counts expose this effect.
- The existing dense-scene guidance arbitration is explicitly enabled in this
  mode; it preserves occupancy/h_full constraints. The waypoint nominal remains
  obstacle-unaware. Online MPC and analytical CBF produce physical wheel commands,
  with the existing unicycle adaptation and fixed lambda=1.25.
- Stationary four-factor PRIOR readiness, field construction and initial MPC
  prediction must pass before wheels. The full robot footprint is independently
  scored against the ball and a disk circumscribing the entire support. This
  conservative oracle monitor is never passed to the controller.
- All estimates/fields are frozen during each static-object pass, h_t=0. Both
  wheel trials have the same 40 s simulated duration bound, waypoint (2.5, 0),
  controller settings and positive-clearance stopping check. No real-time or
  risk-to-clearance monotonicity claim follows from these trials.
- Fixture/material edits use the USD session layer; saved scene hashes are
  checked. No restaurant assets, dependencies, weights or datasets are changed.

## Stationary results

Actual independent reports: `outputs/raised_pair/ground_v2/report.json` and
`outputs/raised_pair/raised_narrow/report.json`. Scores below are the adapter's
interval-aware nominal risks before CRA's additional confidence uplift.

| Quantity | Ground | Raised |
|---|---:|---:|
| Estimated support-to-landing drop, m | 0 (continuity PRIOR) | 0.300858 (INFERRED) |
| Rough mass, kg (PRIOR) | 0.522902 | 0.541045 |
| Potential | 0.023573 | 0.127061 |
| Kinetic | 0.001901 | 0.001970 |
| Contact outline proxy | 0.253370 | 0.278407 |
| Thermal visible-plume proxy | 0.013125 | 0.013125 |
| Conservative aggregate phi | 0.978008 | 0.988005 |
| All-four PRIOR readiness | pass | pass |

The comparison script declares matching tolerances of 0.10 absolute difference
for each nonpotential nominal factor and 25% relative rough-mass difference.
These are bounded matching tolerances, not calibrated accuracy claims. Actual
mass difference is approximately 3.35%; other factor differences are about
0.000069 kinetic, 0.025037 contact and zero thermal. Both aggregate values remain
high because the unchanged uncertainty policy conservatively uplifts weak evidence;
the ball has no unknown-factor fallback in these accepted reports.

## Physical wheel results

Both final independent wheel reports passed all four PRIOR readiness gates,
waypoint completion, full-footprint clearance and the subsequent 0.5 s stopping
check. No scene file changed. Both passed on the left despite soft RIGHT guidance.

| Quantity | Ground | Raised + support |
|---|---:|---:|
| Potential nominal risk | 0.023652 | 0.126172 |
| Conservative phi used as object policy | 0.977990 | 0.987753 |
| Goal completion, simulated s | 28.2 | 35.4 |
| Minimum full-footprint clearance, m | 0.166207 | 0.133492 |
| Minimum MPC margin | 0.039852 | 0.008938 |
| Minimum analytical CBF margin | 0.281752 | 0.179795 |
| Mean / maximum MPC wall time, s | 0.219 / 0.598 | 0.278 / 1.266 |
| Stopping / saved-scene preservation | pass | pass |

The raised wheel capture inferred drop 0.300842 [0.220842, 0.380842] m. Its
rough-mass difference from the ground capture was 3.86%; nonpotential nominal
factor differences were 0.000064 kinetic, 0.021571 contact and zero thermal.
These are repeat captures rather than reused stationary factor values.
The physical raised field still had support-dominated ball occupancy, so its
completion does not establish navigation sensitivity to the ball's phi.
The separate matched measured-geometry replay using these wheel-capture policies
passed, with maximum absolute h_full difference 0.00263137.

Local artifacts: `outputs/raised_pair/motion_ground/report.json`,
`motion_raised/report.json`, `motion-summary.json` and `comparison.png`.
Report SHA256 ground:
`aebef0c34c6c82a00b375b3ac5c2a2b7d24f2358f3f7354247f6f6d3e9adce7d`;
raised: `6b707e5db832d443f07df03c0a8c1519716e582dc8027afe677861578c4a00e6`.
The exercised demo source Git blob is `22c1b21d7c9da82070a3c1e2654697ec20d00124`.
Wall times are diagnostic observations, not real-time performance validation.

## Preserved failures and scope

`outputs/raised_pair/` retains all attempts rather than overwriting them:

- `raised_initial` / `raised_r025`: class-trained YOLO missed the glossy raised
  balls; an unrelated chair mask failed overlap. No wheels were enabled.
- `raised_chromatic`: saturation alone connected differently colored regions;
  this motivated circular hue-separated components rather than a supplied color.
- `raised_chromatic_v2`: specular interior gap exceeded the 1% outline allowance.
  The matched test fixture uses matte appearance; the gap gate was not loosened.
- `raised_matte`: radius 0.25 m left insufficient visible air above the object;
  thermal stayed unavailable and aggregate phi remained 1.
- `raised_small`: six-pixel geometry dilation admitted platform returns and
  failed the unchanged sphere residual gate. Offline replay of the same cloud
  with the exact observed mask passed that gate. The 0.70 m support also lacked
  a sufficiently exposed landing direction under the continuation check.
- `ground`: the initial neighbor rule treated the border-connected floor as a
  competing object. The corrected rule retains rejection of bounded chromatic
  neighbors, with a dedicated floor-background regression test.

General detector robustness, moving-camera plume estimation, exact mass,
physical contact consequence and continuous online field refresh remain outside
this test. The raised support changes occupancy, so path differences cannot be
attributed solely to the ball's potential factor or phi.

The final stationary field diagnostic (`raised_final`) makes this limitation
concrete: all 16 yaw slices had zero occupied cells at the ball's phi, because
the support's unknown phi=1 dominated overlapping configuration occupancy.
This is preserved conservative behavior, not a validated platform risk estimate.
The ground wheel capture retained 37–43 cells per yaw at its ball phi alongside
the conservative environmental occupancy.

An explicitly separate offline matched-geometry policy replay uses exactly the
same measured ground-ball points with each independently computed observed
policy. It excludes support/environment and never commands wheels. The actual
stationary policies phi=0.978008 / 0.988069 gave flux magnitudes 1.673610 /
1.685682 and maximum absolute h_full difference 0.00266119. This demonstrates
computed-policy propagation to the safety field, without establishing physical
raised-platform safety or a potential-only causal navigation effect. Confidence
and the other estimated factors also vary slightly between observations.

## Reproduction and validation

Use the existing Isaac launcher and locally configured weights as in
`paired_factor_object_tests.md`, adding:

```text
--object-radius 0.18 --object-plume-policy
--sphere-density-prior-kg-m3 25 5 100 --ground-continuity-prior
--require-four-factor-readiness --support-width-m 0.5
--instance-mask-geometry --chromatic-instance-proposal
--worst-case-fall-interaction-prior --motion-maximum-seconds 40
```

Ground uses default `--support-drop-m 0`; raised adds `--support-drop-m 0.30`.
Run both stationary first into fresh output directories; then use `--motion`
in fresh directories. The script repeats the complete gate before commanding
wheels. Compare actual reports with:

```text
python scripts/evaluation/cra_potential_pair_report.py
  --ground outputs/raised_pair/ground_v2/report.json
  --raised outputs/raised_pair/raised_narrow/report.json
  --output outputs/raised_pair/stationary-summary.json
```

Add `--matched-ground-capture outputs/raised_pair/ground_v2/capture.npz` for the
separately labeled measured-geometry policy replay. Add `--require-motion` and
use the two actual motion report paths to require both physical wheel passes.

All 1,476 unit tests passed in 34.30 s using the established Isaac/Conda import
configuration (`outputs/raised_pair/unit-tests-final-v2.log`). New tests cover
mask-return alignment/complement accounting, registration margins, bounded RGB
proposal failures, floor-background distinction, specular-gap rejection, and
unchanged conservative risk propagation for retained support. Repository Ruff,
demo AST parsing and whitespace checks passed. Outputs and model weights remain
ignored local artifacts.
