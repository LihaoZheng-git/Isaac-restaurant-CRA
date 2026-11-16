# General object height and support evidence

The opt-in general height path separates three quantities: visible elevation,
support elevation, and possible local drop. A tall person or object standing on
the floor does not receive its body/center height as fall height. A missing
landing observation no longer erases a successfully inferred support surface.
This component does not identify a person carrying a separate object.

`robot/perception/object_height.py` fits a local floor plane from independent
LiDAR returns in a declared metric, gravity-aligned frame. An explicit ground
reference selects the floor band, preventing a visible tabletop from silently
becoming the floor when the real floor is hidden. The fitted height comes from
returns. In Isaac, ego/gravity and ground-band selection use oracle references;
this is reference-assisted sensor inference, not oracle-free floor recognition.

The output includes visible lower-boundary and envelope-center heights above
the fitted plane, observed adjacent support elevation, a support hypothesis
(`ground_supported`, `elevated_supported`, or `uncertain`), bounds, confidence,
source times, floor residual/count/distance, support candidates and rejections.
Visible lower/center positions are not hidden object bottoms or physical COMs.
The existing support-overhang/planarity/gap checks remain; competing plausible
support levels are rejected by the new general path. The legacy drop estimator
keeps its original candidate selection and defaults.

Possible drop still requires the separate strict support/landing geometry gate.
An observed ground-support hypothesis with incomplete landing coverage can
select a **ground-supported baseline prior** (default potential risk .25).
Insufficient support uses the separately labeled **uncertainty prior**. These
labels distinguish why the same policy score was used; neither supplies a
missing physical fall height or changes the current scores, confidence fusion,
energy scales, noisy-OR aggregation or fixed lambda 1.25. Valid physical drop
and mass/interaction inputs continue through the existing potential adapter.
This height-only step does not resolve the known confidence/prior risk-ordering
calibration question in the final experiment plan.

## Runtime integration

`LiveObjectPerception(..., object_height_config=ObjectHeightConfig(...))` applies
the new path to fresh accepted class-free instances. It uses the exact accepted
mask's same-epoch LiDAR returns, updates the visible center height and support
state, and supplies strict drop evidence to the existing physical adapter.
The persistent multi-object policy records the baseline/uncertainty distinction
under `factor_scores.potential.height_context`. It never turns an unowned
LiDAR cluster into a visually assessed object, processes deferred vision as
fresh, retains moving-human height/support across gaps, or removes occupancy.
Unassessed tracks retain the existing numeric factor priors.

The live demo exposes `--object-height-evidence`, `--height-support-margin`
(default .12 m) and `--height-support-gap` (default .06 m). These opt-in settings
are saved with the run and per-instance height diagnostics. It uses a separately
named worst-case fall-interaction PRIOR when an actual drop is available; this
is not measured fall probability. General support and legacy spherical
bottom/continuity modes cannot be mixed implicitly. Omitting the opt-in retains
the legacy gates and runtime behavior. New-mode wheel navigation has not been
validated by this change.

## Executed stationary component diagnostics

The existing Isaac LiDAR diagnostic now has explicit general-height mode:
`CRA_OBJECT_HEIGHT_DIAGNOSTIC=1`. Its old positive-drop acceptance remains
separate as `strict_positive_fall_report`. Zero support elevation is permitted
only in general-height mode. Every wheel command is zero; source scene bytes
and fixture authoring are audited. Target location/height selection is configured
fixture information, not autonomous instance discovery. No authored support or
target height is passed as an estimated scalar.

| Actual trial | Height/support gate | Inferred support above ground | Strict local drop | Position drift |
| --- | --- | --- | --- | --- |
| `ground-live-v2` | Failed: sparse adjacent support | unavailable; visible elevation available | unavailable | .001215 m |
| `ground-live-v3` | Passed: ground-supported hypothesis | 0 [.000, .077] m | unavailable; landing coverage insufficient | .001215 m |
| `raised-live-v1` | Passed: elevated-supported | .299819 [.205643, .393994] m | .300817 [.220817, .380817] m against .300 m fixture truth | .001215 m |

Ground-v3 and raised-v1 use the same explicit .80 m support XY search margin,
.20 m maximum visible-lower-boundary gap and .05 m support overhang. Those are
controlled sparse-scan settings, **not new runtime defaults**. The wider
neighborhood supplies additional actual planar returns; no point-count,
planarity, gap, confidence, residual or interval acceptance check was loosened.
The ground case had 113 support/1,598 floor returns; raised had 965 support/734
floor returns. The observed lower-boundary gaps were .132/.104 m, so support is
an inferred relationship under the declared gap model, not measured contact.
Very small unsupported clearances can be indistinguishable under a large gap
allowance. Hidden bases, crowds and irregular/moving support need more evidence.

`ground-live-v1` failed during startup because Kit's legacy torchvision bundle
was imported after SimulationApp. Loading the existing conda perception modules
before Kit, as the live demo already does, fixed startup without changing any
packages. The original failed logs/reports are preserved. Accepted runs still
record the existing renderer/motion-history and attribute warnings. The launcher
can return zero on script failure; saved report flags and logs determine results.

Run each case separately using the existing environment (never simultaneous
Isaac instances), with the following environment values and a fresh output:

```powershell
$env:PYTHONEXE='"C:/Users/tayli/anaconda3/envs/env_60_isaacsim_yolo/python.exe"'
$env:ISAAC_SIM_HEADLESS='1'
$env:CRA_OBJECT_HEIGHT_DIAGNOSTIC='1'
$env:CRA_FALL_ISOLATED_SCENE='1'
$env:CRA_FALL_SUPPORT_XY_MARGIN_M='0.8'
$env:CRA_FALL_SUPPORT_DROP_M='0' # use '0.30' for the matched raised case
$env:CRA_LIDAR_DIAGNOSTIC_DIR='C:/myworkplace/code/Isaac-restaurant-CRA/outputs/object_height/ground-live-v3'
& C:/isaacsim/python.bat -u scripts/demos/lidar_fall_height_isaac_diagnostic.py
```

`scripts/evaluation/object_height_report.py --trial ... --trial ... --output ...`
replays the original copied scans with their original configuration and source
time, checks independent scoring/interval/stationary gates, hashes its inputs
and produces `stationary-height.png` and `report.json`. The actual two-trial
replay passed under `outputs/object_height/stationary-comparison-final/`.
The sparse-support replay under `rejected-sparse-replay-v1/` also correctly
remained failed and saved its unavailable result, rather than dropping the case.
Its .25 prior-label illustration deliberately leaves mass/interaction unavailable;
it is not a four-factor or potential-risk ordering experiment.

The final full unit regression passed **1,938 tests in 56.37 s**
(`outputs/object_height/unit-regression-v2.log`); repository Ruff and whitespace
checks passed. Unit coverage includes tall grounded objects, raised support, missing landing,
hidden lower portions, tabletop-only views, self-support contamination, sloped
and poorly sampled floors, competing support levels, stale/future epochs,
coordinate changes, invalid references/configuration, fresh-instance runtime
dataflow and unchanged prior/occupancy handling. Moving-human support and separate carried-bowl
ownership/release/landing evidence remain the next discussion, not completed
perception capabilities.
