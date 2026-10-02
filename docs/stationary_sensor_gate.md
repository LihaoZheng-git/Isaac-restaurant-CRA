# Stationary sensor gate

Run the camera-only diagnostic before camera--LiDAR fusion. Both scripts hold
the drive wheels at zero. Fusion may report CRA policy parameters, but they do
not enter the Safe-SAGE field, CBF, MPC, or navigation.

The generated scene references the official Isaac Sim 6.0 Nova Carter asset.
The process therefore needs access to NVIDIA's asset server unless all USD
dependencies are already cached. `load_robot_scene` fails before controller
construction when the chassis, front Hawk camera, or Pandar LiDAR reference is
unresolved.

On Windows, redirect the Warp kernel cache when the default NVIDIA cache is
not writable:

```powershell
$env:ISAAC_SIM_HEADLESS = "1"
$env:YOLO_MODEL_PATH = "C:\path\to\yolo11n-seg.pt"
$env:WARP_CACHE_PATH = "$env:TEMP\cra-warp-cache"
$env:YOLO_AUTOCLOSE = "1"
python -u scripts/demos/yolo_semantic_perception_demo.py
```

After that report passes, run the fusion gate:

```powershell
$env:FUSION_AUTOCLOSE = "1"
python -u scripts/demos/camera_lidar_semantic_fusion_demo.py
```

A valid gate requires real RGB frames, at least one YOLO instance, real LiDAR
tracks, at least one camera--LiDAR association, acceptable sensor age, and a
stationary robot. Saved overlays are diagnostic evidence rather than ground
truth. No factor estimate is classified as sensor-derived until its required
measurements pass their own runtime checks.

## Repeated stationary benchmark

The runner uses an existing Python environment, requires explicit local model
paths, and refuses to overwrite an output directory. It places the catalog's
existing restaurant chair in front of the stopped robot in the session layer.
Authored bounds are saved separately for offline scoring and never enter the
mass or geometry estimators. Other objects retain their detected classes.

```powershell
python -u scripts/evaluation/run_stationary_perception_benchmark.py `
  --isaac-python C:\path\to\existing\python.exe `
  --object-checkpoint C:\path\to\yolo11n-seg.pt `
  --mass-checkpoint C:\path\to\visual-mass-best.pt `
  --contact-checkpoint C:\path\to\contact-best.pt `
  --output-dir outputs\stationary_perception_benchmark\fresh-run --headless
```

Each capture stores RGB, masks, aligned LiDAR returns, transforms, calibration,
estimates with numeric intervals and provenance, per-factor rejection reasons,
model/scene/source fingerprints, and scenario-scoped evidence records. The
Git revision plus source-file hashes identify runs made before commit. Zero
wheel commands reflect this script's stop-only command path; measured pose
drift independently checks stationarity. This is not a wheel-encoder study.

The default three runs are followed by controlled positive fall-height and
moving-object tests with the robot stopped. `summary.json` distinguishes sensor
dataflow from component accuracy. Missing artifacts or failed component checks
make the runner fail, even when Isaac itself exits with code zero. A valid
dataflow result never grants four-factor accuracy or motion readiness.

## 2026-10-02 result

Local artifacts: `outputs/stationary_perception_benchmark/20261002/summary.json`.

| Check | Observed result | Boundary |
| --- | --- | --- |
| Repeated RGB/LiDAR fusion | 3/3 captures passed; 2 chair-track associations per capture | One scene, pose and camera; rendered sensor inputs, not hardware transfer |
| Stationarity | Maximum drift 0.001444 m, yaw 0.002754 degrees; age 0.06667 s | 0.005 m / 0.1 degree / 0.25 s acceptance limits; zero published static-object motion in 139 samples per run |
| Visual mass | Inference ran in all 3 captures, confidence capped at 0.25 | No reference mass; RGB and partial LiDAR dimensions differ from the catalog training domain |
| Contact | Specialist ran in all 3 captures; no hazardous-part detections | Invocation is verified, positive contact perception and physical sharpness are not |
| Thermal | Not invoked | No verified local continuous-plume checkpoint; factor stays unknown |
| Corrected 0.80 m fall fixture | Failed: fall height unavailable | Support band too far below target; earlier positive result invalidated by fixture scaling error |
| Object motion | Passed: 0.566 m/s median for 0.60 m/s proxy; vector error 0.0567 m/s; first confirmation 0.283 s | 48 estimates, zero ID switches, zero speed after stop; slow/occluded/multiple-object motion unvalidated |

The motion test's robot drift was 0.007513 m under its separate 0.01 m / 0.1
degree limits. Its authored trajectory is used only for association/scoring.
It does not command the robot to move.

A separate 0.55 m support trial is retained at
`outputs/stationary_perception_benchmark/20261002-visible-support/report.json`.
It also failed to establish a planar supporting surface. Neither failure is
replaced with zero fall risk. The benchmark execution completed, restaurant
dataflow passed, but the overall component gate failed and motion readiness
remains false.

The next bounded perception work is to resolve support-plane visibility/band
selection using these saved clouds and negative controls, then validate
restaurant mass against independent reference masses. Positive contact and
continuous plume perception still need suitable test inputs and verified
weights before four-factor scenario experiments.
