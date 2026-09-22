# Stationary sensor gate

Run the camera-only diagnostic before camera--LiDAR fusion. Both scripts hold
the drive wheels at zero and stop before Safe-SAGE, CBF, MPC, or navigation.

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
