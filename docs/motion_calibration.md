# LiDAR motion calibration

The stationary camera/LiDAR diagnostic now collects 12 strictly newer LiDAR
frames and runs a parallel tracker with its speed gate disabled. Calibration
uses persistent clusters with at least eight points and four observations; the
threshold recommendation is the configured stationary-speed quantile plus a
0.02 m/s margin. These selection and margin assumptions are configurable.

The current stationary run produced 143 accepted raw samples: median 0.0181
m/s, p95 0.1026 m/s, and maximum 0.2260 m/s. The p95-plus-margin recommendation
was 0.1226 m/s, below the configured 0.15 m/s gate. Isolated larger centroid
jumps remain present, so the runtime now requires three consecutive
direction-consistent motion updates. That combination published zero nonzero
motions across the 143 accepted stationary samples.

This is a stationary noise calibration, not moving-object accuracy evidence.
The three-update gate adds detection latency, and the next controlled moving
obstacle experiment must measure velocity error, time to first confirmed
motion, route-crossing response, and missed slow motion before wheel tests.

The controlled stationary-robot moving-proxy diagnostic now provides positive
motion evidence for the existing tracker. A session-only proxy moved at an
authored 0.60 m/s while Nova Carter received zero wheel commands. Across 48
live PandarXT-32 velocity estimates, the median estimated speed was 0.552 m/s
and the median vector error was 0.061 m/s. Motion was first confirmed after
0.317 s, the proxy retained one track ID, and the published velocity returned
to 0.0 m/s after the proxy stopped. Robot position drift was 0.0075 m and yaw
drift was 0.0093 degrees, within the diagnostic's 0.01 m and 0.1 degree gates.

The proxy pose and 0.60 m/s trajectory are oracle inputs used only to select
the evaluation track and score its inferred LiDAR velocity. They are never
passed into `ObstacleTracker`, CRA, CBF, MPC, or wheel control. The diagnostic
writes `outputs/lidar_motion_isaac/report.json` and labels the estimate source
as `lidar_centroid_tracking`. The current result validates clear lateral
motion in one controlled geometry. It does not validate missed slow motion,
occlusion, multiple crossing objects, or transfer to physical hardware; those
remain separate evaluations before wheel-motion experiments.
