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
