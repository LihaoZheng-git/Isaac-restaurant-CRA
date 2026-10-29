# Computed human CRA and bounded track recovery

The latest observed-motion trial failed arrival after a missing human LiDAR
track. When positive geometry returned, its centroid residual was only
0.001143 m, but the uncommitted geometry timestamp made the original 0.25 s
continuity gate fail at gaps of 0.3, 0.4 and 0.5 s. Preserve the failed report
`outputs/observed_motion_reference/motion-v4/report.json` (SHA256
`13a6ff2ce7acd46ca43009cde7f9a8502ceb0aff59c80408b898f82226cebb24`).

The user approved computed CRA as the human policy and bounded recovery.
The default physical estimator and strict continuity path remain unchanged.
`--human-track-recovery` explicitly enables:

- At most 0.6 s between original positive LiDAR sources. Reacquisition requires
  actual current returns, unchanged 0.25 m prediction-residual and competitor
  gates, and the original 30 s visual identity lease. Same-ID observation counts
  must increase. A new ID needs unique one-to-one positive geometric matching;
  mass/contact/support/plume properties are not transferred to that new ID.
- An explicit 1 m/s separated-walking-human speed prior. No actor truth enters
  control. This bound is not a validated general pedestrian motion guarantee.
- A missing identity can be temporarily excluded only when the entire human
  reachable disk cannot meet the robot reachable disk over 1.55 s (full 1.2 s
  MPC horizon plus 0.35 s field-age limit). The disk includes the unchanged
  body/centroid prior and the actual circumscribed robot footprint with margin.
  This is rechecked every field, expires after 0.6 s and never deletes the
  identity, observes free space, or refreshes its timestamps. Nearby missing
  geometry still rejects the field. No mass or passing cue is emitted for an
  excluded missing identity. All raw/unmatched occupancy remains.

This is a recovery/short-gap policy, not general occlusion tracking, permanent
retirement, future-human prediction or a guarantee of uninterrupted motion.
The unchanged nearby-fragment guard and exact geometry/CBF checks still apply.

The previous command's `--human-boundary-flux -1.7` forced the verified human's
applied field risk to 1. The matched computed-CRA variant omits that override.
Current human-part association now accepts a verified computed parent policy,
while retaining any higher child risk. It no longer requires fixed human flux.
The body-envelope unknown-factor floor remains explicit (default phi
0.68359375); this change neither reduces factor priors nor declares heat absent.
Fixed flux stays available as a separately labeled comparison. Guidance tracking
is still an ablation; this experiment is not the primary controller comparison.

Validation proceeds through focused/complete unit tests and Ruff, a fresh
zero-wheel walking-person component, independent stationary validation, then
the original 60-frame gate and matched navigation. Use the accepted no-advisory
`motion-temporal-jacobian-v4` invocation, remove only fixed human flux and add
bounded recovery. Keep body radius 0.65 m, physical margin 0.05 m, lambda 1.25,
six 0.2 s intervals, robot limits, QP1000/rescue20, original clocks and exact
acceptance. No quiet-person preparation or failed predictive/advisory option.
Native results are pending; unit recovery is not a smoothing result. Arrival,
nearby bend, independent clearance/containment, stopping and the original
0.5 s/50% hold-reduction target remain mandatory. Keep all release holds.
