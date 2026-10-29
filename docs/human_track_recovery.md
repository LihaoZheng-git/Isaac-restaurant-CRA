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
Unit recovery is not a smoothing result. Arrival,
nearby bend, independent clearance/containment, stopping and the original
0.5 s/50% hold-reduction target remain mandatory. Keep all release holds.

## Validation at f6ecfac

The fresh zero-wheel component passed 115 field updates, including human
velocity, current selection, body-envelope containment and original mass-prior
expiry. Its maximum inferred human speed was 0.7814 m/s, within the configured
1 m/s recovery scope; this is not a general motion-bound calibration. The
independent stationary and live run's own 60-frame gates both passed. Runtime
source hashes and the Git revision stayed unchanged throughout all three runs.

`outputs/human_track_recovery/motion-v1/report.json` passed native navigation:

| Metric | Previous fixed-flux run | Computed CRA + recovery option |
|---|---:|---:|
| Final arrival / route guides | Yes / 3 | Yes / 3 |
| Navigation simulation time | 21.2333 s | 19.6500 s |
| Longest actual nearby translation hold | 2.5000 s | 0.8833 s |
| Nearby lateral excursion | 0.2398 m | 0.2372 m |
| Minimum independent clearance | 0.4296 m | 0.4408 m |
| Final stopping drift | 0.002434 m | 0.002417 m |

There were 193 live fields, 184 while translating, 20 supported moving-kinetic
rows and a minimum analytical margin of 0.18733. Applied human risk equaled
computed CRA throughout (range 0.5580–0.6836), without the fixed-flux override.
The largest nearby curve occurred with the human 1.3603 m from the robot center.
Maximum pre-walk lateral excursion was only 0.000448 m; human walking began at
source 11.3000 s and the robot first exceeded a 2 cm bend at 12.6833 s.

The hold decreased 64.7% against the latest 2.50 s baseline, and 62.9% against
the original 2.3833 s target baseline. **The strict smoothness gate still fails
because 0.8833 s exceeds 0.5 s.** The remaining yield followed current-field
plan invalidation and unsuccessful exact solver/rescue checks, with four
stale-field command ticks in the actual hold interval. These solver failures
do not prove infeasibility. No tolerance, horizon, speed or physical margin was
relaxed. See `remaining-yield-diagnosis.json`; its time alignment matches ten
exact command pairs across the independent source and navigation clocks.

No recovery event or human continuity rejection occurred in this live run.
Consequently, do not attribute this navigation improvement to reacquisition.
The recovery branch was separately checked with unit cases and a logged-track
replay: two deliberately removed observations caused the legacy path to reject
the return at 0.3 s, while the new path recovered with the original visual
timestamp. That replay reconstructs tracking from native logged centroids,
velocities and counts; it is not a raw-sensor or natural-occlusion experiment.
Its deliberately near robot test position is diagnostic only. Permanent
retirement and sustained close occlusion remain unimplemented.

This remains a single prior-assisted, paced guidance-tracking experiment.
There were zero generic fresh-instance/complete four-factor updates during
motion; separate human identity/mass-prior and current kinetic evidence checks
must not be presented as fresh four-factor visual perception. Human mass,
blunt contact, unavailable potential/thermal and body geometry retain their
declared priors. No real-time, causal CRA superiority, universal safety or
exact-paper equivalence claim follows.

Final source checks: **3,200 unit tests passed in 113.97 s; Ruff, compilation
and `git diff --check` passed.** An initial sandbox test attempt could not create
its temporary directory; the full suite was rerun using a new repository-local
temporary directory. No environment changes or installs were needed.

Artifacts in `outputs/human_track_recovery/motion-v1/`:

- `navigation-only-v1.mp4`: original cadence; removes only the zero-command
  setup prefix and retains the yield, full navigation and final stop.
- `native-nearby-curve.png`: unedited native video frame at the largest nearby
  bend, with matched human/robot source time in its JSON companion.
- `recovery-comparison.json/png`: original acceptance, failed stricter
  smoothness gate and measured comparison.
- `trajectory-timing-v4.json/png`: actual same-time human/robot positions.

Native report SHA256:
`c7b8f0e8cf5197e51c9f4986bd16c4e7c19f3cf4bf7296336b4206a948e58d9e`.
Keep PR #198 draft and all primary/main/PI/long-service holds. The bounded
policy/dataflow change is validated; the sub-half-second smoothing goal remains
open and should next be investigated through the rejected MPC inputs.
