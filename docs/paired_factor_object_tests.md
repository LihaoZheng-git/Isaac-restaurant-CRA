# Paired physical-factor object tests

These tests extend the [four-factor sphere gate](four_factor_simple_object_readiness.md)
before restaurant enrichment. Factor values and CRA phi are computed from the
estimator outputs, never supplied as experimental profiles. Lambda stays 1.25;
waypoint nominal remains obstacle-unaware. All motion requires stationary
four-factor readiness first. Mass and unseen ground continuity remain explicit
PRIORS, so these are prior-assisted demonstrations, not sensor-only results.

## Thermal comparison

Branch: `codex/cra-thermal-paired-test`. Run the existing simple-object demo with
`--object-radius 0.25 --object-plume-policy --sphere-density-prior-kg-m3 25 5 100
--ground-continuity-prior --require-four-factor-readiness`, changing only
`--plume-control none|intermittent|persistent` and the output directory. Existing
local YOLO, contact and mass weights/environment are unchanged. Add `--motion`
only after the stationary gates pass.

The same actual Hawk RGB object detector and Pandar LiDAR association identify
the sphere. A clean initial RGB reference precedes three fresh stationary
observations. Session-layer puffs change the rendered RGB; they never supply a
thermal score, mask, temperature or class label. They are opaque stylized visual
proxies, not a fluid/steam simulation. LiDAR occupancy is captured before the
puffs appear. Source/air-region association and the unchanged RGB plume backend
compute continuous strength, extent and persistence.

The sphere isolates the thermal intervention from changes in body geometry.
The separate bowl diagnostic retains configured source-region and binding
PRIORS; it is a component test, not a robot detector/navigation result. No bowl
score is transferred to the sphere.

All three stationary conditions passed four-factor PRIOR readiness; sensor-only
readiness remained false. No unknown-factor fallback was used.

| Rendered control | Inferred plume cue | Computed conservative phi |
|---|---:|---:|
| None | 0.000000 | 0.970619 |
| Intermittent, revised | 0.370235 | 0.980808 |
| Persistent | 0.417656 | 0.982479 |

The revised intermittent window still contained weak foreground residuals in
its nominally hidden middle frame (all three frames were positive). Its score
ordering therefore validates combined rendered area/extent/temporal evidence,
not clean on/off persistence timing. RGB timestamps are simulation read times;
the wrapper does not expose independent render-frame acquisition timestamps.
Six physics/render updates separate reads, but hardware-level freshness is not
established. This is a simulation diagnostic limitation.

The separate bowl component passed: cues 0.000000 (none), 0.352861
(intermittent), 0.762454 (dense) and 0.000000 (unrelated region). Unvalidated
reference and moving-camera controls remained unavailable, as required.

Physical-wheel endpoint trials, each freshly assessed and gated:

| Control | Computed phi | Goal time | Minimum footprint clearance | Minimum MPC / CBF margin | Stop |
|---|---:|---:|---:|---:|---|
| No plume | 0.970792 | 29.0 s | 0.20137 m | 0.02293 / 0.26483 | Passed |
| Persistent plume | 0.983087 | 27.6 s | 0.20402 m | 0.02701 / 0.27161 | Passed |

Both completed the waypoint and stopped with positive independent disk-to-full-
robot-footprint clearance. Both actually passed on the left despite soft right-
side guidance. Mean/max MPC wall times were 0.2094/0.7882 s and 0.1986/0.7947 s
against a 0.20 s planning interval. This is not a real-time deadline result.
Scene files were unchanged. The field/assessment is frozen during each static
pass; this does not validate moving-camera plume tracking.

The first intermittent fixture used counts `(1, 0, 1)` and returned cue 0 with
valid observation gates. Its low-contrast small puffs were missed against the
bright window background. This negative result is preserved in
`outputs/thermal_pair/intermittent`. The revised, more resolved fixture uses
`(3, 0, 3)`; the persistent fixture uses `(4, 4, 4)`. No estimator threshold,
confidence, uncertainty uplift, aggregation or controller parameter was changed.
This demonstrates a visibility/contrast limitation: a low cue is not proof that
plume or heat is absent.

The small endpoint clearance difference is one run per condition, with small
sensor/fit variation. It does not establish a causal or monotonic phi-to-clearance
relationship. Factor-specific diagnostics expose the plume response even while
uncertainty keeps aggregate phi high. Physical temperature, aerosol identity,
spill severity and carried state remain unavailable.

Validation: 1,441 unit tests passed; full Ruff check, whitespace and demo syntax
checks passed. Outputs, raw RGB windows, captures, trajectories and model hashes
remain local under `outputs/thermal_pair/`; they are not committed datasets.

Report SHA-256 values are retained in `outputs/thermal_pair/summary.json`.
The actual RGB/results figure is `outputs/thermal_pair/comparison.png`.
Endpoint report SHA-256: no plume
`13ce0b886bde4abe032d00f44c9affd3b1e929837b7a5bd6457b115c3388a15d`;
persistent plume
`9518c2303e363a579b9c7a6b5e6f2a4e39bf62b505d797c3ac128d097ba93c35`.
