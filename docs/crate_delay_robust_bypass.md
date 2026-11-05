# Delay-robust static crate bypass

User-approved separate implementation following the crate progress diagnosis.
Keep old failures/videos and default-OFF/draft/main/primary/PR117 holds. This is
a controller ablation, not exact Safe-SAGE equivalence or a perception change.

## Implementation

`--crate-delay-robust-bypass` requires the existing gated V3 crate profile. The
new recipe differs from that profile only by this flag. The robot receives the
same pose-only, obstacle-unaware waypoint nominal, now predicted for 30 steps;
the primary six-step proposal consumes its unchanged prefix.

The planning-only geometry padding adds maximum forward speed times 0.2 s and
maximum absolute angular speed times 0.2 s (0.05 m / 0.16 rad in this profile).
That duration is one existing execution interval. It is a configurable bounded
planning reserve, not a formal reachable-set guarantee. Existing execution
geometry padding, CBF, freshness, clearance and stopping limits are unchanged.
The final horizon is still revalidated from the actual receipt pose and field.

If the reserved short plan fails, loses its checked reference, or requests less
than 0.02 m/s against a nominal above 0.05 m/s, a six-second (30-step) fallback
uses the same field, CRA, lambda and actuator limits. Cold and available shifted
long-horizon warm starts are compared after complete trajectory checks. The
selection cost adds terminal goal-distance cost with the existing position
weight to the optimizer's objective. This explicit progress preference is part
of the ablation. The longer horizon persists through the shifted warm horizon
until the direct long nominal is checked feasible again. No object identity,
property value, phi, source clock, authored detour or oracle geometry is added.

Returned horizons may contain six or 30 commands. Execution revalidation checks
the **entire** returned horizon with original margins; it never truncates a
30-step path to the short prefix. No feasibility result gives a command an
execution exception. Human selection/predictive policies are outside scope and
rejected. Legacy code is used when the flag is absent.

## Geometry diagnosis and replay

The 21 source-13.4-s candidates that passed the sensor-field checker but failed
the independent disk monitor were also evaluated against the authored crate
AABB. All had AABB separation minus the robot margin at least 0.047873 m. The
disk is more conservative for this set. This does not establish full observed
surface coverage or mesh collision truth; both authored shapes remain strictly
evaluation-only, and the original disk gate stays unchanged in native runs.

All 37 archived request/receipt pairs pass the repaired planner and original
full-horizon checker at both recorded poses. The 17 previous receipt failures
become zero; minimum independent predicted clearance is 0.107327 m. Original
field bytes, source clocks, nominal prefix/goal and old warm inputs are retained.
These are independent frozen-input replays, not a closed-loop native result.
Worst measured offline solve was 3.692489 s wall while unit work also ran; this
is not a standalone performance benchmark. Live freshness limits stay unchanged.

Meaningful regressions cover invalid configuration, opt-in recipe isolation,
unchanged source/nominal data, original execution configuration, full long-tail
revalidation and rejection of unsupported human prediction/short nominal input.
Pre-native validation: **3939 unit tests passed in 260.37 s**, 36 focused tests
passed in 5.71 s, repository Ruff and diff checks passed. Native stationary and
arrival/stopping results are pending at implementation.

## Artifacts

Compact evidence is in `outputs/crate-delay-robust-bypass/`: `geometry-audit.json`,
`four-source-v1.json`, `all-receipts-v1.json`, and test logs. New native runs use
`F:/CRA-experiments/crate-delay-robust-bypass-20261105/`. The separate recipe is
`configs/restaurant_crate_delay_robust_bypass.json`; run through the unchanged
`scripts/evaluation/run_restaurant_demo.py` launcher, case `crate-static`, with
fresh independent60 and own60 before motion. The original 45 s bound and 4 m
straight nominal remain. Preserve any failed attempt before revising anything.
