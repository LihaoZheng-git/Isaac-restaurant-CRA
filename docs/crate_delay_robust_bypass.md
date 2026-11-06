# Delay-robust static crate bypass

User-approved separate implementation following the crate progress diagnosis.
Keep old failures/videos and default-OFF/draft/main/primary/PR117 holds. This is
a controller ablation, not exact Safe-SAGE equivalence or a perception change.

## Implementation

`--crate-delay-robust-bypass` requires the existing gated V3 crate profile. The
new recipe adds this flag and numeric planner recording. The robot receives the
same pose-only, obstacle-unaware waypoint nominal, now predicted for 30 steps;
the primary six-step proposal consumes its unchanged prefix.

The current planning-only geometry reserve retains maximum forward speed times
0.2 s (0.05 m here), independent of stopping/turning, and one physics step of yaw
allowance (0.013333 rad here). The initial version used maximum rates throughout
(0.05 m / 0.16 rad), which proved restrictive during slow turning. An intermediate
command-dependent reserve changed the feasible set while turning or stopping;
the fixed spatial reserve avoids that feedback. Heading delay is represented by
explicit held-input pose probes, rather than extra yaw padding on all future
headings. This is a planning heuristic, not a formal reachable-set guarantee. The
planner also checks complete candidate horizons from six possible receipt poses:
the request pose, the held-input endpoint, and one-step forward/backward/yaw
perturbations there. These are finite probes, not exhaustive reachability.
The real receipt pose remains authoritative. Existing execution
geometry padding, CBF, freshness, clearance and stopping limits are unchanged.
The final horizon is still revalidated from the actual receipt pose and field.

If the reserved short plan fails, loses its checked reference, or requests less
than 0.02 m/s against a nominal above 0.05 m/s, a six-second (30-step) fallback
uses the same field, CRA, lambda and actuator limits. Cold and available shifted
long-horizon warm starts are checked completely. A freshly optimized shifted
continuation with at least 0.05 m/s first command and the existing minimum goal
progress is retained after all receipt probes pass. Otherwise, cold recovery
and the finite arc pool are searched and compared. This continuation preference
avoids repeating expensive cold recovery while a checked moving plan exists;
it does not impose a minimum speed or skip feasibility checks. The
selection cost adds terminal goal-distance cost with the existing position
weight to the common command objective. A finite symmetric online arc pool also
proposes moving candidates. Arc proposals require the complete reserved dense
geometry/DTCBF checks, receipt-pose probes, and an unchanged feasible immediate
analytical command. Neither a moving-speed floor nor a precomputed route is
imposed. The arc proposals are a separate geometric planning ablation, not a
claim that the original Laplace objective produced them. This progress preference is part
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
arrival/stopping results were pending at that implementation.

## First native trial and correction

At runtime `2f02bdf`, independent `stationary-v1` and motion's own 60-source
gate passed. `motion-v1` then stalled: last observed source 26.0 at local
(1.188004, 0.107948, 0.554653). It was explicitly interrupted and preserved with
an interruption record. There is no final arrival/stopping acceptance or
finalized-video claim. Its sources stayed unchanged. The longer optimizer could
still repeatedly choose almost-zero first commands while deferring progress.

The revised implementation adds the arc candidates, held-command reserve, and
receipt probes described above. The held-command-only replay left one of 37
receipt checks invalid (a final DTCBF constraint); the receipt probes removed
that failure. All 37 original request/receipt pairs then passed, minimum
independent predicted clearance 0.110192 m. Old field bytes, source clocks,
nominal prefixes and original warm inputs remain unchanged. The archived last
issued command is matched to the original request epoch. These are offline
checks, not native acceptance.

A frozen-field ideal-kinematics sequence without the receipt probes eventually
stopped after 0.459183 m; predicted disk clearance also became negative at some
steps. The command-dependent version with probes (`closed-loop-v5`) also stopped
after 0.409312 m. Preserve both failed diagnostics; neither establishes live
sensor or wheel performance. The fixed spatial reserve is evaluated separately
in `closed-loop-v6`. These artificial sequences keep the old field and temporal
derivative unchanged, so they do not predict a live sequence of refreshed fields.
Fresh prerequisites and native arrival remain
mandatory for the corrected version. Full successful planner snapshots now
accompany the native trial so a successful-but-stalled horizon is inspectable.

The final fixed-reserve request/receipt replay (`all-receipts-v6`) passes all
37 pairs, minimum independent predicted clearance 0.101931 m. Cost-ordered
receipt checking selects identical command sequences to the exhaustive version
at all 37 pairs. Profiling a later frozen-field state identified repeated cold
recovery (thousands of trajectory checks) as the major planning cost; the
checked-continuation rule addresses that separately. The full unit suite passed
3946 tests in 337.05 s before this final continuation optimization; all 68
focused controller/runtime/recorder/route tests then passed in 9.79 s, with
repository Ruff and diff checks passing. These are not real-time guarantees.

## Second native trial and checked stopping tails

At `f4d7ce5`, the fresh independent 60-source stationary gate and the navigation
run's own 60-source gate passed. `motion-v2` passed the earlier stall and turned,
then stopped progressing near local (1.364, 0.095). Repeated live planner calls
took 24–25 s wall. The run was deliberately interrupted and preserved, sources
unchanged; no arrival/stopping/video acceptance. The exact source-20.9 checkpoint
reproduces the selected 30-step plan: rotate for three intervals, then remain
stationary. It has zero translation over the entire horizon. In the matched
replay the short solve took 0.087 s, shifted long solve 1.727 s and cold recovery
13.087 s (15.216 s total). The short plan actually moved first at 0.047248 m/s,
passed every receipt probe and had 0.072576 m independent predicted clearance.

All 75 continuously moving arc proposals failed dense geometry. Removing the
temporal derivative for diagnosis did not change that count; it stays unchanged
in runtime. Additional turn-return (288) and hold-return (450) geometry/CBF
probes also produced no accepted candidates. Sampled cross sections include
free poses; this is not evidence that the physical corridor is globally blocked.
A separate short-plan-first frozen-field experiment repeatedly stopped at the
same state and was preserved without claiming a fix.

The revised finite search also tests each arc with a predicted stopping tail at
6, 12, 18 or 24 intervals: taper the last two moving commands, then predict zero
for the remaining intervals. **Check all 30 steps**, including the stopped
portion, with the same original constraints and receipt probes. Never publish
an unchecked prefix. Recompute online from the current field/pose/nominal;
neither the oracle fixture nor the offline candidate sequence enters control.
Keep a progressing continuation. Try the extra stopping tails when the shifted
horizon makes no goal progress, or when both the previously issued and proposed
forward speed are below 0.02 m/s despite forward demand above 0.05 m/s. They can
also be tried when no long candidate advances. Prefer a checked moving repair;
retain the checked moving short alternative when the long choices cannot advance.
At source 20.9 the recorded-input replay now
selects (0.099814 m/s, 0.2 rad/s) instead of (0, -0.8), in 3.520 s offline.
This is evidence at the original failure, not a third native run or proof of
arrival. The full-duration simulation still decides navigation acceptance.

The unconditional search changed earlier good decisions and failed the original
independent clearance replay (0.009695 m); it was not sent to wheels. The narrow
repair preserves all 37 earlier command sequences. A separate frozen-field
reproduction from the native stall advanced 0.174662 m over 12 ideal replans.
The final 72-replan sequence from the older bottleneck passed the crate,
advancing 1.804905 m in 10.8 modeled seconds; it did **not** reach the 4 m goal.
Its actual ideal commands missed the original 0.02 m disk gate: minimum 0.017449 m,
304/720 samples below threshold, with some unexecuted prediction tails as low
as -0.013582 m. Against the crate's mesh-derived world bounding box, the **same**
padded footprint gives 0.080439 m minimum. This explains a geometric discrepancy;
the box is diagnostic only. Keep the independent disk guard in the fresh native
test. Frozen fields, ideal integration, and simulated wheels are distinct results.

## Artifacts

Compact evidence is in `outputs/crate-delay-robust-bypass/`: `geometry-audit.json`,
`four-source-v1.json`, `all-receipts-v1.json`, and test logs. New native runs use
`F:/CRA-experiments/crate-delay-robust-bypass-20261105/`. The separate recipe is
`configs/restaurant_crate_delay_robust_bypass.json`; run through the unchanged
`scripts/evaluation/run_restaurant_demo.py` launcher, case `crate-static`, with
fresh independent60 and own60 before motion. The original 45 s bound and 4 m
straight nominal remain. Preserve any failed attempt before revising anything.
