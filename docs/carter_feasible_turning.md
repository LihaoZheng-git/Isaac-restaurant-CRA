# Carter turning feasibility without early preparation

The user's latest decision excludes quiet-person early preparation from the
selected restaurant demonstration. Normal obstacle-unaware waypoint motion,
CRA field policy and online MPC/analytical intervention remain. A stationary
person remains physical occupancy; excluding preparation never removes them.
Observed approaching motion can support the separately labeled passing
guidance. The existing explicit guidance-tracking objective remains an ablation,
not unmodified Safe-SAGE or the primary comparison.

The accepted earlier route-guide run at `3fb7e2e` reached all three guides and
the final goal, with a 0.222896 m nearby bend and a 2.383333 s translation hold.
It recorded 27 fresh `unsafe_configuration_boundary` planner rejections. That
status comes from the **initial** configuration query, before optimization;
it does not establish that different turn seeds at that already blocked state
could pass. The first low-progress guidance reference preceded the boundary
rejection. Earlier five-second replay buckets overwrote the exact failing input.

First preserve the first low-progress and initial-geometry rejected planner
requests, using the existing opt-in trusted local replay path. Export stays
inside worker timing. Snapshots retain original source epochs, are never
loaded by runtime control, and do not change commands or safety checks.
Reproduce with early/rear preparation and predictive redesign disabled, keeping
the original actor route/onset, CRA/factor/body policies, lambda 1.25, nominal
0.20 m/s, MPC 0.25 m/s, solver budgets and all original checks. This diagnostic
does not by itself constitute a fix or a new accepted native trial.

Any implementation must address a demonstrated planning/geometry cause, retain
all occupancy and margins, and pass exact DTCBF, dense configuration checks,
per-step analytical filtering, original freshness/identity guards, independent
clearance/containment, arrival and stopping. Numerical completeness is not
fresh four-factor perception. Revalidate stationary before new navigation.
Keep the separate original smoothness target (actual nearby hold at most 0.5 s
and at least 50% reduction) and preserve every failed report. No primary,
main, PI-demo, long-service or PR117 hold is released by development alone.

## Reproduction and first bounded candidate

At `d892784`, the fresh original human/perception component and independent
60-frame stationary gate passed. The unchanged no-preparation motion invocation
reached all three goals in 21.333333 simulation seconds, clearance 0.352025 m,
stopping drift 0.002406 m and minimum analytical margin 0.080218. It remains
**FAILED overall**: four supported moving-kinetic rows versus the required five.
Its actual translation hold was 2.383333 s, reproducing the original pause.
The saved report is never retroactively passed:
`135b0b8fa9bd3d9eb5f4bcf1d1aed72bc8366e3af4ea23bde6f4151e476ed4c6`.

The preserved first low-progress request at source14.2 had a free starting
configuration but no accepted horizon in the tested bounded search, including
a maximum-speed0.25 m/s probe. The first initial-geometry rejection at source15.3
was occupied by the human body prior (ID3000000000). Its interpolated h was
positive; the mandatory occupied-cell rejection correctly overruled that value.
There is no additional free-cell boundary ring: the Dirichlet mask includes
occupied configuration cells and the outer grid edges.

An offline **body-only component** retained the exact radius0.65 m and footprint
with physical margin0.05 m. The recorded starting pose had continuous
rectangle/circle margin0.199792 m. Its active-yaw raster cell was occupied at
resolution0.10 m and free at0.05 m. This shows numerical conservatism in that
component, not that every physical obstacle is clear or that a navigation
trajectory exists. Original rasterization conservatively accounts for both
obstacle/query cell extents and the yaw slices; none of those checks is bypassed.
See the immutable `turning-rejection-audit-v1.json` in the diagnostic directory.

The next candidate exposes `--grid-resolution .05` (legacy default remains.1),
with unchanged bounds, inflation, footprint, body radius, lambda1.25, nominal.20,
MPC maximum.25 and all physical/source/arrival/stopping checks. The complete
field retains every observed return and unassessed occupancy. The actual
configuration is saved separately and in the final report. More computation is
expected; freshness and optimizer-age limits remain unchanged.

The live CLI also separates the existing explicitly approved three-second
adult-mass/current-body retention option from early preparation. It still
requires explicit body identity and all original provenance/expiry checks;
its implementation is unchanged. The next trial enables it to address the
reproduced evidence gap, not to lower CRA risk or change physical measurements.
Grid refinement plus mass retention is a declared **two-option** candidate,
not a single-variable causal comparison. New no-preparation mass component,
fine-grid independent and own60 gates precede its wheels. Native outcome is
pending; early/rear preparation and predictive geometry/redesign stay disabled.
