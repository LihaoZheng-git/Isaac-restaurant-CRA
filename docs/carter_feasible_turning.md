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
