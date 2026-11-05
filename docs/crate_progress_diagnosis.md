# Crate progress diagnosis

The user authorizes a focused diagnosis of recurring stalls, before another
demo attempt. Preserve both earlier failed closer-crate runs and every V3
video. Keep all default-OFF/draft/main/primary/PR117 holds.

The separate `restaurant_crate_progress_diagnosis.json` recipe reuses the
exact V3 controller, scene, straight nominal, evidence refresh, lambda, limits
and safety checks. It changes ONLY recording to full planner replay with a
source-time window 11.5--15.5 s, and caps navigation at eight simulation seconds.
The expected diagnostic duration limit is not arrival acceptance. Fresh
independent60 and own60 stationary gates precede wheels.

The existing full recorder captures successful low-progress requests, unlike
the prior failure-only mode. Export remains inside worker timing, so capture
overhead is not hidden from freshness validation. Saved snapshots never enter
runtime control. An added regression verifies that a successful zero-forward
result with no checked guidance reference is archived and preserved.

Use original numeric inputs to compare N6/N30 and guidance enabled/disabled,
holding grid, evidence, lambda, constraints and solver budgets fixed. Preserve
the baseline replay first. Report horizon-dependent nominal/warm-start handling
explicitly. Inspect rejected reference candidates and independently checked
turn/move alternatives. A finite unsuccessful search is not proof of global
infeasibility. Oracle fixture geometry is evaluation-only.

Native capture and offline findings pending. Archive new native records in
`F:/CRA-experiments/crate-progress-diagnosis-20261105/`; compact diagnostics in
`outputs/crate-progress-diagnosis/`. No new runtime control policy is adopted.
