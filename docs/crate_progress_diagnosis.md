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

Archive new native records in
`F:/CRA-experiments/crate-progress-diagnosis-20261105/`; compact diagnostics in
`outputs/crate-progress-diagnosis/`. No new runtime control policy is adopted.

## Captured outcome (runtime baf2b9f)

Fresh independent60 and own60 stationary gates passed. The eight-second native
capture ended with the expected `duration_limit`, **not navigation acceptance**.
There were 78 live fields (69 while translating), minimum independent clearance
0.171570 m, minimum analytical CBF margin 0.064551, and stopping drift 0.003519 m.
Source hashes remained unchanged. All 408 previously archived final-demo files
retain their original SHA256 values.

All 67 returned plans reported `converged`; 35 had checked guidance references
and 32 had `no_exact_safe_progress_reference`. The first reference failure was
at navigation 5.233333 s, chassis x=1.039949 m. Of 480 command rows, 137 requested
zero forward velocity: 97 waiting for a plan, 32 horizon invalidations and eight
stale-field rows. Recording overhead remains included in native freshness
timing; do not treat this as a matched pause-duration comparison to the old run.

## Exact diagnosis

1. **Reference collapse at the padded geometry boundary.** At source 13.4 s,
   all four admitted static Laplace references are almost the same straight
   trajectory. The planar constraint does not yet modify the nominal velocity;
   changing the turn-rate cap or enabling turn-first therefore does not produce
   a different path. All four fail the dense padded geometry gate at the final
   N6 state, although the DTCBF margins remain positive. The tracking objective
   then becomes inactive. With N30, these references instead encounter the
   unsafe-configuration boundary at step nine. This is not repaired by changing
   the guidance weight alone.
2. **Request/receipt fragility reproduces the pauses.** All 37 window snapshots
   match their native request and receipt epochs. Their field is identical at
   receipt. Seventeen saved plans pass at the request pose but fail at the
   actual receipt pose, exactly matching all 17 corresponding native
   `new_field_invalidated_plan` rows. Movement of only 0.059--0.898 mm pushes
   the predicted tail over the padded raster boundary at chassis x=1.275 m.
   Unpadded endpoint checks, DTCBF margins and the immediate analytical command
   all still pass. The diagnostic reason's name does **not** mean a new field
   caused these particular stops. No phi change or track loss is needed to
   reproduce them. A synthetic 0.2 mm regression preserves this distinction.
3. **A longer horizon exposes feasible turns, but is not a complete fix.**
   Exact native command replay passes at sources 9.9, 13.2, 13.4 and 15.3 s.
   The 2x2 comparisons reset the warm start in every cell and regenerate the
   same pose-only waypoint nominal (the original prefix and target must match).
   Every other solver/field/geometry/risk/limit setting stays fixed. At source
   13.4, both N6 variants select a straight 0.18 m prediction; both N30 variants
   select a 1.139643 m turning prediction with 0.082890 m independent minimum
   clearance. At the first failing receipt, the N30 alternative also passes
   at the actual delayed pose (0.083861 m independent clearance). These are
   frozen-field witnesses, not executable paths or full-route success.
4. **Field and independent geometry are not interchangeable.** A finite,
   symmetric, field-checked turn/move pool at source 13.4 admits 291/295 N6
   candidates and 174/425 N30 candidates, including the analytical first-command
   check. Evaluation afterward finds 21 of those 174 longer candidates below
   the original 0.02 m independent disk-monitor threshold (minimum -0.090055 m).
   No N6 candidate in this pool violates that monitor. This shows a
   representation/monitor discrepancy, not necessarily mesh collision: the
   monitor is a conservative circumscribed disk. Its conservatism versus
   incomplete observed surfaces is not separated by this audit. Keep the
   independent gate and the old full-run clearance failure. The footprint audit
   explicitly undoes the controller's lookahead reference shift before querying
   chassis-to-fixture clearance.

The native warm start and the cold-start N6 comparison differ: the former
optimizes forward speed against the edge, while the latter chooses a slower
checked seed. Do not attribute that difference to horizon or guidance. The
matched N6/N30 matrix isolates horizon only **within its cold-start cells**.

## Next bounded repair

Address plan execution robustness and static bypass selection together: test
candidate horizons against the small state evolution during solver/receipt
latency, and generate a checked turning alternative when the static reference
fails instead of repeatedly optimizing a straight horizon against the raster
edge. Use current sensed fields and the pose-only nominal; no authored bypass,
oracle geometry, frozen phi or relaxed acceptance gates. The existing longer
horizon provides an offline witness worth testing, not a reason to promote N30
alone. Resolve/report the independent geometry discrepancy before a full demo.
Retain all original dense geometry, analytical CBF, clearance and stopping
checks. Validate the exact archived failing inputs first, then fresh stationary
gates and a bounded native pass with actual arrival/stopping.

## Reproduction and verification

The offline tool is `python -m robot.evaluation.crate_progress_diagnosis`.
It requires `--trusted-local-replay`, validates sibling pickle paths and hashes,
rejects output replacement, checks native command/nominal parity, and never
publishes a command or renews an evidence clock. Only load trusted local captures.

```text
--metadata F:/CRA-experiments/crate-progress-diagnosis-20261105/motion-v1/planner-replays/window-013.json
--fixture F:/CRA-experiments/crate-progress-diagnosis-20261105/motion-v1/fixture-monitor.json
--pool --output <new-matrix.json> --trusted-local-replay

--receipt-run F:/CRA-experiments/crate-progress-diagnosis-20261105/motion-v1
--output <new-receipt-audit.json> --trusted-local-replay
```

Saved outputs include `window013-matrix-v2.json`, `receipt-audit-v2.json`, three
additional source matrices, `window011-delayed-variants-v1.json`, `summary.json`,
`diagnosis.png`, and `preservation-after.json`. The failed initial pytest attempt
could not create the sandbox's default temporary directory; reruns use a new
workspace-local `--basetemp`. Final targeted regression: **171 passed in 20.78 s**
(MPC, guidance, selected tangent, replay, diagnostic recipe and V3 controller
profile); repository Ruff, compile and diff checks pass. All 40 captured
snapshot hashes verify. No complete repository suite or new full mission is
claimed for this diagnostic-only change.
