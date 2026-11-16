# Closer-crate straight-route test

The user requested a simple test with the crate nearer the unchanged straight
waypoint direction, retaining the confirmed better-evidence update rule. This
separate default-off `--crate-route-probe` uses the same robot start
(15.65, 4.05), yaw zero, and a 4 m straight nominal goal. The prior 40 cm approach
stopped before passing the crate; it cannot answer the new bypass question.

The original licensed crate moves 45 cm toward the route, from world center
(17.5, 3.0) to (17.5, 3.45). The bin is inactive in this crate-only layer.
The original mesh/scale, camera, mass/contact estimators, confirmed refresh,
live kinetic/interaction, priors, lambda, MPC/analytical CBF, physical limits,
clearance, arrival and stopping checks remain unchanged. No authored detour,
supplied phi or moving-human guidance preference is introduced. Original V3
and short-test outputs remain untouched.

Require a fresh source-matched independent60 stationary gate, then own60,
before the bounded motion trial. Report actual lateral deviation from the
straight nominal, passage beyond the crate, arrival/stopping/clearance, fresh
and retained evidence, and accepted/rejected refresh attempts. A successful
curve demonstrates this controller/scene combination, not isolated CRA benefit
over fixed-risk control. Preserve failures and default-off/draft/main holds.

## Result at 79e0c9e

The closer placement produces a curved correction, but the bounded navigation
test FAILS arrival: maximum lateral deviation0.345838m; final local pose
(1.957624,0.346059,0.171103), still2.071487m from the4m goal after45 simulation
seconds. The robot center passes the crate center, but its rear has not passed
the crate's far edge along the travel direction. Minimum independent
clearance0.094290m, minimum CBF margin0.185449,
and stopping drift0.000927m remain within their original checks. There are310
live fields,238 committed while translating. Do not label the curve alone a
successful bypass or a completed demo.

Fresh independent60 and own60 stationary gates both PASS, each with58 accepted
crate assessments after qualification and a current saved mass/contact record.
The source/point replay has zero errors, but continuous target reuse FAILS:
84/310 live fields contain the original crate record, at source8.2--20.3s.
Its saved rough mass3.028027kg and shape cue0.459253 remain fixed. Selected points
stay within the original mesh-box allowance; only36/84 observed-surface centers
meet the separate20cm whole-center diagnostic. Current velocity/interaction
remain live: phi ranges0.619316--0.814060, including unavailable-speed fallback.
No constant-phi or full four-factor sensor claim is justified. A last-source
independent box query finds zero current crate-region returns; this is not
evidence that its physical geometry vanished or that retained occupancy cleared.

Exact refresh replay PASS covers60 independent and370 own-stationary/live
sources. All12 crate refresh attempts reject incomplete old support and lack of
meaningful new coverage; five also have lower/unavailable segmentation quality,
three omit projected old anchors. ZERO crate geometric confirmations and ZERO
native property replacements occur. Three non-target geometric confirmations
also fail property storage. Positive replacement remains controlled-test evidence.

All332 returned planner results are accepted at request evaluation, but subsequent
field updates can invalidate them. Reasons:178 exact-safe-seed recoveries,
95 converged solutions and59 exact-safe-progress seeds. Of2701 command samples,
1025 have zero forward velocity, including594 without a stop reason. Separate
plan waits/invalidations/stale-field stops occupy approximately7.18s. Between
navigation40--44s every applied forward command is zero; accepted zero-forward
seeds alternate angular commands +/-0.1rad/s, with only1.43mm actual translation.
Thus the result includes genuine forward stalling, not just a low constant speed.
This does not prove global infeasibility or isolate a single underlying cause.

Important comparison boundary: this experiment deliberately inherits PR237's
short-bench controller (30 MPC steps/6s horizon, guidance tracking OFF, simulation
rate.05). V3 uses6 steps/1.2s, the separately reported guidance-tracking objective,
and rate.025. Physical limits/lambda are unchanged, but this is NOT a matched V3
controller test. The next bounded comparison should use the existing V3 navigation
settings in this same isolated placement before changing risk scores or proposing
a new controller objective. Keep the current failed result.

Validation:80 focused tests,3902 unit tests in233.57s, Ruff and diff checks PASS.
Runtime source/scene hashes remain frozen. Draft PR238 stays unmerged.

## Artifacts and storage

Reports, plot, exact source/refresh audits, command diagnosis and archive pointers
are in `outputs/crate-route-probe/`. Native recordings are in
`F:/CRA-experiments/crate-route-probe-20261105/stationary-v2` and `motion-v2`.
The video `motion-v2/service-navigation.mp4` retains the incomplete run and its
pauses; `service-risk-camera.mp4` records the actual perception presentation.

The first C: stationary attempt stops after46 fields with Errno28 while saving
a sensor snapshot. Preserve `outputs/crate-route-probe/stationary-v1`: no wheels
ran and no gate completed. The separate F: retry passed stationary validation;
the motion run then ended with the intentional failed-navigation gate exception,
not another storage failure. Thirteen completed historical JSONL logs received
lossless NTFS compression to relieve C: pressure; before/after SHA256 hashes are
identical. Two already-compressed logs were also verified unchanged. No source,
asset, old video or archive content was deleted/replaced. All408 previous final-
demo files remain byte-identical. Storage audits are saved with the new results.
