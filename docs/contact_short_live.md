# Corrected contact: short moving validation

The user authorized a short bin/crate test after the pixel-boundary correction's
stationary checks. This is separate from V3; all 324 existing final-demo files,
including the PI-sharing copy and its preservation manifest, are protected by
`outputs/contact-short-live/preservation-before.json`.

The test uses the existing V3 open-aisle bench and its original two explicit
observation poses: bin robot(19.35,4.05,0), crate robot(15.65,4.05,0). Original
licensed assets, scale, materials, raised Owl camera and current image/LiDAR
association remain unchanged. The obstacle-unaware nominal route is the existing
0.40m local forward segment. Original arrival tolerance, online MPC, analytical
CBF, fixed lambda, robot limits, margins and stopping checks remain unchanged.
No person, thermal fixture or long service demonstration is added.

The CLI now admits `--contact-pixel-denoise` motion only with the explicit gated
`--image-object-approach --static-prop-evidence` mode. Explicit observation poses
remain zero-wheel-only in older recipes; the exception also requires the contact
correction. All existing approach requirements, including own60 stationary
fields, camera resolution and no human/server, remain. The launcher admits only
the two named corrected-contact cases and requires a separately passed stationary
gate with exact runtime/scene hashes and the same executed command.

The correction and static-property algorithm are not changed by this branch.
Accepted intrinsic mass/contact remain tied to their original source, confidence
decay and expiry. Kinetic and interaction stay live; final contact risk and phi
are recomputed. This short test is not a general sharpness calibration, complete
four-factor sensor assessment, live initial-acquisition test along the full
service route, or acceptance of a replacement V3 recording.

## Reproduction

Use the existing environment; no installation or upgrade is required. Run one
Isaac process at a time, from a clean committed checkout. Each output must be new.

```powershell
python scripts/evaluation/run_restaurant_demo.py --config configs/restaurant_contact_short_live.json --case bin-static --stage stationary --output outputs/contact-short-live/bin-stationary-v1
python scripts/evaluation/run_restaurant_demo.py --config configs/restaurant_contact_short_live.json --case bin-static --stage motion --stationary-source outputs/contact-short-live/bin-stationary-v1 --output outputs/contact-short-live/bin-motion-v1
python scripts/evaluation/run_restaurant_demo.py --config configs/restaurant_contact_short_live.json --case crate-static --stage stationary --output outputs/contact-short-live/crate-stationary-v1
python scripts/evaluation/run_restaurant_demo.py --config configs/restaurant_contact_short_live.json --case crate-static --stage motion --stationary-source outputs/contact-short-live/crate-stationary-v1 --output outputs/contact-short-live/crate-motion-v1
```

Audit each completed moving run using `scripts/evaluation/audit_static_prop_motion.py`
with `--source` and a new `--output`. The auditor replays current-point membership
and exact original mass/contact values, bounds, timestamps, source frames and
confidence decay. Motion clocks must remain current. It reports the unchanged
20cm whole-center diagnostic separately from current partial-surface support,
and uses fixture geometry only for independent reporting. It checks original
input hashes before and after replay.

Before native tests, the auditor reproduces the prior accepted bin's16/16 retained
updates. A separate copied negative control deliberately changes a mass value,
contact timestamp, kinetic timestamp and point count; all four errors are caught.
The original source is untouched. The first focused test run exposed the old
explicit-pose zero-wheel guard; its failure is preserved, and only the authorized
corrected-contact approach receives an exception. Prior zero-wheel recipes still
reject motion.49 focused tests pass (29.95s), 3,872 full unit tests pass
(222.18s), and repository Ruff/diff checks pass.

## Native results at e1f2624

All four runs PASS with frozen runtime and scene sources, no tracebacks and
unchanged acceptance thresholds. Both independent and both own60 stationary
gates pass; each retains its initial camera/LiDAR epoch rejection. Each moving
run reaches its single waypoint and stops. This is approximately30cm measured
travel toward the authored40cm goal under the original10cm arrival tolerance.

| Measure | Bin | Broken crate |
|---|---:|---:|
| Navigation simulation seconds | 1.800000 | 1.816667 |
| Final local forward displacement (m) | 0.304202 | 0.304336 |
| Live fields / translating fields | 12 / 11 | 12 / 11 |
| Minimum independent clearance (m) | 0.530769 | 0.529563 |
| Minimum CBF margin | 0.286068 | 0.222315 |
| Stopping drift (m) | 0.002724 | 0.003184 |
| Exact retained target rows | 12 / 12 | 12 / 12 |
| Retained rough mass (kg) | 0.695455 | 2.550128 |
| Retained intrinsic outline cue | 0.095997 | 0.394840 |
| Final contact-factor range | 0.140538 | 0.162113--0.494840 |
| CRA phi range | 0.607244--0.607764 | 0.617110--0.786885 |

The source/point auditor passes every retained row without errors. Original
values, intervals, camera source frame, timestamps and confidence decay match
exactly; current LiDAR membership is replayed, not assumed from an unchanged ID.
The20cm whole-center diagnostic passes12/12 for each target in these views.
Maximum independent mesh-box excess is0.002309m for the bin and0 for the crate,
within the unchanged3cm diagnostic allowance. This does not repair or erase
earlier whole-center failures in other views or full-mission tracking gaps.

The crate's visual contact cue remains higher in this pair and never changes
during motion. However, final contact risk and phi are deliberately not frozen.
Bin relative speed is inferred in12/12 updates (0.00636--0.21338m/s). The crate
has10/12 inferred speed updates (0.02045--0.50622m/s) and two unavailable updates
at source9.4/9.5s: new raw fragments187/192 have only one observation. The
composite motion record uses the least-established contributing fragment, so
those two updates retain the configured0.25 kinetic prior and widen route
interaction bounds to[0,1]. This raises final contact risk to0.494840 and phi
to0.786885, despite an unchanged intrinsic cue of0.394840. Recovery at9.7s
restores inferred speed with lower confidence and phi0.629725. Arrival still
passes. Do not attribute the phi peak to new sharpness or hide these updates.

There is also a crate velocity peak at8.6s coincident with a two-fragment to
one-fragment transition. This remains a centroid/membership sensitivity, not
validated physical movement of the static crate. The short test establishes
retention and navigation with the existing explicit fallback policy; it does
not establish uninterrupted, calibrated kinetic perception.

`bin-motion-audit.json`, `crate-motion-audit.json`, `motion-evidence-diagnostic.json`
and `summary.json` under `outputs/contact-short-live/` retain the exact sources
and all rows. `short-live-evidence.png` displays the intrinsic cue, current
contact/phi and speed, including both missing-speed updates. Native videos remain
in each run folder. All324 previous final-demo files were rehashed unchanged.
This branch is draft PR235 stacked on PR234; no merges. No full restaurant rerun
or new V3 video is accepted by this short result.

Keep all previous failures and default-off/draft/main/primary/PR117 holds. Do not
overwrite V3 or promote a short retention/navigation pass into a reliable
contact-ranking claim.
