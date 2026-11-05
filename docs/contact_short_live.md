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
reject motion.49 focused tests pass; full regression and native outcomes are
pending at implementation.

Keep all previous failures and default-off/draft/main/primary/PR117 holds. Do not
overwrite V3 or promote a short retention/navigation pass into a reliable
contact-ranking claim.
