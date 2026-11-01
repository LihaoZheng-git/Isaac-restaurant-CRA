# Service demo v2 development

User-approved scope: preserve the accepted v1, separate the station/bin/caddy
from surrounding furniture, add visible pot steam, keep a native human walking
before/after the existing parallel encounter, and record the live assessed
robot-camera risk panel beside the follow-camera view. The robot retains its
obstacle-unaware waypoints, fixed lambda, current CRA policy, controller limits,
geometry uncertainty policy and all command/arrival/stopping checks.

## Scene and actor changes

`build_restaurant_demo.py --version 2` authors a new layer over the accepted v1
scene. The initial `restaurant_service_demo_v2.usda` is preserved as an unused
construction attempt; its pot translation composed with the asset scale. The
corrected pot transform in `restaurant_service_demo_v2b.usda` applies a world-aligned
translation to the existing local transform. Its bin placement failed the offline
near-robot setup clearance check (-.0102 m); that attempt is preserved. Candidate
`restaurant_service_demo_v2c.usda` moves that bin clear of the unchanged setup.
The original files are hash-checked unchanged.

The clearing station is centered at (29.45, 5.55), the closed and utensil holders
are separated on its top, and the rounded bin is at (28.15, 6.10). The pot shifts
40 cm toward the existing table edge while keeping its original base height.
The separate animated puff layer is **rendering only**, without collision, mass,
temperature or CRA labels. It is stylized visible steam, not physical aerosol
simulation. Its appearance can be seen by sensors; no authored puff mask or
score enters estimation. A controlled zero-wheel pot diagnostic hides this
layer for clean/no-plume phases and reveals it for the positive phase. The
existing clean-reference/association gates decide whether thermal evidence is
usable; the normal mission retains a labeled prior when they do not pass.

`--background-service-walk` starts a native upper-aisle circuit during the last
20 zero-wheel route-start assessments, before robot navigation begins. It uses
the existing 0.5 m/s native animation and collider.
When the robot reaches authored world y=7.0, the human completes its current
loop to the exit portal, approaches (29.8, 10.5), walks the existing oncoming
segment to (29.8, 7.0), and continues through the aisle. The trigger affects
only the authored actor. It does not supply human state to CRA or robot control,
and it does not pre-script robot avoidance. No teleport or native endpoint idle
is used. Original native geometry checks apply on every update.

The offline authored-layout check found minimum clearance beyond the unchanged
0.4 m square human footprint of .2521 m for the background/oncoming paths and
.2771 m for continuation. The six robot observation/start poses also pass the
unchanged offline setup clearance check (minimum .1468 m). This is a proposed-route geometry check, not native
walking or navigation acceptance.

## Validation sequence

The frozen-source recipe is `configs/restaurant_service_demo_v2.json`.

1. Full unit/regression and Ruff checks; commit before native execution.
2. Actor-only 180 s held startup plus 180 s continuous-walk/release test. The
   robot is inactive only in this actor-only test. Require actual circuit,
   oncoming and continuation progress, native walking and original geometry.
3. Fresh current-source near-human perception component, independent route-start
   and close-human stationary gates. Existing oncoming checks retain their
   exact thresholds on that explicit segment; full circuit geometry, clocks
   and containment remain separately checked across the entire mission.
4. Stationary revised-prop and clean/no-plume/plume diagnostics. Rendered scenery
   alone does not establish accepted object ownership or thermal influence.
5. Full guarded mission with actual final arrival/stopping and source-aligned
   camera-mask/video audits. Preserve failures, and do not promote incomplete
   arrival, prior-only factors or partial component success as a full result.

The full unit suite passed 3389 tests before the final pose adjustment; focused
checks and Ruff were rerun afterward. A default sandbox temporary-directory
failure was resolved by selecting a new repository-local pytest temporary path.
## Preserved v2c native results (dd7b7c9)

All four main prerequisites passed with unchanged source: actor-only hold and
continuous circuit (minimum layout clearance .227185 m), 115-source moving-human
perception component (65 furniture-retention sources), startup60 and near60.
Both camera audits passed exact frame/mask/score parity: startup60 mask updates
of one furniture track, near51 updates of one human track,70 video frames each.

The five separate prop diagnostics failed. Each recorded one initial camera/
LiDAR epoch rejection. The bin had accepted image/LiDAR evidence (911 returns,
overlap1.0 in its final source), but its visible-surface centroid was .201734 m
from the authored center and failed the unchanged .20m fixture-matching gate.
The closed/utensil caddies were not separately matched; the pot lacked accepted
visual mass/contact. Both pot thermal phases had zero accepted matched cues.
Their camera presentation audits passed separately; this does not repair the
failed perception gates. Preserve all outputs under `outputs/service-demo-v2/`.

Inspection of the raw plume images exposed an authoring defect: USD scale was
before translation, stretching puff world height by1.5 as well as shape. The
first full-route launch was stopped at stationary frame13, before navigation;
it has no final native report or accepted motion result. The corrected candidate
is `restaurant_service_demo_v2d.usda`. A world-transform regression now checks
puff positions at three times. A proposed prop-only .2s reassessment was caught
by five CLI regression failures: that rate is allowed only for native-human
experiments. It is reverted to the original1.0s setting, preserving the guard.
No safety, ownership or score gates change.
All four fresh native prerequisites must be rerun for this candidate.
After restoring the supported configuration, the full suite passed3390 tests
in117.82s; Ruff and the unchanged offline layout-clearance checks also passed.

Keep the development PR chain and main/primary/PR117 holds. The eventual final
package belongs in `outputs/final demo/v2/`; the camera-only preview and v1 remain
separate. No trained steam model or environment upgrade is introduced.
