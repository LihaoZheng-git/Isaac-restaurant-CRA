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

## Preserved v2d full-route failure (efc4059)

All four fresh prerequisites passed: actor circuit,115-source component,
startup60 and near60. The full mission `outputs/service-demo-v2/long-motion-v2`
then failed `nearby_track_lost` at113.666667 navigation seconds,4/8 guides and
749 fields. Clearance was .391761m, minimum CBF .138337 and stopping drift
.001332m. These are not final arrival. Original report SHA:
`a6ac73d110e90bd1881079e5394679a59c56efd733590ed95dfb586dbdcb6f55`.

The original four-return furniture fragment2836 (source112.2) remained fully
covered at the rejected source121.6, maximum correspondence .003308m, by331
nearby returns uniquely owned by3004. However, that large parent's centroid
velocity crossed .2m/s;11/7248 current parent points lacked adjacent-scan .06m
support. The existing whole-parent motion exclusion purged the old fragment.
The saved raw comparison is a diagnosis, not a three-source qualification replay.
The user approved testing a separate bounded fragment proof with all original
support, human-exclusion, expiry, risk and actual-field checks unchanged.

The camera presentation audit passed117 accepted masks/25 tracks/1216 frames,
using actual phi .561113..807260. Fresh complete four-factor updates remain zero.
The correctly placed steam diagnostic still failed thermal association: zero
matched accepted pot cues in both18-source no-plume and plume phases. Visible
puffs do not justify a thermal score or heat-driven avoidance claim. The failed
route and its images are retained; accepted final demo/v1 is unchanged.

## Separate continuous upper-aisle departure

The later93c6463 trial passed all four prerequisites, own60 and every earlier
furniture-stall location, but did not reach the final goal. The continuously
walking human repeatedly returned across the delivery corridor. The robot made
only small gains between repeated yields. All1,089 stationary/live sources had
zero unresolved human regions; this is distinct from the earlier duplicate-ID
failure. Preserve `outputs/verified-part-recovery/long-motion-v1` as a failed
stress case. It does not establish geometric infeasibility or general liveness.

`--background-upper-departure` is a separate actor-only demo variant. The original
background circuit, approach, oncoming encounter and first continuation goals
(27.8,7) and(27.8,11.2) stay. The actor then walks through(27.8,14.4) and(29.7,15.1)
to(29.7,23.5), continuing between upper-aisle y18.5 and23.5. This keeps the human
walking after the encounter without repeatedly cutting back across the delivery
corridor. Authored actor waypoints never enter robot perception or control.

Robot waypoints, CRA, fixed lambda1.25, body/factor/unknown priors, margins, source
and identity clocks, online MPC and analytical checks are unchanged. This new
scenario cannot be presented as repair of the repeated-return stress case. It
uses the same v2d scene layer, props and rendered plume, with all existing thermal
and four-factor perception limitations retained.

The authored layout survey checks the same0.4m half-extent square used by the
independent actor guard, including pixel allowance and a0.0125m between-sample
allowance. Minimum planned clearance is0.239645m. Native walking can deviate,
so this is only a route-design check. The actor diagnostic extends release to
210s to cover the longer continuation; the explicit region audit allows y24.1
only for this option. Actual geometry, collider, grounding and walking checks
remain unchanged. Fresh actor/component/startup/near and own60 stationary gates
must pass before a complete wheel mission can establish acceptance.

Recipe: `configs/restaurant_service_demo_v2_upper_departure.json`.
Design survey: `outputs/upper-service-departure/route-survey.json`.
The parked sensor record now also exports the actual recovery configuration and
visual-part proofs already computed by the worker, so future recovery events can
be audited without reconstructing omitted configuration. This is report content,
not a new estimator or recovery rule.

Source validation:63 focused tests passed in5.58s; all3,622 unit tests passed
in122.08s. Ruff, compilation and diff checks passed. Native validation is pending.

The first actor-only upper-departure attempt at7571115 failed before a final
report or any wheel experiment. Its last actor sample was(27.8815,11.0167) at
325.4333s, with minimum measured layout clearance0.227185m. The next target at
y14.4 lay outside the native actor NavMesh's unchanged y14 boundary. The old
setup is preserved under `outputs/upper-service-departure/background-v1`.
The explicit option now extends only the actor NavMesh to y25, retaining the
legacy domain and all collision/clearance checks. Setup exports the actor extent,
and unexpected actor-diagnostic exceptions are saved before SimulationApp closes.
Fresh native prerequisites are required; the planned geometry survey is not a pass.

NavMesh correction validation:44 focused tests passed in6.05s; all3,623 unit tests
passed in122.74s. Ruff, compilation and diff checks passed. Legacy actor domains
remain unchanged; the upper route is tested to fit within its explicit domain.
# Upper-departure preflight correction

At c30ff7a, all four fresh prerequisites passed: actor background-v2
(180 s hold, 210 s release, 0.060520 m minimum layout clearance), parked
component (104 sources), startup (60) and near-human (60). The attempted
long-motion-v1 exited before Isaac because the inner validator still required
the legacy exact actor check set. It rejected the two additional upper-route
checks despite their passing. This was a preflight defect, not a navigation
failure. The corrected validator explicitly matches the requested route mode
and requires both additional checks. The failed launch remains archived;
fresh source-matched prerequisites precede the retry.

The retry at 992993b passed all fresh prerequisites and its own 60 stationary
updates, but was interrupted after 697 live fields at world (25.9419, 5.2294),
before the human encounter. It made only 3.34 mm progress over the final five
source seconds. All 757 archived stationary/live sources passed geometry and
verified-evidence audits; 18 raw person cues produced no verified human dynamic
labels, and there were no uncertain-human regions or recovery events. The
retained pillar surface has 5385 points, of which only 3333 have current raw
support within 10 cm; the unobserved geometry must remain occupied.

An offline positive-geometry reconstruction rejected all 35 tested constant
arcs at the stopped pose. This is not native MPC/temporal parity or global
infeasibility. A separate authored route-design survey moves just the corner
guide from x=25.8 to x=26.2 m. The original nominal route failed 12/81 dense
geometry probes in the reconstructed subset; the revised route passed 88/88.
This motivates a separate native route variant, not a claim that the old corner
or controller liveness is fixed. The human encounter, CRA, margins and priors
remain unchanged. All failure records are preserved under
`outputs/upper-service-departure/long-motion-v2/`.

The separate `--service-wide-corner` recipe changes only the authored guide
(25.8, 4.05) to (26.2, 4.05). It preserves the start, seven other guides, final
goal, actor route/trigger and every controller/perception setting. The nominal
controller still receives fixed waypoints only. This is route design around
restaurant architecture; the planned bend is not counted as CRA avoidance.
Native prerequisites and full mission validation remain required.

The retry also enables `--save-failed-planner-replay`: one original rejected
numeric request per five-second source bucket, no success snapshots or later
replacement. Serialization remains inside worker timing and does not renew
any age or make a late command usable. Snapshots are local offline diagnostics
and are never loaded by runtime. This preserves exact failure inputs if a new
stall occurs without continuously exporting every successful plan.
