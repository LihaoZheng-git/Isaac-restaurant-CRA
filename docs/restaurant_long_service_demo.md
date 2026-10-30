# Continuous restaurant service demo development

The requested demo uses the existing restaurant's left pickup/front aisle and
right dining aisle, with a mounted tray and a render-only follow camera. The
candidate nominal route is 30.878 m. The authored raster screen is in
`outputs/restaurant_long_route_review/2026-10-29/`: its smallest conservative
all-yaw disk allowance is 0.092 m. This is an offline layout screen, not motion
acceptance. The earlier V1 video contains three separate native trials.

## Translating local field

`--rolling-local-field` keeps the original fixed world anchor and yaw for
tracking, nominal waypoints and MPC states. Only the 7 x 6 m window bounds move,
in whole one-metre steps. Geometry, policy, inflation, resolution, field solvers,
source clocks and command freshness limits remain unchanged. The previous
actual field is aligned by integer cell displacement. Its real boundary change
is included in the temporal difference; it is not re-solved using new boundary
conditions. Temporal smoothing uses only aligned valid history.

New edge cells lack a consecutive-source derivative. They remain explicitly
invalid for scalar AND batched control queries until a second actual source
covers them. No unknown derivative becomes a usable zero. The current acquisition
pose must have valid overlap. Asynchronous field proposals stage the grid,
tracking and evidence together until admission. An unpublished gap-recovery
baseline includes its original grid origin and timestamps.

The property retention limit remains 120 s. Expired properties can be acquired
again only from accepted new observations; otherwise factor priors apply.
Positive occupancy memory is separately labeled and static-scene only.

## Validation sequence

1. World-coordinate alignment, source-gap/edge rejection, scalar/batched CBF
   sampling, staged rollback and existing regressions.
2. Fresh stationary gate and short service motion with the translating window.
3. Long-route stationary sensor gate, then route-only continuous navigation.
4. Validate each distributed prop/human from its actual encounter view before
   introducing dynamic episodes. Existing unresolved caddy/plume cases remain
   unresolved; scenery is never perception evidence.

The long-route startup gate is distinct from the remote FoodBowl component gate:
one currently observed track needs at least three fresh owned RGB/LiDAR mass,
contact-proxy and motion assessments, including one within 0.35 s. Every track
still receives numeric evidence/prior scores. This is route-coverage development,
not proof that every restaurant asset has complete fresh four-factor evidence.

## Presentation

`--presentation-follow-camera` uses actual robot pose only. The virtual camera
sits 1.8 m behind the smoothed pose at 2.5 m height and looks 0.8 m ahead at
0.55 m height. Pose smoothing has a 0.35 s time constant. Camera transforms are
updated before rendering, and their source epochs accompany video samples.
This camera never supplies RGB, labels, geometry or commands to the controller.
The original fixed overview remains the default. Native validation must inspect
ceiling/column occlusion; rendering quality does not establish navigation success.

## Current status

The full unit suite passes: 3,257 tests in 141.09 s; Ruff passes. A focused
80-test run also passes. Initial pytest attempts failed temporary-directory
setup; creating the dedicated output parent resolved that environment issue.
Native short rolling validation at `30651c5` passed its independent and own
60-frame stationary gates, then all three service waypoints with 209 live fields,
1.150603 m minimum clearance and 0.002235 m stopping drift. Sources remained
unchanged. The follow camera was visually inspected at the starting pose.

The first long-route attempt was rejected by the legacy 120 s CLI duration cap
before native startup. The long-service selector now permits an explicit bounded
300 s trial; other callers retain 120 s. This does not change the 120 s property
expiry, source-age limits or command leases. Preserve the rejected launch.
The default narrow-camera long-start gate failed: 32 accepted mass instances
but only one resolved contact observation. The existing `front_owl` camera passed
the same unchanged 60-frame gate; no model, confidence or acceptance threshold
was changed. Preserve both attempts.

At `a7b87a9`, `native-long-motion-owl-v1` then PASSED its own stationary gate and
all eight route waypoints: 158.350 s navigation, 1,353 live fields (1,351 while
translating), minimum independent clearance 0.310740 m, minimum CBF margin
0.236671 and stopping drift 0.002264 m. The launcher recorded no traceback and
unchanged runtime/scene sources. The 120 s property expiry remained unchanged.
This establishes one continuous route-only mission, not distributed-hazard
acceptance. All 1,662 follow-camera frames are saved at 10 fps; start, front
bar aisle, turn and destination frames were visually inspected. The mounted
meal is presentation geometry, not payload-dynamics validation. Runtime is
paced simulation (3,750.6 wall seconds), not real time.

Native evidence and video: `outputs/rolling_service_field/native-long-motion-owl-v1/`.
The long-route recipe now records the actually validated front_owl selection.
Human, utensil and live thermal encounters are not included in this run. Primary-controller,
main and PR117 holds remain unchanged.

The complete append-only history contains 70 fresh accepted visual instances
across 93,858 current object rows. Contact has 959 evidence-supported rows and
kinetic 1,217; potential and thermal remain configured priors throughout this
route. No row has four evidence-supported factors. Current-track phi spans
0.591428..0.854707. There were 34 distinct field origins; admitted field ages
were 0.133333..0.283333 s. Numeric completeness is not complete perception.

## Next bounded parallel-human integration

### Latest native result and remaining decision

At`cacbb56` all four fresh prerequisites passed:180s idle/release, route-start,
moving-human component and near-view stationary gates. The full suite passed
3,269tests in118.21s;65focused tests and Ruff passed. Native
`human-long-motion-v3` then FAILED `nearby_track_lost` at83.016667s navigation,
676live fields and2waypoints. It stopped atworld(24.430313,4.147136), before the
person's trigger. Minimum independent clearance.442919m, minimum CBF margin
.177991, stopping drift.004666m. The human's held idle drift was exactly zero.
Report SHA256:`63e03e7eab1c279492bc77d726837a76d4190d3cd6df61ab783809e9ac34b7f9`.

The new correspondence path WAS exercised:30many-to-one events passed full
current-surface coverage, maximum distance.073510m. The final fragment2045 was
different: its8old returns had nearest current returns.158283..168582m away,
outside the original.10m correspondence gate. The guard correctly left this
gap unresolved. Do not widen the threshold or count the partial route as arrival.
There were39fresh accepted visual instances across54,663object rows and zero
complete fresh four-factor rows. Original priors and all source clocks remain.

A separate sensor-built stationary-furniture memory layer for the mixed scene
has been proposed to the user as a geometry-uncertainty research choice. It is
NOT implemented or approved yet. Moving-human geometry must remain live and
unresolved human associations must retain their stop behavior. Do not silently
enable the existing static-scene memory globally. The accepted continuous
route-only video is ready; the earlier human clips are separate recordings.
Primary/main/PR117 holds remain. No native process is left running.

At5a8ab20 the180s actor hold/release gate PASSED (zero idle drift,3.470465m native
travel,.256891m minimum actor clearance). Refreshed startup, human component and
near-view stationary gates also PASSED. Full unit suite:3261passed/130.28s; Ruff
passed. The first actor-only launch failed Kit's legacy torchvision import path;
using the existing pre-Kit import order repaired it without changing packages.

`human-long-motion-v2` nevertheless FAILED at14.533333s navigation: the nearby
continuity guard retained furniture track250 after it and308 merged into21.
Both old surfaces remained fully observed:767 and12returns, maximum current
correspondence distances.044899/.019467m. The old one-to-one rule rejected this
many-to-one correspondence. No person had started walking; the held idle stayed
fixed. Stop drift was.005151m with.532680m stopping clearance. A final audit
exception compared a settled-anchor goal against the authored nominal endpoint;
the reporting fix now transforms the entire nominal route with the control
anchor, without relaxing the curve or goal check. Original failed record stays.

A long-service-only option now admits many-to-one geometric continuity only
when EVERY child has a fresh original co-observation and its WHOLE positive
surface is uniquely covered by current returns. Original .35s source-age/.10m
distance/min3return gates remain; partial, ambiguous, unclustered and stale
coverage reject. There is no identity/factor transfer, static occupancy memory,
new geometry, free-space claim or source renewal. Archived two-scan replay and
synthetic rejection tests precede fresh native validation; replay alone is not
navigation acceptance.

At225290b, startup, native human component and near-view stationary gates passed.
The first full mission then FAILED at24.78s navigation, before the encounter:
native idle root motion drifted about26cm into a chair's independent clearance
guard (.019682m, required>.02m). Robot progress was normal; neither tracking loss
nor phi caused the exception. Preserve `human-long-motion-v1`, including its
incomplete route and missing final stopping audit.

Long-service now holds the settled native agent with `set_enabled(False)` until
the actor trigger, then enables it before `move_to`. No teleport, altered robot
policy or relaxed geometry gate is used. A separate180s actor-only hold/release
audit disables the robot in its unsaved diagnostic session; all original actor
collider/ground/layout checks remain. That audit plus current-source startup,
near-view and human component gates are mandatory before new long-human wheels.

The route presentation overlay's raster orientation was corrected: PNG row0 is
maximum worldy. Authoritative presentation is
`outputs/rolling_service_field/restaurant-long-service-route-v2.mp4`; control and
native footage were unaffected. It trims initial setup only, retaining every
navigation frame and pause.

`configs/restaurant_long_service_human_v1.json` uses the same existing native
adult and table layout. The actor waits at (29.8,10.5), then walks south to
(29.8,7.0) when robot world y reaches9.0. This changes actor timing only; all
robot avoidance uses current RGB/LiDAR, CRA and online filtering. The existing
moving-arc guidance/route-guide ablation remains separately labeled, fixed
lambda1.25; no early-preparation or predictive-geometry option is enabled.
The main recording follows the actual robot through the continuous route.

Static occupancy memory is OFF. Existing .15s positive occupancy, mass-only3s
retention, track-loss coverage, body-envelope and freshness gates stay unchanged.
The distant start cannot establish human perception: the launcher and runtime
require current-source near-human stationary and native component records in
addition to the route-start gate. Nearby curve evaluation uses distance from
the configured nominal polyline, so its planned corners do not count as
avoidance. Human integration is pending native validation; failure of the short
memory in the front aisle must be preserved, not repaired by retaining human
positions as static occupancy.

The first new long-start zero-wheel run (`b8e554c`, human-long-stationary-v1)
passed owned mass/contact/motion readiness but failed the legacy requirement
for five retained human-mass sources, because the person is outside this view.
Keep that failed result. The distant-start audit now requires all identity,
source-age and confidence checks but records exercise coverage as untested;
five retained sources remain mandatory in the separate near-view gate and
complete navigation audit. Any emitted human envelope is still independently
audited at the distant start. This changes test placement, not runtime priors,
source clocks, identity association or the required human evidence.
