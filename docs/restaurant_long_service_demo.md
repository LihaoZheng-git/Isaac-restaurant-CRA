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
