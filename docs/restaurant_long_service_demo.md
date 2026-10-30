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
Native validation is pending. No continuous long mission
or distributed-hazard native acceptance is claimed yet. Primary-controller,
main and PR117 holds remain unchanged.
