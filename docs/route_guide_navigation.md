# Route guides and checked rear-pass evaluation

The user approved intermediate waypoints as route guides, while retaining
mandatory final-goal arrival and stopping. This is an opt-in navigation
variant. The strict checkpoint controller, existing accepted native stop-and-go
result and fixed-lambda primary remain available and unchanged by default.

## Nominal contract

`WaypointControllerConfig.intermediate_waypoints_as_guides=True` enables the
new path. It takes only pose, a configured route and an optional route start.
There is no obstacle, actor, risk, segmentation or field input. MPC and the
analytical CBF still supply all online safety intervention.

- The nominal target looks 1 m ahead along the configured polyline from the
  clipped projection on the active segment. It may look past an intermediate
  anchor, so it does not demand a return toward a nearby point during a detour.
- An intermediate guide advances when reached within the existing tolerance,
  or when its incoming-segment end plane has been passed within a configurable
  1.5 m cross-track route-association scope. This is a navigation assumption,
  **not** a safety margin, free-space observation or obstacle-geometry prior.
- The final point can never advance through the plane rule. It still requires
  Euclidean distance <=.10 m; the existing downstream actual-stopping checks
  remain mandatory. Passing the final goal from the side does not complete it.
- `current_waypoint` remains the active configured anchor. In guide mode,
  `WaypointCommand.target_waypoint` is the pose-only lookahead target, while its
  index records the active anchor. Callers must not label guide progress as
  arrival at every original checkpoint.
- Prediction clones the original route anchor and current progress. It never
  commits predicted advances to the live controller. Reset restarts progress.
  If no route start is supplied, the first observed pose supplies the start.

The ORACLE benchmark records original anchors, completion policy, lookahead
targets and guide advances separately. Its final-goal/clearance/analytical
acceptance is unchanged. The live Isaac demo exposes `--waypoint-mode guides`,
`--route-guide-lookahead` and `--route-guide-max-cross-track`; defaults remain
checkpoints. Encounter route direction uses the configured segment endpoints,
not the moving lookahead target. Its report labels the navigation policy.
That CLI support is **not native motion validation** of the new mode.

## Rear-reference behavior diagnosis

Guide semantics alone did not fix the original late encounter. Keeping onset
11.5 s, original anchors 2.2/3.1/3.9 and the 1.4 m maneuver-lead preference gave
two route arrivals and one mirrored `analytical_emergency` failure; all three
failed the curved strategy. The failed right-mirror clearance was .465790 m:
positive clearance does not repair an analytical failure. The matched guide/
stop-and-go variant reproduces that failure, while the fixed-lambda primary
reaches that goal. All original reports remain intact.

An explicitly declared .8 m lead trial reached all three goals but still failed
all walking-window curve gates (.01995/.02131/.01995 m). With onset 8 s and
original anchors, the original 1.4 m lead behavior also reached all three goals
but failed the curve gates (.08062/.08453/.08062 m). These are failures of the
proposed curved behavior, not successful bypasses inferred from arrival.

The traces exposed another behavior interaction: as the robot started turning,
the shrinking forward lead could cancel the rear preference before enough
lateral translation accumulated. A separate optional current-heading gate
(`existing_turn_heading_radians=.2`, about 11.5 degrees) permits an already
trailing turn to continue while heading still has a forward component. A turn
toward the pedestrian does not qualify. This consumes current robot pose only;
it creates no retained identity, source clock, actor-phase input or physical
evidence. It is a maneuver preference, not proof of clearance. The default is
`None`, preserving the previous reference behavior.

All reference candidates still need the complete original-horizon exact
temporal DTCBF, dense geometry, future mapped analytical inequalities, >=.01 m
translation and unchanged first-command analytical acceptance. Normal risk,
flux, unrelated occupancy and all source clocks are preserved. The separate
rear mode uses applied lambda1.25 and checked wait uses lambda0 only on a current
qualified perpendicular human. This is not the primary MPC selection objective.

The single late .8 m lead/.2 rad continuation probe still failed the curve gate
(.01995 m during walking), despite arrival in22.8 s and .367377 m clearance.
Continuing the turn alone does not repair the original late encounter.

## Matched early-crossing comparison

The final matched comparison keeps **original guide anchors** 2.2/3.1/3.9,
onset8 s, N6/.2 s, QP1000/rescue20 and the same physical geometry and risk. It
uses guide mode in every variant. The rear variant uses the optional .2 rad
continuation gate, original 1.4 m lead and all checks above. This is a different
actor-timing fixture from the failed11.5 s case, not a retroactive repair.

The nine matched trials all pass route arrival, independent clearance and
analytical checks. The rear variant passes all three curved-strategy gates:

| Historical case key | Rear arrival | Rear walking-window arc | Rear minimum clearance | Stop-and-go arrival | Fixed-primary arrival |
| --- | --- | --- | --- | --- | --- |
| cross-left-right |19.8 s |.302250 m |.605440 m |26.2 s |19.6 s |
| cross-right-left |19.8 s |.302931 m |.587563 m |26.2 s |19.6 s |
| cross-then-parallel |19.8 s |.302250 m |.605440 m |23.0 s |19.6 s |

The fixed primary has zero lateral excursion and .601906 m minimum clearance;
stop-and-go has1.080000 m minimum clearance. The rear variant is one .2 s
planning interval slower than the fixed primary, not a speed improvement.
Walking-plane crossings occur at16.77–16.85 s, after the8–12 s transverse window.
The matched late baseline also preserves the right-mirror analytical failure;
all three late fixed-primary runs arrive in22.4/22.6/22.4 s.

Final matched results are recorded in
`outputs/paper_flux/route-guide-early8-turn02-matched-v1/summary.json`.
Interpret complete controller variants separately; a curved path is not by
itself evidence of faster travel or a causal CRA advantage. The robot reaches
the original walking plane after the transverse walking window, so even a
passing curve gate does not establish close timed passage beside a still-near
perpendicular pedestrian.

The fixture supplies ORACLE identity, geometry, risk and velocity. There is no
RGB/LiDAR acceptance, physical wheel/braking validation, real-time guarantee or
exact paper equivalence. No new Isaac process or wheel trial was launched:
the original late-case failure blocks promoting this strategy to native motion.
Native four-factor priors, zero-complete-fresh-four-factor finding and existing
main/long-service/PR117 holds remain unchanged.

## Validation and artifacts

All **2,590 unit tests passed in88.53 s**. Focused tests cover off-route guide
passage, final-goal rejection/arrival, route turns, bounded association,
zero-length segments, prediction/reset, legacy checkpoints, explicit benchmark
metadata, both trailing directions and rejected wrong-direction turn reuse.
Native CLI help and invalid-distance rejection were exercised without starting
Isaac. The smoke check also found and fixed a pre-existing unescaped percent
sign in help text; no runtime policy was changed by that help correction.
Ruff over `robot`, `tests` and `scripts`, Python compilation and diff checks
passed. No environment or dependency change was made.

Reproduce an ORACLE comparison in a fresh output folder:

```text
python scripts/evaluation/compare_checked_rear_pass.py --output outputs/guide-comparison --waypoint-mode guides --actor-onset-seconds 8 --rear-existing-turn-heading .2 --controller rear stop-go primary
```

Default checkpoint mode, onset11.5 s, 1.4 m lead and disabled heading continuation
remain. Manifests retain Git revision, source hashes and declared configuration.

Ignored local artifacts retain all earlier source-hashed probes:

- `route-guide-original-onset-rear-v1/`: late guide/1.4 m lead failures.
- `route-guide-original-onset-lead08-rear-v1/`: late .8 m lead attempts.
- `route-guide-early8-legacy-anchors-rear-v1/`: early guides without continuation.
- `route-guide-original-lead08-turn02-v1/`: failed late continuation probe.
- `route-guide-early8-turn02-v1/`: first bounded positive continuation probe.
- `route-guide-early8-turn02-matched-v1/`: final matched nine-run comparison.
- `route-guide-original-onset-baselines-v1/`: matched late baseline comparison.
- `route-guide-navigation-v1.png`: recorded early paths and goal progress,
  labeling actual mirrored velocity directions and the retained late failure.

Historical virtual case aliases have mirrored robot-frame wording; preserve
their keys and label figures by actual observed velocity vectors.
