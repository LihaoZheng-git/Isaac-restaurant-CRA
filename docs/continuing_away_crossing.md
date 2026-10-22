# Continuing-away pedestrian: separate wait-and-go fixture

The user selected a continuing-away pedestrian after both PR #167 crossing
strategies failed full route completion. Those close-endpoint failures remain
stress tests. This branch changes actor rendering/evaluation choreography only;
it does not add a global/path-reference planner or repair the old results.

The explicit `--human-continuing-away` navigation option retains the bare
left-to-right crossing at .5 m/s, from (27.8, 8.9) toward (29.8, 8.9). Within
.25 m of that target, the native agent is sent to (29.8, 12.5) in the upper aisle
without braking at the crossing handoff. The second native task may end there;
`away_endpoint` makes no human stopping claim. There is no loop, teleport, frozen
pose or return through the robot route. These coordinates and speed are authored
actor metadata, never estimator inputs. Original scene assets remain unchanged.

The robot retains its three original world waypoints (28.8, 7.8), (28.8, 8.7),
(28.8, 9.5), obstacle-unaware nominal, N6/.2 s MPC, fixed physical CRA lambda1.25,
human mass/contact and missing-factor priors, .40+.25 m body prior, current raw
occupancy, source clocks, temporal guards, exact predictive/dense geometry,
per-tick analytical CBF, independent .02 m clearance guard and safe stopping.
The separately labeled wait-and-go ablation still applies field lambda0 and
tracking weight0; this is not the primary Safe-SAGE objective comparison.

Validation first runs `restaurant_human_locomotion_isaac.py` with explicit
`--continuing-away-crossing`, robot wheels at zero and the existing fresh person/
motion sensor diagnostic. Actual root motion, articulated feet, grounding,
collider alignment and the navigation episode's square layout-clearance audit
must pass. The independent departure report requires actual transverse crossing,
upper-aisle progress to y>=11.5 and no return below11.2 thereafter. Native task
status alone cannot pass it. No human endpoint-stop test is removed from legacy
fixtures; this is a different declared scenario with a different claim.

Before robot motion the live runner must pass its own60-frame stationary gate.
Navigation requires actual three-waypoint arrival and safe stopping under the
same90 s limit,40 wall seconds per simulated second,12 s no-progress watchdog,
solver budgets and command freshness. Human/containment and observed checked-wait
gates remain mandatory. Strategy scoring uses only the transverse `move` phase
to define the walking plane; the `walk_away` leg cannot rotate that plane. Old
closest-encounter/arc scoring is retained separately and may fail a wait-and-go
trial. Numeric prior-aware scores are not fresh four-factor perception.

Implementation unit/regression, parked native and robot-motion results are
pending. No new native acceptance or main/long-service merge is claimed here.
