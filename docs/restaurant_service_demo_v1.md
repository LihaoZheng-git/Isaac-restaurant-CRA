# First restaurant service demonstration

The first version uses separate bounded trials in a newly dressed restaurant.
Carter carries a mounted tray and meal through a service aisle, encounters a
parallel walker, and encounters a person crossing the route. These are separate
recordings, not one accepted continuous long service mission.

## Results

All recorded navigation uses the robot's actual online commands. No offline
trajectory, authored desired phi, actor future route, or asset name supplies
control or physical-factor evidence.

| Chapter | Result | Navigation time | Minimum monitored clearance | Stopping drift |
| --- | --- | ---: | ---: | ---: |
| Service segment | Passed; three waypoints and final stop | 20.50 s | 1.151 m | 2.27 mm |
| Parallel walker | Passed navigation and nearby curve; yield remains | 21.73 s | 0.400 m | 2.46 mm |
| Perpendicular crossing | Passed; three waypoints and final stop | 23.45 s | 0.654 m | 2.30 mm |

The parallel bend reached **23.27 cm**. When the robot first reached a 20 cm
bend, the human and robot centers were **1.291 m apart**, at source time 14.95 s.
The synchronized plot and native frame document where the person was during
the maneuver. The plotted positions are independent evaluation truth only.

The parallel run's longest nearby actual-speed hold below 0.02 m/s was
**2.617 s**, and its longest slow interval below 0.05 m/s was 2.633 s. It does
not meet the earlier 0.5 s smoothness target. The old 0.30 s result belongs to
a different scene and must not be presented as this demo's performance.
Every interaction pause remains in the video. The changed scene is not a
matched-input experiment identifying the cause of the longer yield.

The service, parallel and crossing runs took 602.71, 1,163.19 and 1,195.06 wall seconds respectively.
Their videos use simulation timing; neither run establishes real-time control.

## What was built

- A separate additions layer contains a clearing station, rounded bin, closed
  utensil caddy and exposed-fork caddy. An existing chair is pulled out by 5 cm.
- The tray, plate and food are rigidly attached to the existing chassis. Their
  native chassis-relative bounds are approximately x[-0.16, 0.08],
  y[-0.14, 0.14], z[0.439, 0.49968] m. They remain inside the original planar
  footprint and below the declared LiDAR origin. Convex collision approximations
  are explicit. This is a mounted serving presentation, not a calibrated
  payload, manipulation, spilling or loose-food dynamics study.
- Fork, plate and food meshes reuse the audited Kenney Food Kit 2.0 CC0 assets.
  Original restaurant and robot files remain unchanged. No package, model,
  weights or dataset upgrade was introduced.

Authored names, positions and bounds are rendering and independent abort or
evaluation information. Unassessed props retain physical occupancy and labeled
factor priors; their appearance does not establish a measured hazard score.

## Control and perception scope

Waypoint nominal control remains obstacle-unaware. Online MPC and analytical
CBF checks provide intervention against current occupancy and CRA policies.
The parallel chapter uses the separate guidance-tracking/moving-arc ablation,
with lambda fixed at 1.25 and no early preparation, predictive geometry or
unfinished-QP proposal. The crossing chapter uses the separately checked
wait-and-go guidance transition. Neither uses a fixed human boundary-flux
override. These adaptations are not exact Safe-SAGE paper equivalence.
The crossing envelope used lambda 0 in nine accepted field commits and 1.25
in 225 commits; zero guidance is a checked encounter transition, not a global
zero-lambda trial. The pedestrian crosses and continues walking away. Its
wait-and-go acceptance passed, including observed motion and departure checks.

Four numeric factor scores use accepted evidence or configured priors. The
default missing-factor prior is 0.25, not measured absence. Object height is
never substituted for fall height. Human mass/contact and body extent retain
their declared prior provenance.

The existing human-envelope policy applies the maximum of the assessed track
risk and the all-unknown factor-prior risk. Thus applied envelope phi can exceed
the aggregation of the displayed track factors. The video labels both values.
This conservative geometry policy can mask a lower assessed track risk.

| Navigation evidence count | Service | Parallel | Crossing |
| --- | ---: | ---: | ---: |
| Live field commits | 204 | 190 | 234 |
| Accepted visual instances across all tracks | 4 | 26 | 78 |
| Retained-property uses | 583 | 110 | 373 |
| Configured factor-prior uses | 44,627 | 48,064 | 63,908 |
| Cumulative track assessments | 11,304 | 12,056 | 16,094 |
| Supported moving-human kinetic rows | — | 20 | 25 |
| Complete fresh generic four-factor assessments | 0 | 0 | 0 |

These counts describe prior-assisted navigation and numeric policy dataflow.
They do not establish complete fresh perception for every environmental object
or a causal advantage over a fixed-risk baseline. The primary-first-track
diagnostic differs from the all-track counts and must not replace them.

## Component evidence and retained failures

The mounted-tray bin stationary run passed with 55 accepted owned observations.
Its median conservative scores were potential 0.25, kinetic 0.18752, contact
0.12249 and thermal 0.25. Potential and thermal were configured priors. Visible
contact geometry is a proxy, not physical sharpness, hardness or hidden-surface
safety. See the native bin report and `bin-summary-v4/results.json`.

The small tabletop caddies did **not** pass their ownership/mass gates. Table
support merges, partial views and clipping remain documented in the v3–v5
attempts. No fork-versus-rounded risk ordering is claimed. Their dedicated
wheel tests remain blocked; the accepted service segment retains their unknown
occupancy and evaluates a different, already accepted component view.

The thermal inset is the earlier **separate fixed-camera RGB plume replay**.
It retains an observed source-region/identity prior and continuous cue scores.
It is not temperature or moving-camera thermal acceptance, and its offline
factor-specific phi was never supplied to these navigation trials. Preserve
the first five positive-window misses and failed registered live association.
See [restaurant_visible_plume.md](restaurant_visible_plume.md).

Other failed attempts remain intact:

- V5 parallel stationary: a 15 cm chair pull-out reduced idle-sway clearance to
  0.017757 m, below the original 0.02 m guard. V6 reduced the pull-out to 5 cm;
  the repeat measured 0.083307 m minimum over its stationary window.
- V5 crossing stationary: the optional mass-identity-retention exercise lacked
  five retained sources. Core perception and envelope checks passed. Crossing
  leaves this optional feature off; the parallel case retains its validated use.
- V6 crossing component: the CLI rejected track recovery because that component
  mode requires parallel body-identity selection. No simulation or wheels ran.
  Crossing uses its bounded configuration without that optional feature.

No confidence, association, source-age, geometry, clearance or stopping threshold
was relaxed to repair these results.

## Validation and reproduction

At runtime revision `240cb9d`, **3,242 unit tests passed in 148.57 s**, and Ruff
over `robot`, `scripts` and `tests` passed. Source/configuration/scene hashes
were retained around every native run. All selected cases passed independent
stationary checks before motion, and human cases also required component checks.
Every motion process repeated its own stationary gate.
All three selected native executions passed with unchanged runtime sources
and no traceback. Later presentation scripts and this report do not change
that validated controller/runtime revision.

`configs/restaurant_demo_v1.json` contains the explicit commands.
`scripts/scene_creation/build_restaurant_demo.py` creates new composed scene
paths; `scripts/evaluation/run_restaurant_demo.py` runs one case and stage in a
fresh output directory. Original files and failed outputs are preserved.

The runner requires a clean tree. A launcher-only or unrelated-recipe revision
can reuse a passing prerequisite only when runtime and scene hashes, and the
exact executed case command, still match. Only the output path is removed from
command comparison. Both revision IDs and full configuration hashes remain in
the audit. Legacy numeric setup reports must retain finite clearances above
the original 0.02 m limit. Isaac exiting zero does not override a traceback or
failed native report.

The generated scenes are `restaurant_demo_v6.usda` and
`restaurant_demo_human_v6.usda`, based on the existing service layers. Outputs
are under `outputs/restaurant_demo_v1/`. The stacked experimental PR preserves
the existing primary-controller, main and PR #117 holds.

## Demo artifacts and next work

The **97.9-second** assembled video uses explicit chapter cuts and native 10 fps timing. Only
the zero-command setup prefixes are omitted. Score overlays read committed
policies, label priors and reject stale display data. The manifest records
source-file hashes and chapter times.

Artifacts in `outputs/restaurant_demo_v1/`:

- [Demo video](../outputs/restaurant_demo_v1/pi-restaurant-demo-v1-final.mp4)
- [Source hashes and chapter times](../outputs/restaurant_demo_v1/pi-restaurant-demo-v1-final-manifest.json)
- [Native result summary](../outputs/restaurant_demo_v1/chapter-results.json)
- [Synchronized trajectories](../outputs/restaurant_demo_v1/navigation-results.png)
- [Parallel bend frame](../outputs/restaurant_demo_v1/parallel-nearby-bend.png)
  and its timestamp metadata.

From the repository root, the postprocessing commands are:

```powershell
python scripts/evaluation/summarize_restaurant_demo.py --output-directory outputs/restaurant_demo_v1
python scripts/evaluation/plot_restaurant_demo.py --output-directory outputs/restaurant_demo_v1
python scripts/evaluation/compose_restaurant_demo.py --playlist configs/restaurant_demo_v1_playlist.json --output outputs/restaurant_demo_v1/pi-restaurant-demo-new.mp4
```

The composer preserves existing videos, checks native acceptance and exact
frame/epoch counts, retains pauses and records source hashes. The selected
video sample frames and trajectory figure were visually inspected. Outputs
are local, ignored artifacts; the PR contains their reproducible scripts and
documentation, not uploaded recordings.

Next work is the continuous service mission with valid field coverage and
evidence lifetime, reliable tabletop instance separation, and a matched study
of the new-scene yield. Broad physical calibration and generalization remain
outside this first demonstration's acceptance.
