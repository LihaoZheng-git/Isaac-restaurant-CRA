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
(28.8, 9.5), obstacle-unaware nominal, N6/.2 s MPC, fixed physical CRA lambda 1.25,
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
upper-aisle progress to y>=11.5 and no return below 11.2 thereafter. Native task
status alone cannot pass it. No human endpoint-stop test is removed from legacy
fixtures; this is a different declared scenario with a different claim.

Before robot motion the live runner must pass its own 60-frame stationary gate.
Navigation requires actual three-waypoint arrival and safe stopping under the
same 90 s limit, 40 wall seconds per simulated second, 12 s no-progress watchdog,
solver budgets and command freshness. Human/containment and observed checked-wait
gates remain mandatory. Strategy scoring uses only the transverse `move` phase
to define the walking plane; the `walk_away` leg cannot rotate that plane. Old
closest-encounter/arc scoring is retained separately and may fail a wait-and-go
trial. Numeric prior-aware scores are not fresh four-factor perception.

## Validation at 84e503d

All 2,467 unit tests passed in 86.99 s; Ruff passed for robot/tests/scripts.
Source was clean and frozen at 84e503d70dedeac32964586f734c76d2169c88e8
through both parked diagnostics and the robot trial. The later changes only
correct report wording and record these outcomes; they do not change control.

| Separate native trial | Outcome |
|---|---|
| Original distant parked view, `continuing-away-parked-v2/native` | FAILED. Actor crossing/departure/articulation/grounding/geometry passed; zero-command robot drift was .022337 m against .02 m. Fresh stationary median speed .127371 m/s exceeded .12; only two moving fresh rows against five. Preserve this failure. |
| Close parked view, `continuing-away-parked-v3/native` | PASSED. Nine stationary/ten moving fresh rows; median stationary speed .021527 and moving velocity error .149332 m/s. Actual moving speed .520357 m/s, layout clearance .283841 m, collider alignment <2.87e-8 m, zero wheel commands, 157 decoded video frames. The different parked view is explicit; the robot navigation start remains original. |
| Original robot route, `continuing-away-wait-v3` | PASSED own 60-frame stationary gate, actual 3/3-waypoint arrival in 24.10 navigation seconds, and safe stopping. No traceback; source assets unchanged. Minimum independent clearance including braking .470310 m, stopping drift .002336 m, minimum analytical margin .111236. |

The launcher-only `continuing-away-parked-v1` was rejected before simulator startup
because its output directory already existed. It is not a native test; the retry
uses a fresh native subdirectory and preserves the failed invocation.

Robot navigation committed 240 live fields, 204 while translating, with 15,830
all-track assessments and 84 accepted all-object visual instances. **Zero**
assessments were complete fresh four-factor perception. Human-specific motion
validation passed 29 supported moving / 19 simultaneous robot-human rows, median
world-velocity error .067952 and relative-speed error .014260 m/s. The generic
`simultaneous_robot_actor_kinetic_updates=0` counter concerns the other scripted
actor fixture, not native humans; use the explicit human report for these counts.

All 240/240 active human-envelope rows passed: maximum center error .179969 m,
minimum containment margin .070031 m under the unchanged .25 m uncertainty.
The configured .40 m body radius, mass 70 [45,100] kg, contact .1 and unavailable
potential/thermal .25 priors remain priors, not measured physical properties.

Three accepted exact/analytical checked-yield field sources occurred during
transverse walking; the longest observed wait was 2.75 s. An independent ORIGINAL
OPTIMIZER-REQUEST-clock join verifies 14 physics command rows / .233333 s of executed
checked-yield zero commands during walking. The remainder of the wait included
freshness/invalid-plan safety stops: this is not 2.75 s of continuously executed
intentional yield. Preserve the stop records and the accepted-plan/actual-command
distinction. The audit is `verified-wait-command-audit-v2.json` in the new folder.

The robot crossed the transverse walking plane at source 29.585631 s, with
-.671928 m projection behind the actor along its transverse direction. Final-goal
error after braking was .096589 m. The older closest-encounter/curved metric
remains FAILED (only .000181 m lateral arc, -.019338 m behind at closest encounter).
This accepted behavior is **wait-and-go**, not a pronounced curved bypass.

The unchanged functional run used applied field lambda 0 / tracking 0, human
b=-1.7, N6, QP 1,000/rescue 20 and .025 pacing. Physical CRA retained lambda 1.25.
One generic limitations string in the original report says fixed lambda 1.25;
its explicit `fixed_lambda=0`, `physical_CRA_lambda=1.25` and ablation flags are
the actual configuration. A subsequent report-only correction makes that text
conditional. The original report is retained unchanged, with a separate metadata
correction record. No primary fixed-lambda comparison or causal CRA advantage
is established by changing the actor scenario. No main/long-service merge or
PR117 fresh moving-view four-factor acceptance follows.

The inspected figure is `outputs/paper_flux/continuing-away-wait-result-v1.png`;
the native video is `continuing-away-wait-v3/human-navigation.mp4` (321 decoded
frames at 10 Hz). The native run took 1177.75 wall seconds including startup and
stationary validation; navigation itself took 1023.78 wall seconds. This is
paced development, not real time. The close-endpoint stress tests remain failed.
