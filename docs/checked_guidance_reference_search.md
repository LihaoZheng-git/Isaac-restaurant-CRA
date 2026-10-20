# Checked local reference search (separate ablation)

The Owl/.25 m-prior trial recovered fresh human observations and validated
body coverage, but still waited through the crossing. The greedy reference
builder can encounter a configuration boundary later in its original horizon
and reject the entire reference. See [the native results](native_camera_coverage.md).

`--guidance-reference-search-width 4` or `8` enables a bounded fallback only
when the existing greedy references fail in `planar_forecast_velocity` mode.
Default zero preserves every legacy caller and the primary controller.
`--guidance-reference-search-candidates` now defaults to 4,800 (initial 3,600). This is part of
the already separate guidance-tracking ablation, not a primary objective change.

The search expands zero/half/preferred forward speeds and five bounded angular
rates from the original actuator limits, plus gentle +/- .1 rad/s steering
while translating. Pure-turn caps are unchanged. Each prefix must pass the original
exact temporal DTCBF and dense padded geometry checks, plus the mapped
analytical guidance condition at its new interval. Forecast time remains
relative to the original initial state. A goal-distance/heading heuristic and
bounded position/yaw diversity select at most eight prefixes per depth; those
are numerical search choices, not physical safety evidence or a new theorem.

Only a complete original-length horizon can become a reference. An exhausted
budget may return an already completed exact-safe horizon; a shorter prefix
is never executable. Without a useful full horizon, retain the unchanged
primary fallback. A selected reference uses the existing tracking objective
and existing exact-safe reference warm start. The nominal remains the same
obstacle-unaware waypoint command. CRA evidence, priors, fixed lambda 1.25,
field/optimizer/execution source clocks and all current-state revalidation and
per-step analytical CBF checks remain unchanged. Search time counts against
the original worker/candidate age; a late result is still rejected.

This produces a guidance-feasible path candidate, not exact Laplace-flow
tracking, enforced passing side, global convergence or paper equivalence.
The configured side remains soft; the actual arc/behind/arrival/stopping gates
must independently pass in native simulation. Real actuator response and
moving-field changes can still invalidate a candidate.

## Validation before wheels

82 focused guidance/search/audit tests passed, including primary identity at weight
zero, unchanged nominal, full-horizon-only return, budget exhaustion, temporal
origin, semantic rejection despite geometric feasibility and unchanged primary
fallback. The latest full unit suite passed **2,273 tests** in 80.64 s, with
repository Ruff/whitespace checks. Fixed-field ORACLE routes and native tests
are separate gates; unit success alone never enables a wheel acceptance claim.

The initial search version's trusted local numerical replay of the original 19.8 s Hawk and 19.9 s Owl
requests produced active exact-safe references where their historical greedy
references were inactive. Both MPC results converged, minimum DTCBF margins
.004150/.005769, and their first commands passed the analytical CBF unchanged.
The two search wall times were 3.86/3.26 s in this offline run; this is neither
a matched latency benchmark nor a real-time/native acceptance result. The
outputs-only simpler turn-prefix prototypes failed and remain unintegrated.
Those references predate the later productive-tail and soft-ranking changes;
they are historical dataflow diagnostics, not validation of the latest search.
Preserve all previous failed native records and main/long-service holds.

## Rejected command-only variants

At 6e4c200 all 2,267 unit tests passed (75.49 s), but both fixed-field ORACLE
routes in `virtual-v6-checked-search` failed the original four-second stationary
solution rejection at 9.75 s. Exact geometry, CBF and MPC margins passed;
arrival did not. No wheels were commanded. This variant is not native-ready.

The next numerical proposal added gentle steering while translating and
preserved lateral alternatives before one-cell displacement, retaining the
old pure-turn caps. Its 4,800-candidate bound remained explicit. 65 focused
tests passed, but both `virtual-v7-checked-search-lateral` routes still failed
the same stationary-solution rejection at 10.95 s (source hashes in manifest).
The positive curve did not establish arrival. Preserve this second failure.
Move-then-stop references can still lead into a local trap; reference quality
needs correction before any new wheel trial. The primary controller and all
legacy defaults remain unchanged; main/long-service remain held.

## Productive-reference correction

The next revision rejects idle gaps and stationary terminal references outside
the goal region. Turn-first references may still use the existing turn cap
before translating; stopping at the goal is valid. The primary may always
wait safely. These are auxiliary-reference quality gates, never executable
minimum-motion constraints, altered safety checks or changed waypoint arrival.

69 focused tests and all **2,271 unit tests** passed (75.97 s), with Ruff.
Both `virtual-v8-checked-search-quality` ORACLE routes arrived with geometry,
CBF and MPC checks: 23.25/22.45 s. Their observed negative-y excursions were
-.69736/-.66682 m for configured left/right fields. Thus arrival passed but
the search selected the same flank in both cases; do not claim enforced side,
native crossing or exact guidance-flow tracking. The soft guidance ranking
needs review before wheel tests. Keep both earlier rejected variants.

## Soft lateral guidance ranking

The latest fallback adds the initial field guidance's lateral component relative
to the current goal direction to the numerical ranking. Its normalized
preference is in [-1, 1]; the score uses the existing lookahead length and the
preferred full-horizon travel distance as scales. This favors the indicated
flank without introducing a hard side constraint. The full CBF normal is never
treated as a travel heading: its radial component can point away from the goal.
Both lateral exploration reserves and every exact prefix/final check remain.

Two symmetric configuration-barrier tests produce opposite, exact-safe curved
references when the guidance preference reverses. The current trusted replay
(`checked-search-quality-replay-v2.json`) also accepts complete productive
references on both original Hawk/Owl input exports: DTCBF minima .010824/.008825,
unchanged first-command analytical CBF margins .715286/.637034. Those are
offline numerical inputs, with original timestamps and no execution; 5.02/3.24 s
total compute times are not a matched latency comparison or real-time claim.

Both `virtual-v9-checked-search-guidance` ORACLE routes reached every waypoint
and passed executed geometry, CBF and MPC checks: left 22.85 s with +.677047 m
lateral excursion; right 23.05 s with -.696301 m. These use the same authored
fixed geometry/phi and the original 45 s/arrival/stationary-solution bounds.
The manifest records the tested helper hash before the documentation commit.
This resolves the command-only same-flank result in that symmetric fixture;
it is not enforced side or native human bypass acceptance. A new Owl/.25 m-prior
trial must repeat 60 stationary frames before its bounded motion diagnostic.

## Native 13 s encounter diagnostic (ff4f55b)

`native-checked-search-13s-v1` passed its 60-frame zero-wheel human/envelope
gate. All 199 live independent envelope rows passed the approved .25 m centroid
uncertainty PRIOR (total radius .65 m); worst containment margin .069605 m.
The original failed .20 m trial remains unchanged. This supports the wider
prior in this bounded trajectory, not a general human uncertainty calibration.

Overall acceptance **failed**: intentional 20 s duration limit, 1/3 waypoints,
-.011057 m walking-window left arc, -.005773 m closest-encounter behind
projection, 1.526290 m goal error and zero simultaneous robot/human kinetic
updates. Monitored full-sweep clearance .434759 m, stop drift .000222 m and
minimum analytical CBF margin .124087 do not establish arrival or bypass.
The 199 fields/148 translating commits took 851.0 navigation wall seconds.
Selection/provenance/fixed-lambda checks and nine supported moving-kinetic
rows passed. Keep 67 accepted visual instances across 13,759 assessments,
54,744 configured factor-prior uses, 208 property-prior uses and zero complete
four-factor assessments separate from numeric control-policy completeness.

The exact-safe fallback was exercised natively, but a late accepted reference
alternated +turn/idle/-turn/idle for several intervals. The two-interval activity
test considered these active despite near-zero net motion. Command auditing
also retains 287 waiting-for-plan, 29 new-field-invalidated and 28 stale-field
stop frames. The launcher exited zero despite the full acceptance exception;
the report, not process exit, establishes failure. No main/long-service merge.

## Rejected longer-window activity screening

An unintegrated follow-up screened net translation/rotation over four- and eight-interval
rolling windows as well as the original two-interval check. This rejects
oscillating turn cycles while retaining sustained useful turn-first activity
and goal stopping. It is solely auxiliary-reference quality; primary safety
waits, exact constraints, source clocks and the nominal remain unchanged.
Although 24 focused and 2,276 unit tests passed with Ruff, the left virtual
route developed a receding-horizon turn reversal: unchanged position at
6/8/10/12/14 s with alternating short turns. The zero-wheel diagnostic was
terminated at this observed regression, before its 45 s bound; no completed
route result is claimed. The patch is retained in
`outputs/guidance_progress/rejected-cycle-screening.patch` and was removed
from runtime. Source/tests exactly match ff4f55b again (2,273 passing unit
tests); the 2,276 result belongs only to that rejected proposal.

The next separately reported encounter brings the human crossing forward
from 13 to 11 s, giving observed motion earlier in the approach. It uses the
same restored controller, .25 m centroid PRIOR, .025 pacing, 60-frame
stationary gate and all original exact/freshness checks. The original full
90 s bound, 12 s progress watchdog, arrival and stop gates will be used.
This is a changed encounter fixture, not a matched causal controller comparison
or a retroactive pass of the original 13 s case.
