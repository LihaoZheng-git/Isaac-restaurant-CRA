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
rates from the original actuator limits. Each prefix must pass the original
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

65 focused guidance/search tests passed, including primary identity at weight
zero, unchanged nominal, full-horizon-only return, budget exhaustion, temporal
origin, semantic rejection despite geometric feasibility and unchanged primary
fallback. Repository Ruff/whitespace passed. Full regression and fixed-field
ORACLE route results will be recorded separately.

Trusted local numerical replay of the original 19.8 s Hawk and 19.9 s Owl
requests produced active exact-safe references where their historical greedy
references were inactive. Both MPC results converged, minimum DTCBF margins
.004150/.005769, and their first commands passed the analytical CBF unchanged.
The two search wall times were 3.86/3.26 s in this offline run; this is neither
a matched latency benchmark nor a real-time/native acceptance result. The
outputs-only simpler turn-prefix prototypes failed and remain unintegrated.
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
