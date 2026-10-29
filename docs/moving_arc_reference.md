# Online moving-arc reference experiment

The parallel-walking smoothing experiment adds a default-off
`--guidance-moving-arc-reference` option. It addresses the archived reference
selection miss documented in `mpc_plan_continuity.md`. The existing waypoint
nominal remains obstacle-unaware; CRA, field construction, lambda 1.25, physical
margins, solver budgets, source clocks and execution checks remain unchanged.
This belongs to the separate guidance-tracking ablation, not the primary
Safe-SAGE objective or a claim of exact paper equivalence.

For an actually selected current human encounter, the planner generates constant
unicycle arcs from the current state and field. At most four configured nominal
speed fractions and 33 turn rates across the existing actuator limits are
checked. The matched baseline uses one speed fraction, hence 33 arcs. No saved
trajectory, future actor position or authored human route enters control.
Quiet/unselected encounters cannot enable this search.

Each candidate must pass the original complete horizon's exact DTCBF and padded
dense geometry checks, plus the projected reference builder's analytical
half-space at every forecast command start. Useful displacement is required.
Candidates share the existing progress-plus-heading reference score. A passing
moving arc takes priority over a turn-first reference; otherwise it replaces a
moving reference only with a better score. This is an explicit reference-selection
change, not a minimum executed-speed constraint. Failed searches preserve the
original turn/wait fallback. The optimizer objective and all current-state,
new-field, freshness and analytical execution checks remain mandatory.

The unfinished-QP option is OFF. There is no cross-plan smoothness penalty,
predictive geometry, preparation while the person is quiet, reduced padding,
shorter horizon or changed acceptance threshold in this experiment.

Paired archived-request validation (`outputs/moving_arc_reference/replay-v1.json`)
recovers approximately .19960 m/s at source 15.1 s, versus .00017 m/s with the
option disabled. Both use the original QP policy. All successful results pass
full revalidation and unchanged current analytical filtering. The 15.3 s miss
and later geometry-blocked states still reject. Snapshot hashes and field-byte
immutability are checked. These are frozen-input comparisons, not a live
smoothness result or counterfactual sensor replay.

Native acceptance requires the existing independent and own 60-frame stationary
gates, then a matched live comparison against the accepted .883333 s hold run.
The original actual-speed hold target remains at most .5 s and at least 50%
reduction from the original 2.383333 s baseline, with actual arrival, nearby
curve, physical clearance, source continuity and safe stopping. Native results
are pending. Human/factor/body priors and all main/PI/long-service release holds
remain explicit.

Pre-native validation: 3,233 unit tests passed in 120.85 s. Ruff passes for the
runtime/test tree and new diagnostic helpers; changed runtime modules compile.
The new 17 tests include full-horizon/forecast rejection, input immutability,
unchanged unselected behavior, rotation, actuator/nominal bounds, turn-first
replacement and actual CLI guards before Isaac imports.
