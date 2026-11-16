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
Candidates share the existing progress-plus-heading reference score. Search now
runs only when the existing reference is absent or its first forward command
does not exceed the existing intervention tolerance. A passing moving arc takes
priority there; an already-translating reference is preserved exactly. The
first trial also re-ranked moving references, with the regression below. This
is an explicit reference-selection
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
curve, physical clearance, source continuity and safe stopping. The narrowed
retry passed the bounded gates below. Human/factor/body priors and all main/PI/long-service release holds
remain explicit.

Pre-native validation: 3,233 unit tests passed in 120.85 s. Ruff passes for the
runtime/test tree and new diagnostic helpers; changed runtime modules compile.
The new 17 tests include full-horizon/forecast rejection, input immutability,
unchanged unselected behavior, rotation, actuator/nominal bounds, turn-first
replacement and actual CLI guards before Isaac imports.

## First native trial: navigation passed, smoothing regressed

Frozen source `1fe4185` passed independent and own 60-frame stationary gates.
The matched `motion-v1` reached all three guides/final goal in 21.15 simulation
seconds, with .239189 m clearance, .002439 m stopping drift and .249465 m nearby
bend. Source hashes remained unchanged. However, the actual nearby translation
hold increased to **2.366667 s**, versus the accepted **.883333 s** baseline;
the .5 s and 50% reduction criteria fail. Keep this failed smoothing result.

Twenty recorded requests searched arcs, seventeen selected and accepted them,
and 143 issued translating command rows join those requests by original source
clock. Five saved selected-arc requests independently pass full reference and
optimized-horizon revalidation and unchanged analytical filtering. These checks
do not turn a longer physical pause into success. Computed human CRA equaled
applied risk (.485120..683594); zero fresh complete four-factor updates remain.

The first selected baseline request at source time 11.7 used .1162 m/s while turning;
the broad arc ranking replaces it with .20 m/s. That changes the approach before
the diagnosed delayed-reference decision. The narrowed retry therefore preserves
every already-translating reference, searching only on missing/delayed references.
This removes an unnecessary intervention; it does not prove the sole cause of
the regression or guarantee smoothness. The full native gates must be repeated.

An additional offline 343-candidate two-stage arc probe found no full-check
witness on the accepted baseline's 14.2 or earlier candidate's 15.3 state. It is
not integrated or an infeasibility proof. Existing accepted demo/release holds
remain. Original v1 report SHA-256:
`c1f3b78d3ae42c8e6ebc652fc4aa8d854d2eaa582f7e6487dfab748cadc43408`.

Narrowed pre-native replay (`replay-v2.json`) preserves the original commands
exactly at 14.8/14.9, still recovers the checked moving command at 15.1, and keeps
all later rejections. The added regression test requires exact command-sequence
parity when a reference is already translating. All 178 focused controller tests
pass in 18.59 s; these checks preceded fresh native validation below.
The full narrowed-source suite passes 3,234 tests in 118.16 s; Ruff and compilation
pass. The accepted baseline's first selected and later already-moving requests
also retain identical first commands in `accepted-baseline-replay-v2.json`.

## Narrowed native retry: navigation and smoothness passed

Frozen source `7a1e1fc23475b5d60346d7c2eabaeddc80e6f16d` passed a fresh
independent 60-frame stationary gate (220.94 wall seconds) and its own 60-frame
gate before navigation. `motion-v2` then passed the original navigation,
human-motion, body-envelope, source-continuity and stopping checks. The run took
1,027.03 wall seconds at the original paced simulation setting; it is not a
real-time result. Runtime revision, clean tree and every source hash remained
unchanged throughout each native process.

| Metric | Accepted baseline | Broad v1 (failed smoothing) | Narrowed v2 |
|---|---:|---:|---:|
| Longest nearby actual-speed hold below .02 m/s | .883333 s | 2.366667 s | **.300000 s** |
| Longest nearby slow interval below .05 m/s | .933333 s | 2.433333 s | .350000 s |
| Navigation simulation time | 19.650000 s | 21.150000 s | 18.833333 s |
| Independent minimum clearance | .440801 m | .239189 m | .474256 m |
| Final stopping drift | .002417 m | .002439 m | .002451 m |
| Largest bend while human walking within 2 m | .237194 m | .249465 m | .234823 m |
| Route guides/final goal | 3/3, reached | 3/3, reached | 3/3, reached |

The .30 s hold is 66.0% shorter than the accepted .883333 s baseline and 87.4%
shorter than the original 2.383333 s target baseline. It satisfies both original
criteria: at most .5 s and at least 50% reduction against the original target.
A brief physical yield remains; this does not establish pause-free behavior.
At the largest nearby bend (source time 16.083333 s), robot and human centers were
**1.231460 m apart**. The synchronized pair is plotted and an original native
video frame is extracted. Human walking begins at 11.3 s; the first 2 cm bend is
at 12.816667 s, with only .000429 m pre-walking excursion. No quiet-person
preparation or authored future trajectory was introduced.

Five planner requests searched the added arcs; three selected and accepted
one. Fifteen issued translating command rows join those requests using the
original optimizer request clocks. All three saved selected-arc requests pass
independent full reference and optimized-horizon revalidation and unchanged
current analytical filtering. The minimum reported analytical margin is
.169928. This demonstrates the implemented path was actually used, but a single
native comparison does not isolate every source of variation or establish
repeatability. The first failed broad search remains a failed result.

There are 187 fresh live fields and 19 supported moving-human kinetic rows.
Applied human risk equals computed CRA (.558042..683594), with fixed lambda 1.25
and unchanged adult-mass, contact, missing-factor and body-envelope priors.
Zero complete fresh generic four-factor assessments remain. This is a bounded
prior-assisted controller experiment, not new perception-model acceptance,
physical human calibration, primary-controller equivalence or long-service
restaurant validation. All broader release holds remain; the option stays OFF
by default and PR #200 remains a separate draft.

Artifacts in `outputs/moving_arc_reference/motion-v2/`:

- `report.json`, `execution.json`: original native result and frozen-source audit.
- `moving-arc-comparison.json` / `.png`: actual-speed and matched-policy audit.
- `moving-arc-execution-audit.json`: selected references and exact revalidations.
- `navigation-only-v1.mp4`: 19.9 s native video at unchanged 10 fps, omitting only
  the zero-command setup prefix; every interaction pause and final stop remains.
- `native-nearby-curve.png` / `.json`: original rendered frame, source time and
  synchronized evaluation pair; actor truth is evaluation-only.

Original v2 report SHA-256:
`0a22e5e0312ba7f4b383dca9527c8cf0707485612854d39e886efb4ee92f2083`.
