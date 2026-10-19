# Bounded restaurant baseline comparison

This protocol stops before the long service demonstration. It implements the
first matched static-aisle, standing-server and crossing-server comparisons;
table-side factor studies and separate carried-group navigation remain later
gates. Successful earlier development runs are not repetitions of this study.

## Methods and matching

- `geometric_velocity_cbf`: the existing velocity-aware circular CBF projects
  nominal forward speed and preserves nominal omega. Its radius circumscribes
  the actual asymmetric robot footprint and policy margin, plus the occupied
  cell half diagonal. It consumes every unexpanded occupied grid cell and
  current tracker velocity, predicts supported cells over their original age,
  and labels remembered/unmatched cells' zero velocity as a prior. The .35 s
  freshness guard and nearby-track-loss stop remain. This different controller
  can stop without finding a detour; it is not an MPC/Safe-SAGE ablation.
- `fixed_class_safe_sage`: uses the fingerprinted repository policy in
  `configs/safe_sage_paper_baseline.json` at the field boundary. Only a fresh
  accepted RGB/SAM person association selects `person`; SAM-only objects and
  unmatched environment stay `unknown`. Both currently map to risk 1. The
  configured generic `object` risk 0 is not selected by inventing a class.
- `continuous_cra_safe_sage`: applies the unchanged per-object physical-cue
  assessment and configured factor priors to the field boundary. This versus
  fixed-class is the primary comparison of CRA under the same controller.

The two field controllers share nominal waypoints, tracker, models, occupancy,
MPC/recovery budgets, exact DTCBF/dense geometry and analytical CBF checks,
expiry and stopping. Primary lambda is 1.25. The geometric reference retains
shadow CRA field computation for the same perception/scheduling path, but never
uses a risk field to command wheels. Its wall time therefore is not a standalone
CBF speed benchmark, and its distance-squared CBF margins cannot be numerically
compared to h_full margins. All records retain raw CRA factors/phi separately
from the applied baseline field policy; classifier absence remains explicit.
The explicit comparison loader rejects a changed flux scale or lambda. The
legacy guidance default uses a different 1..6 flux scale and must never be
inherited by the fixed-class comparison.

Human mass remains the visually selected adult 70 [45,100] kg prior. The new
human body contact prior is fixed at .10 with fresh association for every
method; raw outline/old saturation records remain. Potential/thermal default
factor priors remain .25 where unsupported. No temperature, physical sharpness,
new visual model, or complete fresh four-factor perception is claimed.

## Prespecified collection

`scripts/evaluation/run_restaurant_baselines.py` writes an immutable manifest,
code revision and options hash before any trial. The pilot has three methods
on each of three nominal cases (nine trials). The study has five repetitions
per method on static, standing, and three crossing timings (75 trials). Method
order rotates by case/repetition before outcomes. Repetitions are descriptive,
not independent proof of statistical safety; native animation/inference need
not be deterministic.

The crossing study uses an **exogenous navigation-elapsed-time** trigger at
7/8/9 s (pilot 8 s), .5 m/s native walk and the existing world actor path. This
refines the prospective placement variants to timing variants without changing
hazard geometry. The legacy robot-y trigger remains for old callers. A human
cannot be triggered by controller progress in these matched trials. Standing
and crossing use the same +15 degree mount, .2 s visual cadence, two visual
slots and .10 body prior. Static uses its existing accepted bowl view/route.
Every run repeats stationary perception acceptance before nonzero wheels.
The first 18-frame static pilot failed the third method's component gate with
only two fresh bowl associations and no confirmed motion on its last matched
new track. Its two earlier navigation passes and failed zero-wheel attempt
remain under `outputs/restaurant_baselines/pilot-v2/`; they do not form a
complete comparative cell. The subsequent development recipe uses **60 frames
and the audited observed-LiDAR SAM context-box retry for all three methods**.
All original mask/ownership/three-observation and factor checks stay unchanged.
This is a newly labeled initial-acquisition configuration, not a retroactive
pass or a claim that one of those two changes alone fixes perception. Component
observation history and current live field freshness remain separate.
All cases retain a 90 s navigation limit, fixed .05 simulation pacing, 1000 QP /
20 nonlinear-rescue iterations and .5 s optimizer candidate age; original field
and execution expiry stay unchanged. No observation pauses are scheduled.

```powershell
& 'C:/Users/tayli/anaconda3/envs/env_60_isaacsim_yolo/python.exe' scripts/evaluation/run_restaurant_baselines.py --output outputs/restaurant_baselines/pilot-v1 --execute
```

The runner resumes completed invocations, including failures, without replacing
their artifacts. An unfinished/crashed directory is retained and cannot be
silently overwritten. Source changes require a new output root. Launcher exit
code alone never passes Isaac; saved validation, traceback and stopping records
matter. Each completed invocation enters `comparison.json`, including a missing
motion report when the stationary gate or process fails. `--case`/`--baseline`
can execute a subset of the unchanged manifest; incomplete cells stay visible.
A `STOP` file prevents subsequent starts while allowing the current bounded
trial to stop normally. Missing motion reports create this file automatically;
ordinary motion timeouts remain comparative outcomes and do not halt the batch.
The explicit `--continue-after-stationary-rejection` collection option can count
the saved zero-wheel component rejection as an end-to-end failure and proceed
to the next independent trial. It requires the exact gate rejection, saved
failed gate and zero wheel count; a logging/process error still halts. Every
later trial repeats its own unchanged gate. Report all attempted trials and
controller-stage attempts separately, never silently discard initial failures
or treat a missing trajectory as a navigation collision.

Three independent zero-wheel repeats of the 60-frame/context recipe passed all
component checks at source `5878a35`, each with eight fresh matched bowl masks.
Artifacts: `outputs/restaurant_baselines/init60-context-v1/repeat-00..02/`.
These support the next newly frozen pilot, not the failed 18-frame case or
complete fresh perception during navigation.

## Measurements and interpretation

Report actual waypoint arrival **and** full acceptance, safe stopping, signed
independent monitor clearance, path/time/stops, intervention samples, safety-stop
episodes, field ages/build latency, score traces and evidence modes. Monitor
overlap is a conservative inflated-geometry overlap, not measured PhysX contact.
The older `RestaurantTrialResult` now retains negative clearance with a flagged
collision outcome instead of discarding it. The new extractor keeps this
monitor's meaning explicit and can retain unavailable measurements as null.

Configuration matching checks the route, scene hashes, mount, priors, timing,
lambda, budgets and simulation pace across each method cell. Different robot
paths lead to different sensor streams; these are matched scenario conditions,
not identical-observation replays. Perception coverage and prior reliance must
accompany navigation summaries. Positive clearance alone is not completion,
and successful completion alone is not CRA superiority. Keep every failure and
denominator. Do not change scores, priors, unknown handling or geometry after
seeing results to improve a method.

Long-route field transitions, payload transport, carried-group wheel control
and moving-camera plume acceptance are outside this comparison gate. Main's
separate strict fresh-perception merge hold remains unchanged.

## First complete pilot: actual results, not final paper collection

`outputs/restaurant_baselines/pilot-v3/` collected all nine attempted trials
at source **9220f4c**. All three case cells matched the invocation contract,
actual model hashes, route, scene hashes, source handling, priors, mount,
clustering, control budgets and lambda. The post-collection extraction under
`pilot-v3-analysis/` makes acceptance failures after arrival explicit; original
reports and `comparison.json` are preserved. One trial per method/case is
descriptive and does not establish statistical superiority.

| Case | Method | Full acceptance | Navigation simulation seconds | Minimum independent monitor clearance (m) |
| --- | --- | --- | --- | --- |
| Static aisle | Geometric CBF | Pass | 23.867 | 1.213 |
| Static aisle | Fixed-class Safe-SAGE | Pass | 20.500 | 1.219 |
| Static aisle | CRA + Safe-SAGE | Pass | 20.500 | 1.219 |
| Standing server | Geometric CBF | Pass | 25.150 | .255 |
| Standing server | Fixed-class Safe-SAGE | Pass | 21.917 | .222 |
| Standing server | CRA + Safe-SAGE | Pass | 23.100 | .273 |
| Timed crossing | Geometric CBF | Fail: lost-track stop | 14.350 (abort) | .643 |
| Timed crossing | Fixed-class Safe-SAGE | Fail: perception coverage after arrival | 22.533 | .309 |
| Timed crossing | CRA + Safe-SAGE | Fail: lost-track stop | 14.750 (abort) | .709 |

Thus **six full passes and three failures** were retained. Seven robots reached
their goals, but fixed-class crossing had only four supported steady-moving
kinetic rows versus the unchanged requirement of five. CRA crossing also had
four, and geometric crossing had six; all three passed the median motion-error
checks. Positive clearance and numerical phi never replace full acceptance.

Static fixed-class and CRA paths/times were identical: this is a wide-aisle
negative control, not a demonstrated CRA advantage. Standing CRA had about
5 cm more monitor clearance and 1.18 s more travel time than fixed-class in
this pair. Geometric CBF had 178 fail-closed command samples (about 2.97 sim s)
in the static case; rejected margins were materially negative, not rounding
noise. Its circular model and margin units remain distinct from the primary
fixed-class/CRA comparison. All these runs are prior-assisted and simulation
paced. Sparse fresh masks and numerous environmental factor priors remain in
the full streams; numeric completeness is not complete physical estimation.

### Crossing limitation and bounded next diagnostic

CRA stopped on track626's former `(28.716,9.066)` position, originally seen at
18.1 s and missing 4.7 s when the robot entered its 1.2 m watch region.
Geometric CBF stopped on track596's former `(29.334,9.003)` position, missing
5.0 s. Both had finite CRA policies; the unresolved positive geometry, rather
than missing factor scores, caused the stops. Do not delete those observations,
extend merge/source clocks or call the stops successful arrivals.

An **offline, one-scan** clustering audit used the nearest copied original scans
(18.2 / 18.1 s). At .12 m, the independently selected human region split into
854+79 returns and 903+112 returns (plus three unclustered). At .18 m, those
same 933 / 1,018 positive returns formed one component, with no extra-region
returns in those selected components; .20 m gave the same body component.
The actor reference selects the scoring region only and never supplies runtime
geometry/factors. This supports testing the existing explicit .18 m setting,
not a repaired identity history or new motion acceptance. The radius is not a
mass, confidence, risk, safety-distance or unknown-evidence adjustment.

Audit: `outputs/restaurant_baselines/crossing-cluster-audit.json`. Validate a
bare crossing with the robot parked before any new wheel trials; retain every
failed original case. A later comparison must share its newly declared
configuration across methods. Broad occlusion/crowd tracking, articulated
free-space inference and general body reconstruction remain outside this gate.

The **75-trial replicated collection is implemented but has not been run**.
Resolve the selected crossing's continuity/kinetic coverage before spending
that collection. No long service mission was started, and main stays unchanged.

### Explicit .18 m human connectivity recipe

The bare-human parked crossing at `bare-cluster18-stationary-v1/` passed native
arrival/stopping, articulated walking, independent layout clearance, current
RGB/person/SAM and velocity checks, and the separately typed .10 body-contact
prior validation. All robot wheel commands were zero. This gate permits testing
the existing .18 m connected-cluster setting during navigation; it does not
establish coupled navigation or repair any original failed trial.

The next comparison recipe explicitly selects .18 m for **every human method**,
standing and crossing. Static options and legacy demo defaults remain unchanged.
All positive occupancy, association/freshness/track-loss checks, risk/confidence
formulas, factor priors, field/controller budgets and lambda 1.25 remain intact.
Use a new immutable output root and report its results separately from pilot-v3.
