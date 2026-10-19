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
