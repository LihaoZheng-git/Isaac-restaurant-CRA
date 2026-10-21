# Optional worker profiling

The approved observed-only crossing at c3f577d failed at 14.133333 simulation
seconds, with 1/3 waypoints and an unchanged safe-stop response. Its log contains
34 `Another profiling tool is already active` rejections. A separate concurrency
probe reproduced this CPython 3.12.13 diagnostic error before the second worker's
function ran. See [the preserved trial](human_flux_ablation.md).

`robot/evaluation/runtime_diagnostics.py` now reserves a single profiling slot
with a nonblocking lock. Overlapping and nested calls run their actual work
immediately without a profile. An external active profiler's known enable-time
collision also skips only local instrumentation. Older Python profiling hooks
are left intact. The lock does not serialize planner and perception computation.

Wrapped functions execute once with their original arguments, results and
exceptions. In particular, a worker's own identical `ValueError` is not caught
or retried. The slot is released after worker/enable failures. Existing slow-call
thresholds and profile-write warnings remain; unrelated instrumentation failures
are not silently hidden. No dependency, thread-pool limit, control objective,
source clock, physical factor, field, freshness or exact safety check changes.

Profile files sample only calls that obtained the profiling slot and met the
duration threshold. A missing file is not proof of fast computation, missing
work or missing evidence. Profile overhead and overlap selection prevent using
this collection as an unbiased latency benchmark. Existing outer worker wall
timings still cover unprofiled calls.

## Validation without wheels

- 38 focused runtime/inference/human-policy tests passed in 4.96 s.
- All 2,297 unit tests passed in 87.91 s; repository Ruff and whitespace checks
  passed.
- Real concurrent threads complete the second call while the first profiler is
  held; nested calls, external profiling, genuine worker failures and recovery
  after failures are tested. Both workers run once.
- A trusted, SHA-verified original planner request (frame 4681, epoch 78.9 s)
  was deserialized independently for unprofiled, overlapping and profiled calls.
  All three produced exactly equal command sequences and the same status and
  `no_exact_safe_progress_reference` result. The overlapping planner finished
  before the other profiler was released, without errors. Original epochs were
  unchanged. Audit: `outputs/paper_flux/safe-profiling-replay-v1/audit.json`.

This is an instrumentation fix and offline numeric regression, not a successful
crossing, fresh sensor experiment, native real-time result or repair of the old
90-second stall, which had profiling disabled. The one-trial geometry-hold
exception has been consumed. No additional wheel trial or main merge follows
from these software checks; another bounded motion trial requires a new explicit
decision, with its own stationary gate and all original checks.

The user subsequently authorized autonomous diagnosis, fixes and bounded
guarded retries until this test works, without repeated approval questions.
That newer instruction supersedes the one-trial permission limit. Each trial
still needs its own stationary gate, frozen validated runtime, independent
clearance/braking monitor and all original freshness/exact checks. No reverse,
wider uncertainty bound, safety relaxation or main acceptance follows.

## Native retry at 1082393 (functional source 49f56f2)

`outputs/paper_flux/isolated-human17-observed-motion-v2` passed its own 60-frame
zero-wheel gate and completed all three waypoints in 29.216667 simulation
seconds, with .365797 m minimum independent clearance, .318745 minimum CBF
margin and .002303 m stopping drift. It published 285 fields (248 translating)
without any inference-worker rejection or profiling collision during motion.
The startup camera/LiDAR epoch rejection remains recorded. Navigation took
1,243.759 wall seconds; the full launcher took 1,414.858 s. This is paced
simulation, not real-time acceptance.

Human-motion validation passed: 270 current matched rows, 12 supported moving
kinetic rows, five simultaneous rows, .140783/.014298 m/s median world/relative
speed errors and actual native crossing. There were 78 selected passing-policy
rows, all based on a fresh original selection, and no identity-prior selections.
Keep 83 all-track accepted visual instances, 19,200 object assessments, 76,411
configured factor-prior uses, 290 property-prior uses and zero complete
four-factor assessments separate from first-track counters.

**Overall bypass acceptance still failed.** Walking-window lateral excursion
was -.028852 m (required >=.20 m) and behind projection was +.032850 m (required
<=-.10 m), although arrival/rejoin passed. The unused circle passed its shadow
audit; it was not control geometry. No old active-circle failure is repaired.

An independent actual-trace timing audit found the robot at world y=7.5336--
7.6747 m during supported walking, while the human crossed near y=8.9 m. The
closest moving center separation was 1.300220 m. This separates the repaired
diagnostic interruption from a poorly aligned curved-bypass encounter. It does
not prove an unfiltered counterfactual safe or establish a causal CRA benefit.
Preserve the failed arc/behind scores and the original 11 s fixture. A later
declared crossing delay is a different development scenario, not its repair.

## Additional profiling limitation

A separate zero-wheel probe on the installed CPython 3.12.13 found a function
executed in another thread inside the worker-triggered cProfile output
(`outputs/paper_flux/foreign-thread-probe-v1.json`). Native perception-triggered
files also contain planner calls. The collision correction prevents rejection
of wrapped work, but does not make cProfile thread-local. These concurrent
files cannot establish per-worker call attribution or reliable call-tree timing;
some self/cumulative entries are inconsistent. Preserve them as failed
instrumentation diagnostics, and use isolated trusted replays for call profiles.
Outer worker/stage wall timers still measure elapsed time, including contention
and diagnostic overhead, rather than pure algorithm throughput. Do not use the
mixed profiles to assert a new perception/planning bottleneck.
