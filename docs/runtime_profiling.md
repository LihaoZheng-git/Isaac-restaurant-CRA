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
