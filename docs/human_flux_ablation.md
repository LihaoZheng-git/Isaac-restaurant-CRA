# Isolated human flux and observed-occupancy ablation

The user approved this comparison on 2026-10-21. It is separate from the primary
CRA experiment and from the guidance-tracking objective ablation already used
in the fixture. Fixed lambda remains 1.25. The nominal waypoint command, all
physical factors, and all nonhuman field policies remain unchanged.

## Two independent options

`--human-boundary-flux -1.7` raises applied boundary repulsion for a verified
human and, if enabled, that human's auxiliary circle. The allowed range is
[-1.7,-.5] on the existing mapping; the override can never reduce computed
risk. Original `cra_phi` and physical attributes remain intact. Each applied
override records original visual identity/frame, current LiDAR epoch, identity
age and an explicit control-ablation basis. Unknown/unmatched geometry keeps
its own computed CRA policy; it is not globally assigned unit risk.

`--human-observed-occupancy-only` suppresses the added circular occupancy while
retaining the existing identity memory, domain protection against catalog mass
inference on people, original clocks, current support, competitor and continuity
checks. It requires explicit human identity configuration through the existing
`--human-body-envelope-radius .4 --human-envelope-centroid-uncertainty .25`
options; those circle dimensions do not enter control occupancy in this mode.
The mode retains all original observed returns and robot configuration-space
geometry. It rejects envelope-based missing-fragment coverage and retained-
identity passing selection; matched comparison recipes disable both options
in all variants. Fresh-only side selection remains available.

Retained identity may authorize this explicitly labeled high-repulsion policy
within its existing original-source age limit (30 s in the recipe), with current
LiDAR support. It never supplies or renews mass, motion, contact, plume, support
or fall-height evidence. It is not fresh visual classification.

The unused circle remains a **shadow** evaluation record with
`applied_to_occupancy=false` / `used_for_control=false`. Its containment result
is reported without being treated as acceptance of an unused geometry model.
The independent configured actor-body/robot clearance guard still aborts at
.02 m, including braking frames and robot yaw. Actor truth enters evaluation/
abort only, never field construction, passing selection or CRA.

Larger flux does not reconstruct an unobserved body. Observed-only navigation
has a different geometry assumption and requires its own stationary component,
independent clearance, source continuity, actual arrival and stopping checks.
Preserve the original .25 m containment failures; neither a shadow audit nor
disabling the disk retroactively passes the old active-disk experiment.

## Diagnosis before new motion

`outputs/paper_flux/command-profile-v1/command-join.json` joins accepted plan
events and executed commands by their original optimizer source epoch. During
70--90 s of the preserved failed run, 1,076/1,145 joined valid command frames
had zero planned forward speed; median planned/applied speed was zero while
median nominal speed was .142972 m/s. No analytical command changes occurred
in these joined rows. This excludes stopped/unmatched frames and does not
claim the filter never intervened elsewhere. The planner's choice of motion
needs attention, rather than inferring that slower playback caused this.

A SHA-verified offline replay of source frame 4681 / 78.9 s again converged
to (0,0) with `no_exact_safe_progress_reference`. Its profiled 1.471 s call
spent 1.067 s in progress-seed selection and .341 s in checked reference search
(nested timings are not additive). There were 658 exact trajectory evaluations;
field alignment checks consumed .266 s. This is one cold, profiler-instrumented
request, not a matched latency benchmark or proof of forward-only infeasibility.
No safety checks, cadence, solver budgets or simulation pacing were weakened.

## Software validation

The 18 new tests cover human-only policy selection, unchanged physical CRA and
other policies, original source clocks, stale/foreign/duplicate rejection,
increase-only flux, mode incompatibilities, and actual raw-grid/field
integration. The final full unit suite passed 2,291 tests in 84.41 s; 52 focused
tests passed in 5.08 s and repository Ruff passed. The native source must be
frozen after validation and all stationary
components must finish before the bounded observed-only motion trial.

## Completed stationary results at 4964cf4

All three independent native components passed 60 stationary frames with zero
nonzero wheel commands and no traceback. Their human association, current
motion and contact-proxy/declared-body-prior checks passed. Both active-circle
containment audits passed; the observed-only shadow audit also passed but is
explicitly not used as control-geometry acceptance.

| Variant | All-track accepted instances | Added-circle rows | Human override rows | Complete four-factor assessments |
| --- | ---: | ---: | ---: | ---: |
| Computed CRA, active circle | 79 | 23 | 0 | 0 |
| Human-only b -1.7, active circle | 84 | 35 | 70 | 0 |
| Human-only b -1.7, observed occupancy | 79 | 0 | 23 | 0 |

Override rows include both parent and auxiliary circle policies when enabled;
they are not counts of unique humans. Of the 70/23 overrides, 22/4 used retained
original visual identity with current positive LiDAR. Computed human CRA stayed
.557431--.683594 / .557440--.683594 while applied human field risk was 1.
Every audited nonhuman field policy equaled its computed CRA, and every lambda
was 1.25. All runs retained rejected future/stale RGB/LiDAR pairs, original
identity/factor source epochs and unavailable inputs. The declared mass/contact
and missing-factor priors are not complete visual physical estimation.

Outputs are in `outputs/paper_flux/isolated-{cra-circle,human17-circle,human17-observed}-stationary-v1`.
Compact audits are `isolated-stationary-v1.json` and
`isolated-stationary-audit-v1.json`. These are independent stationary inputs,
not bit-identical replay or a moving bypass/causal navigation comparison.
The measured launch-to-shutdown times were 167.477/167.172/167.232 wall seconds;
they do not measure navigation throughput or prove a timing improvement.

The actual recorded-progress figure is
`outputs/paper_flux/recorded-progress-v1.png`. It visualizes the old failed
active-circle trial only; source-time-matched MPC and post-CBF commands overlap.
It must not be presented as a new envelope-free motion result.

## Motion hold and review-ready next trial

Automatic approval review rejected the proposed combined stationary/motion
launcher **before execution** because its automatic wheel trial would conflict
with the standing geometry-failure hold. The safe alternative runs only the
three zero-wheel components and cannot automatically promote to motion. The
rejected launcher was not used through a workaround or indirect execution.

`outputs/paper_flux/review-ready-observed-motion-v1.json` contains the concrete
proposed invocation: one observed-only human crossing, b -1.7 on verified human
tracks only, unchanged guidance ablation/budgets/priors, lambda 1.25, its own
60-frame stationary gate, strict missing-fragment continuity, original exact
and freshness checks, independent .02 m clearance abort including braking,
90 sim seconds and actual three-waypoint arrival/stopping/passing evaluation.
It was prepared without launching any simulator or wheels. The .025 pacing
remains explicit; source/candidate ages are not renewed to improve progress.

An explicit decision is needed to conduct that bounded observed-occupancy
ablation while full body-coverage uncertainty remains unresolved. This would
not repair or waive the old active-circle failures for their original scope,
authorize reverse/wider bounds or establish universal safety. Main and
long-service remain held. No new moving bypass is claimed by this branch.
