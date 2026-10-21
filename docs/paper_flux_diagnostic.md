# Paper flux and control-rate diagnostic

The user approved a separately reported stronger-flux / observed-occupancy
comparison after reviewing Safe-SAGE. This branch starts with zero-wheel
components. It changes no runtime source, solver settings, source clocks,
physical evidence, primary objective or main-merge hold.

## Paper interpretation

Safe-SAGE v3 [III-C](https://arxiv.org/html/2603.05497v3#S3.SS3) assigns
negative normal boundary data `b`; greater magnitude means stronger repulsion.
It is neither CRA phi nor a distance. The paper uses semantic occupancy,
orientation-dependent robot geometry and a buffered tangential interface.
The signed margin in [IV-A](https://arxiv.org/html/2603.05497v3#S4.SS1) is
deviation from the enclosed free-space center, not minimum body clearance.
That U-shaped benchmark deliberately includes stabilization after enclosure;
it is not a general arrival guarantee.

Our existing mapping is `b = -(.5 + 1.2 * phi)`, so phi 1 gives b -1.7.
The 899 human rows in `native-checked-search-11s-v1` had computed parent phi
.557435--.733311; none was 1. The added envelope's applied risk was
.683594--.733311 and exceeded its parent's computed phi in 485 rows. Keep
this geometry/control-risk floor distinct from the four physical factors.
Overriding applied boundary data must never rewrite the computed assessment.

## Recorded timing

`outputs/paper_flux/recorded-timing-v1.json` analyzes the preserved failed run
at 95ea935. The 90 s command-log span took 3,824.330 wall seconds. Its explicit
simulation rate .025 imposes 40 wall seconds per simulated second: the paced
90 s run alone requires at least 3,600 wall seconds. This intentional pacing
explains much of elapsed runtime, not the weak goal progress within 90 sim s.

The physics/control tick is 1/60 s. With a fresh verified candidate, the
analytical CBF executes each physics step; otherwise the caller sends zero.
MPC is asynchronous and scheduled every .15 sim seconds when the worker is
free and the field is usable; invalidations can request earlier retries.
The .2 s horizon integration interval is not the replanning period.
There were 577 completed planner events (550 accepted), or 6.411 events per
sim second. Their wall-time min/median/p90/max were .674907/2.760717/7.670311/
9.126633 seconds. This includes rejected outcomes and is not a solver-only
or matched-hardware latency benchmark.

The paper's [hardware paragraph](https://arxiv.org/html/2603.05497v3#S4)
reports about 15 Hz fields, 100 Hz MPC and analytical filtering at odometry
rate; it does not specify the numeric odometry rate there. Our multirate
structure already separates planning and analytical filtering. These timings
do not establish that changing their rates repairs the native bypass. No
real-time or paper-equivalent performance claim follows.

## Preliminary zero-wheel recipe

Two independent 60-frame stationary runs use the existing
`fixed_class_safe_sage` baseline, one with the .4 + .25 m disk and one with
that disk disabled. Both disable envelope-dependent missing-fragment coverage
and retained-identity passing selection. All physical occupancy, person/mass
association, uncertainty inputs, lambda 1.25 and original checks remain.
The native actor does not begin crossing and wheels are never enabled.

Important limitation: this existing fixed-class baseline sets b -1.7 for
**unknown as well as person**. Consequently this pair is preliminary component
validation, not an isolated human-only flux experiment or exact replication
of the paper's b -0.5 static-object setting. Never relabel unknown restaurant
geometry as a low-risk known object to produce that setting. Separate runs
are not bit-identical recorded input comparisons.

The original three-way comparison remains an intended next diagnostic:
computed CRA with current envelope; increased applied human flux with the
same geometry; then observed occupancy without the added circular prior at
the same flux. Keep all nonhuman policies matched. Removing the prior disk
must retain observed returns and robot configuration-space expansion; larger
flux cannot establish missing-body coverage. Preserve independent clearance,
arrival, stopping and continuity checks. New motion remains blocked pending
an adequate geometry diagnostic; stationary numeric completeness is not it.

## Completed component results

Both runs used runtime source 7686090 (unchanged from ff4f55b). Both passed
their 60-frame human stationary gate with zero nonzero wheel commands and no
traceback. With the circle, all stationary body/source/provenance checks also
passed; without it there were no added envelope rows or containment acceptance
claim. The selected component is the human, not the unobserved restaurant bowl.

| Component | Accepted visual instances across all tracks | Complete four-factor assessments | Added envelope rows | Wall seconds including launch/shutdown |
| --- | ---: | ---: | ---: | ---: |
| Fixed-class flux, .65 m circle | 81 | 0 | 34 | 166.998 |
| Fixed-class flux, observed occupancy only | 82 | 0 | 0 | 162.172 |

Every applied policy row in both components had b -1.7 and lambda 1.25.
Fresh-person policy row counts were 42/22 and unknown row counts 5,486/5,457;
the envelope component counts auxiliary policies too. These are policy rows,
not numbers of unique humans. Both preserved rejected future/stale LiDAR/RGB
epoch pairs. No invalid pair was accepted to improve readiness. All other
factor priors and the zero complete-factor result remain explicit.

Artifacts are under `outputs/paper_flux/fixed-flux-envelope-stationary-v1`
and `fixed-flux-observed-stationary-v1`, with compact results in
`stationary-pair-v1.json` and `stationary-pair-audit-v1.json`. Their original
RGB `stationary-current.png` captures are simulator observations, not generated
illustrations. These are independent stationary components, not moving-camera,
live human coverage, matched navigation, causal CRA superiority or bypass
acceptance. The moving comparison remains pending an isolated human policy
and sufficient geometry diagnostics; no wheel trial was launched.

Validation: 34 relevant baseline/body-envelope unit tests passed in 6.25 s,
repository Ruff passed, and whitespace checks passed. Runtime source is
unchanged; this step adds the research/timing record rather than a new
controller or a claimed 100 Hz implementation. Existing failed native trials,
PR #117's separate perception gate, main and long-service holds remain.
