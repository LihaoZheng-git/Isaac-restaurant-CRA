# Separately declared crossing lead-time trial

The8.4 m approach-event fixture now has a bfe507d three-waypoint arrival with
explicit physical/association priors, but its visible curved/behind gates still
fail. At first22.9 s selection, bounded exact search and left-turn pattern
probes find no accepted complete left reference; primary exact recovery turns
right and waits. The human crosses very close to that already committed robot
configuration. These bounded probes are not a global infeasibility theorem.

The next trial changes **only actor start-event y from8.4 to8.0 m**, within the
existing declared standalone range. This is a different, explicitly authored
lead-time scenario. Actor start truth never enters physical estimation/control.
It does not retroactively pass the tight case or establish a controller change.
Keep the original all-gates thresholds (including >=.20 m visible arc and
<=-.10 m behind projection), three actual waypoints, stopping, source/identity/
geometry/velocity and exact MPC/CBF/independent .02 m clearance checks unchanged.

Runtime source remains bfe507d: N6/.2 s intervals, selected-obstacle tangent
fraction1/.15 m reference fade/search0, fresh class-policy cue, configured
one-source-pixel part prior, unique-full-surface option,.5 m/s native bare human,
observed occupancy only, factor priors.25, fixed lambda1.25 and explicit human
b=-1.7 control-flux override. Mass/contact/height/thermal models and policies are
unchanged. Runtime still uses no language/VLM reasoning or actor truth.

Own60-frame stationary acceptance is mandatory before wheels. Native lead-time
acceptance is pending. Baseline comparisons must match the same start-event
policy/horizon/controller options; a successful new scenario alone cannot prove
CRA causality, broad robustness, exact paper equivalence or real time.

## Recorded8.0 m result

At e70d4cd (unchanged bfe507d runtime), `lead8-human17-observed-N6-v1` passed its
parked/human-motion components and reached3/3 waypoints in22.216667 navigation
seconds:221 fields/212 translating,.174573 m clearance,.002244 m drift,.204948
CBF margin;7 supported/5 simultaneous human rows. Full bypass still FAILED:
.032518 m left arc and-.048984 m behind projection. Earlier timing alone is not
the required curved-bypass fix. Preserve this separate negative result and the
original8.4 m result. The published original field audit also identifies three
human-part/parent passing-policy conflicts; that implementation issue requires
a separately validated correction before further controller tuning.
