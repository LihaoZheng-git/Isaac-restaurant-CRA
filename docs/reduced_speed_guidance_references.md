# Reduced-speed checked guidance references

This is an explicit candidate-generation option within the separately approved
guidance-tracking MPC ablation. The primary objective, obstacle-unaware waypoint
nominal, physical CRA, lambda1.25, occupancy and every exact safety/freshness
check remain unchanged. Default minimum speed fraction1 retains the old single
preferred speed and candidate order. The opt-in minimum.25 tries fractions
1,.75,.5,.25, with the existing cap/turn-first pool at each speed. At most four
speed groups are evaluated. The normal progress-quality gate still rejects
turn-only references. Every selected reference and final command horizon needs
full exact DTCBF/dense geometry acceptance and the ordinary analytical filter.
The pool is neither an executable minimum speed nor a convergence guarantee.

`--guidance-reference-minimum-speed-fraction .25` enables it in the live demo;
positive tracking weight is required. Diagnostics retain the original preferred
speed, configured minimum and selected fraction, independently of source clocks.

## Original-input diagnosis

The body-prior/N6 run's first selected native request has source19.7 s,
frame1116, SHA256
`a6cfce471d70f71a3689d61bffb66b3371db56f45332bf85992b27cccfa76c83`.
The single-speed generator found no useful exact-safe reference and recovered
with(v=.037499734,omega=-.4), opposite the selected left preference. Exact same
fields, source and state with minimum.25 produce a useful checked reference:
initial(v=.029610147,omega=+.4), displacement.040771278 m, minimum DTCBF
margin.000605402. The final analytical filter leaves that first command unchanged.
The old single-speed replay reproduces its original command exactly. These are
trusted local numeric replays, not new wheel/sensor acceptance or proof that all
left paths are feasible. Prior geometry and factor assumptions remain explicit.

Validation:84 focused tests passed in6.40 s, all2,437 unit tests passed in85.33 s,
Ruff and whitespace checks passed. New numeric tests reproduce a case where
full-speed reference horizons fail and a slower one passes the same checks;
stationary/unsafe/zero-weight behavior and nominal immutability remain checked.
A new own-stationary/native trial has not been enabled; virtual results follow. The earlier
frozen native attempt is retained independently; this change cannot repair it
retroactively. No main, long-service, real-time or complete fresh four-factor
perception acceptance follows.

## Full route diagnostics and remaining decision

All inputs below are authored ORACLE geometry/side, without physical wheel
commands or sensor/motion-selector acceptance. The selected-tangent mode's
matched N6 single/reduced-speed pairs both failed stationary-solution rejection.
Reduced-speed arcs were+.252099/-.253806 m, but arrival still failed. The
forecast-only reference pair also stalled. These failures remain recorded.

At N30, both legacy single-speed selected-tangent routes failed. Both reduced-
speed routes arrived (left35.55 s/right34.0 s), with passed exact geometry/CBF/
MPC checks. Their main flanks were+.682294/-.683540 m, but opposite excursions
-.210126/+.212026 m remain explicit. This is soft guidance, not enforced side.
Output: `outputs/paper_flux/virtual-selected-speed-N30-v1`.

The original native19.7 s field/state parameter diagnostic still rejects a full
N30 dynamic forecast while N6 succeeds. Its N30 nominal repeats the original
first command, so it is not an exact new native nominal-horizon replay. Failed
margin is explicitly unavailable/null. Do not combine the static N30 arrivals
with this moving-field probe into native acceptance.

An unintegrated selected-tangent bounded-search prototype passed52 focused tests
but both N6 virtual routes stalled. Its exact patch is archived in
`outputs/paper_flux/selected-search-rejected-N6-v1.patch`; the runtime was restored
to2c71d0e. Two outputs-only longer-static-reference/short-prefix prototypes also
stalled: v2 arcs+.210604/-.216257 m; v3+.310529/-.322571 m. v1 was a harness
TypeError, not a completed controller result. These are not integrated modes.

The current bounded changes do not complete the native curved-bypass task.
A longer-range kinematic path-reference mechanism would be a new controller
ablation beyond the already approved guidance-tracking cost. The user has been
asked to choose that direction versus an explicitly different crossing route
that continues away from the final-goal corridor. Keep the primary CRA/Safe-
SAGE comparison unchanged and this PR in draft; do not relax safety, geometry,
source clocks or arrival criteria to convert these failures into passes.
