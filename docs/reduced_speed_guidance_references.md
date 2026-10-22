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
Virtual route and a new own-stationary/native trial remain pending. The earlier
frozen native attempt is retained independently; this change cannot repair it
retroactively. No main, long-service, real-time or complete fresh four-factor
perception acceptance follows.
