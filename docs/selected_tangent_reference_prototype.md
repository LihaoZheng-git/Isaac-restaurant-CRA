# Offline selected-tangent reference prototype

This branch is an **unmerged prototype**, not an integrated controller fix.
The waypoint nominal, CRA factors, fields, actuator limits, exact MPC/DTCBF/
dense geometry and analytical CBF checks remain unchanged. A separate positive
tracking objective may prefer the semantic field's component tangent to the
actual planar safety gradient. The runtime permits this preference only with
the existing current positive passing-selection track IDs; original source
clocks and identity/factor provenance remain unchanged.

The generator limits desired travel speed to the original nominal speed,
projects the actually requested `(v, omega)` through the existing lookahead
control matrix, and still requires full exact horizon acceptance before using
a reference. No primary caller enables the new fraction by default. A
per-request configuration copy disables the prototype when selection is absent;
it does not mutate the shared planner. Geometry, missing-factor handling and
passing-selection gates are unchanged.

Seventeen new component tests passed, including steering before an inactive
halfspace binds, exact full-horizon checks, zero/parallel-gradient behavior,
invalid/current identity and epoch handling, request-local gating and unchanged
source/nominal inputs. The first implementation passed all 2,344 unit tests in
85.79 s. The subsequent angular-coupled generator passed 67 focused tests and
Ruff; its full regression/native validation remains pending.

Trusted SHA-checked original-state replays in
`outputs/paper_flux/selected-tangent-coupled-v1` did **not** improve either late
crossing snapshot at the legacy 30 x .2 s horizon. Both zero/positive fractions
retained the same failure or rotation-only recovery. Keep this negative result.
The separate horizon diagnostic at 6/10/15 steps preserved exact acceptance and
original epochs, and found forward commands at six steps in both snapshots;
it is not a native bypass or real-time result. Shorter horizons change prediction
coverage and must be reported as a separate matched ablation. This prototype
does not justify changing main or promising a curved human bypass.
