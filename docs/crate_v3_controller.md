# Closer crate with the V3 controller

The user requested a repeat of the failed closer-crate probe using the V3
controller settings. This separate default-OFF `--crate-v3-controller` recipe
keeps the original crate, start (15.65, 4.05, 0), straight 4 m nominal route,
45 s bound, retained/confirmed-refresh evidence, physical limits, lambda and
all clearance/arrival/stopping checks. Earlier failures and V3 videos remain.

`configs/restaurant_crate_v3_controller.json` matches the applicable settings
of `restaurant_service_demo_v3_corrected_contact.json`: N6 / 1.2 s MPC horizon,
5 cm grid, existing guidance objective (weight 100), tapered rescue seed,
consistent temporal Jacobian, 0.9 progress-seed fraction, guide mode and 0.025
simulation pacing. Unfinished-QP proposals remain disabled. These are multiple
existing changes together, not an isolated single-parameter ablation.

V3 selects a tangential human passing reference only when a verified human
selection exists. Without one, its runtime makes a request-local planner with
tangent fraction zero. This static test explicitly uses that same fallback;
it does not give the crate a human identity, passing side or authored detour.
The existing moving-arc option remains inert with no selected reference and
zero tangent fraction. Unit tests compare the actual runtime fallback with
the explicit configuration and reject profile drift or human-policy flags.

The CLI validates the opt-in before simulator startup and saves actual MPC
and guidance configurations in `controller-profile-validation.json`. Fresh
independent60 and own60 stationary gates are required before wheel motion.
Runtime and scene sources must remain frozen. Native evidence will be saved
separately under `F:/CRA-experiments/crate-v3-controller-20261105/`; compact
summaries and preservation checks belong in `outputs/crate-v3-controller/`.

Validation before native: 150 focused tests and all 3919 unit tests PASS
(249.72 s), plus repository Ruff, compile and diff checks. The first full-suite
attempt hit sandbox temporary-directory setup errors; a fresh repository-local
temporary directory resolved them without code changes.

## Native result at 8e4faaa

Both fresh independent60 and own60 stationary gates PASS with frozen runtime
and scene sources. The recorded MPC configuration exactly equals the accepted
V3 report. Navigation FAILS the unchanged independent clearance guard after
19.10 simulation seconds: 0/1 goal, final post-stop local pose
(1.410707, -0.061937, -0.581521), about 2.59 m short of the goal. Maximum lateral
deviation is only 0.061224 m, toward the crate, versus 0.345838 m away in the
earlier N30 run. Neither trial completes the pass. The V3 repeat does not
repair the closer-crate case and must not replace an accepted demo.

Independent clearance reaches 0.019707 m before stopping and 0.018789 m during
stopping, below the original 0.02 m threshold. Stop drift is 0.000963 m. The
minimum recorded CBF margin remains +0.069408 against its sensor-derived field;
that is not proof of the independent full-fixture margin. The fixture monitor
uses an authored conservative disk as well as the restaurant raster, never
control inputs. This failed conservative-clearance gate is not a claim of
observed physical mesh collision. Keep the failed result and original margins.

There are 187 live fields, 143 committed while translating, and 147 returned
plans accepted at request evaluation: 86 converged, 49 progress seeds and 12
feasibility seeds. Guidance obtains a checked reference for 35 plans; 112 say
`no_exact_safe_progress_reference` (first at navigation 5.2333 s). Those requests
use the original MPC fallback, with no active guidance tracking objective.
Of 1146 applied command rows, 358 have zero forward speed, including 188 with
no stop reason. Plan waits account for 106 rows, new-field invalidations 58,
and stale fields six. No dynamic uncertainty regions occur. Thus accepted
low/zero-forward recovery commands and plan turnover remain; this is not a
missing-track stop, nor evidence that no feasible route exists globally.

Exact retained-source/point replay has ZERO errors, but continuous target
retention FAILS: only 33/187 live fields contain the original crate record,
at source 8.2--11.8 s. Its rough mass 2.999399 kg and shape cue 0.506347 stay
fixed, contact factor 0.170476, phi 0.620930--0.622548 with live kinetic inputs.
All selected points are inside the original mesh bounds; only 17/33 surface
centers pass the separate whole-center diagnostic. The last source has zero
current raw returns inside the independent crate box with the original 3 cm
reporting allowance. This does not establish free space or absence of retained
occupancy. No constant identity/phi or continuous full-object sensing claim.

Exact view-refresh replay PASS covers 60 independent and 247 own/live sources.
All 21 crate refresh attempts reject incomplete old support and no meaningful
new coverage; ZERO crate confirmations/replacements occur. One NON-TARGET
record (8000000003, observed center approximately 18.665/5.084/0.732) receives
a properly replayed property replacement at source 13.4 s. Do not credit that
event as a successful crate update.

The next bounded diagnosis should examine the rejected unicycle guidance
references and sensor-versus-independent geometry near the crate. V3's
human-specific tangent selection is absent by design in this static repeat.
Neither weakening the clearance check nor increasing phi alone is justified
by these results. Multiple controller settings changed together, so this run
does not isolate one causal parameter. No further runtime policy is adopted.

## Artifacts and preservation

Native videos and records are in the `stationary-v1` / `motion-v1` folders of
`F:/CRA-experiments/crate-v3-controller-20261105/`. The full failed video retains
the pauses and guard stop. Compact reports, command diagnosis, comparison
figure, exact source/refresh audits and configuration parity are under
`outputs/crate-v3-controller/`.

All 408 prior final-demo files have unchanged SHA256 hashes. Three completed
historical JSONL logs were losslessly NTFS-compressed to provide temporary
disk headroom; before/after SHA256 values match, paths and bytes are unchanged,
and nothing was deleted. The new archive is separate from every older trial.
Keep default-OFF/draft/main/primary/PR117 holds. No full-scene V3 promotion.
