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

Native outcomes pending. A successful route would not establish continuous
whole-crate visual retention, calibrated sharpness, generic four-factor sensing
or a full-scene V3 result. Keep default-OFF/draft/main/primary/PR117 holds.
