# Corrected contact in the long service demo

The user approved adding the short-test correction to a separately saved demo.
Use `configs/restaurant_service_demo_v3_corrected_contact.json`, stacked on the
accepted open-aisle V3 scene. Only the existing bounded contact-contour correction
is enabled in the two prop benches and mixed service commands. Its CLI now allows
the already-gated mixed static-prop/human mode. All original mixed-scene guards,
person exclusions, stationary prerequisites, source clocks, evidence retention,
controller settings and acceptance thresholds remain.

The licensed crate/bin positions, original long waypoint route, raised Owl view,
one-way human walk, fixed lambda 1.25 and left-perception/right-map-and-follow-view
presentation are unchanged. No supplied contact ranking or fixed final phi is
introduced. Previous V3 videos and failures must remain byte-identical.

Run fresh bin/crate stationary tests, actor and human-sensor components, startup
and near-human stationary gates, then the full mission with its own60 startup.
Use new outputs under `outputs/service-demo-v3-corrected-contact/`. Require actual
arrival, stopping, source/point and geometry/camera audits before creating the new
presentation under `outputs/final demo/v3-corrected-contact/`.

The short tests establish retained mass/contact on current LiDAR support after
stationary acquisition. They do not establish acquisition while moving along the
full route. In the crate short test, two new fragments lacked velocity history;
configured kinetic/interaction fallbacks briefly raised phi despite unchanged
intrinsic contact. Preserve such updates. Relative velocity can also be sensitive
to fragment membership. Rough mass and outline contact remain inferred proxies;
potential/thermal retain labeled priors. No complete fresh four-factor, calibrated
sharpness, real-time or universal-safety claim is made.

Source validation: 84 focused tests and 3,875 full unit tests (215.08s) pass;
Ruff and diff checks pass. The first focused run correctly rejected a motion
test without its required human prerequisites; the CLI test now checks that
rejection and verifies delegation to the original prerequisite validator.
No native gate was bypassed. Preserve `focused-v1.log`.

Native integration acceptance is pending. Keep default-off/draft/main/primary/
PR117 holds; no merge or overwrite of previous recordings.
