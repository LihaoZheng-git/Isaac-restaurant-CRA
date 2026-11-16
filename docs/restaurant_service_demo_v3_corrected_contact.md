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

## Native result at 01bc981

All six fresh prerequisites, the mission's own60 stationary gate, and full native
arrival/stopping passed with frozen source hashes. The robot reached 8/8 guides
in 155.10 simulation seconds, with 1,128 live fields (1,123 while translating),
0.355711 m minimum independent clearance and 0.002195 m stopping drift. Minimum
CBF margin was -1.39e-17, a roundoff-scale value within the original 1e-9 analytical
tolerance; no tolerance was changed. Preserve the one rejected camera/LiDAR epoch.

The nearby human curve was 0.380220 m relative to the actual nominal polyline.
Minimum human-center separation was 1.418761 m; 30 rows had supported moving
kinetic evidence. All 265 human-envelope rows passed, with maximum center error
0.147907 m. No translation hold below 0.02 m/s was measured during nearby walking;
the slow interval below 0.05 m/s was 0.016667 s. This is one paced run, not a
matched smoothing comparison or a pause-free whole-mission claim. The evidence
audit retains 18 uncertain-human rows, 225 body joins, zero visual-part joins
and zero native recovery events.

## Contact acquisition remains limited

The independent point audit distinguishes fresh visual surface assessments from
qualified retained whole-object evidence:

| Prop | Fresh contact assessments | Retained current-point rows | Result |
| --- | ---: | ---: | --- |
| Rounded bin | 20 | 81 | Retained cue 0.304051; phi 0.614051-0.617660 |
| Broken crate | 2 | 0 | Fresh cues 0.493514/0.568536; phi 0.602145/0.605926 |

The crate assessments were at sources 45.0 and 52.7 s, raw ID407. Neither seeded
the retained group required by the approved static-property policy. This is an
initial acquisition limitation, not evidence that stored crate properties were
overwritten. Between accepted assessments it remains ordinary occupied geometry
under the existing unknown-obstacle policy. Do not transfer the short test's
12/12 retention result to this moving-acquisition mission.

All selected prop returns pass the original 3 cm independent mesh-box allowance.
The separate 20 cm whole-center diagnostic passes only 50/81 retained bin rows,
19/20 fresh bin rows and 1/2 fresh crate rows. Observed surfaces are not whole
object centers. Fresh bin cues range from 0.009197 to 1.0, so the correction has
not eliminated every view-dependent contact spike. The crate's two cues exceed
the retained bin cue, but this is not a dependable paired sharpness ranking;
total phi also includes other factors, interaction and priors.

The fresh stationary benches each accepted 58/60 target assessments. Median
bin/crate cues were 0.362731/0.424374, contact factors 0.115938/0.121492 and phi
0.595967/0.598511, with overlapping ranges. Preserve all startup rejections and
unfavorable values. Rough mass remains uncalibrated, potential/thermal use labeled
priors, and simulator ego reference remains an oracle. No complete fresh generic
four-factor or physical-sharpness validation is established.

## Audits and presentation

Source/point retention passed for 1,314 rows across 15 objects. Other audits passed
for 37,360 geometry references/750 archives, 1,115 ambiguous motion proofs,
3,882 fragment proofs and 1,130 camera masks/63 tracks. These verify provenance
and presentation parity within their stated scope, not general perception safety.
See `outputs/service-demo-v3-corrected-contact/long-motion-v1/` for original reports,
especially `contact-coverage.json`, `static-prop-audit.json` and
`actual-motion-summary.json`.

The new video is `outputs/final demo/v3-corrected-contact/restaurant-service-v3-map.mp4`:
left perception, upper-right map, lower-right follow camera. It retains all 1,562
navigation/stopping frames plus 60 credit frames, at 1x simulation playback. Native
execution was compute-paced at 0.025, not real time. All 324 previous final-demo
files remain byte-identical. Lossless NTFS compression of 19 completed sensor logs
was separately hash-verified; no records were deleted or rewritten.

Keep default-off/draft PR236/main/primary/PR117 holds; no merge or overwrite of
previous recordings. Navigation passed; continuous crate acquisition and reliable
contact ranking remain unvalidated.
