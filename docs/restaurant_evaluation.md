# Restaurant evaluation protocol

Primary comparisons use matched scenario, seed, geometry, nominal waypoint,
initial state, solver settings, and sensor conditions. The baselines are the
geometric velocity-aware CBF, fixed-class Safe-SAGE-style policy, and continuous
CRA Safe-SAGE policy. Primary Safe-SAGE comparisons require lambda 1.25.
Risk-dependent lambda is a separately labeled ablation.

Each trial records evidence mode (`sensor`, `prior`, or `oracle`), completion,
collision, fail-closed state, minimum clearance and CBF margin, interventions,
fallbacks, and elapsed time. Oracle/prior trials cannot be reported as sensor
performance. Baseline summaries are grouped by evidence mode so oracle, prior,
and sensor results cannot be pooled into one completion, collision, or
clearance statistic. Repeated trials and uncertainty checks remain required for
conclusions within the selected restaurant scenarios. Broad domain transfer is
future work.

The protocol and validators are implemented; closed-loop restaurant
experiments remain pending. A local exposed-edge contact checkpoint and visual
mass baseline can be invoked, but their complete factor accuracy is unvalidated.
Component results now include catalog relative-heaviness validation, a visible
positive fall-surface fixture, an explicit visual contact-proxy adapter, and a
prior-assisted fixed-camera RGB plume method (see results below). Their input
domains and limitations remain separate. Full sensor-derived restaurant
acceptance is still pending. Wheel motion should only begin after
stationary scenario-specific perception acceptance checks pass. See
[stationary benchmark results](stationary_sensor_gate.md).

The stationary readiness gate enforces that requirement. Sensor trials accept
only measured or inferred inputs; prior demonstrations may additionally use
explicit priors; oracle trials require the explicit `oracle` estimate status.
Fallback and unavailable evidence never pass readiness. Four-factor scope
requires all four diagnostics before aggregate `phi` can be reported.
Factor-specific scope can validate one hazard family, but explicitly leaves
aggregate `phi` unreportable so unknown factors are not silently treated as
zero.

## Accepted thesis scope

The user confirmed that the features need to support the selected final
restaurant settings. Universal object coverage and deployment-quality
perception are not completion criteria for this thesis.

- Reuse the existing object detector, visible LiDAR geometry, and tracker for
  a stationary-object baseline and a clear crossing case. Complete object
  reconstruction, extensive occlusion, slow-motion coverage and crowded scenes
  remain future evaluations, not blockers for those bounded cases.
- Use continuous approximate heaviness. Check lighter/heavier ordering and
  resulting factor-score trends on independent objects; exact kilogram accuracy
  and a large general-object mass study are secondary.
- Demonstrate potential risk with one elevated-object case whose support and
  landing evidence justifies the drop. General toppling/stability is deferred.
- Demonstrate contact with an explicitly named continuous visual exposed-edge
  geometry proxy. Broader nails/tips/protrusions/jagged surfaces and physical
  sharpness/hardness remain future work. The opt-in proxy path is implemented
  and tested; the default physical contact path still requires those physical
  inputs. Missing inputs are never populated with invented values.
- Demonstrate thermal evidence with a continuous visible plume associated with
  an open container. Temperature, burn severity, aerosol identity and a general
  thermal model are outside the primary claim. A learned model is not mandatory,
  but the selected image-based estimation method must pass bounded positive and
  low-cue tests before being reported as sensor evidence.

The sequence is controlled component tests outside the restaurant, a stationary
combined CRA benchmark, restaurant enrichment, and matched navigation trials.
Scene additions should serve these factors: ordinary obstacles, a crossing
human, an elevated object, an exposed edge and a plume-associated container.
The study evaluates continuous visual/physical risk proxies in these scenarios.
Unknown handling, provenance separation, online intervention and fixed primary
lambda 1.25 remain unchanged. This scope decision does not itself change
runtime code, relax existing readiness gates, or turn a failed test into a pass.

## Visible support component result (2026-10-02)

The stopped-robot Pandar diagnostic now selects geometrically valid support
bands before ranking their density, rejecting narrow vertical-face bands.
Partial directional coverage cannot imply zero local drop. The fixture filter
also uses the settled chassis frame instead of the authored pre-settling pose.
An isolated 0.300 m support-to-floor case estimated 0.30117 m, covered the
oracle in its evidence interval, and drifted 0.00120 m / 0.0100 degrees with
zero wheel commands. This uses live LiDAR; authored target location and height
select the controlled cluster, and authored heights score it offline.

The explicitly recorded diagnostic configuration uses a 1.50 m landing search
because the mounted scanner first sees the floor beyond the platform at that
distance. Runtime defaults remain unchanged. Replay of the corrected 0.800 m
fixture remains unavailable because its support top is occluded. These results
do not establish tabletop coverage or harmful-fall interaction likelihood.
Artifacts: `outputs/visible_support_estimation/isolated-030-final/`.
Validation: 1,286 unit tests passed; repository Ruff and whitespace checks passed.

## Relative heaviness component result (2026-10-02)

The existing frozen ABO model was evaluated on all 152 held-out objects without
retraining or selecting successful examples. For catalog masses separated by
at least 3x (chosen before inference), 6,758 / 6,943 pairs had the correct
lighter/heavier order (97.34%); within the same class, 587 / 646 pairs did
(90.87%). Pairs share objects, so these are descriptive ordering metrics,
not independent statistical samples. All 185 reversals are retained in the
report. Catalog dimensions remain external metadata, not live sensor evidence.
This supports bounded catalog relative heaviness, but does not validate mass
from the robot's incomplete visible envelope. The live 0.25 confidence cap and
uncertainty bounds are unchanged. Exact kilograms remain a secondary claim.
Artifact: `outputs/relative_heaviness/report.json`; checkpoint SHA-256
`92d437580efb283071e0ac0cd7c16de77f76ef3017a243329e4bf826e0e5a31e`.
Validation: 18 relevant tests passed; repository Ruff passed.

## Explicit visual contact proxy (2026-10-02)

Accepted exposed-edge masks now produce a continuous score from exposure,
elongation and boundary complexity. The default shape weight is configurable
(0.6 elongation / 0.4 complexity); it is an engineering proxy, not injury
calibration. Adapter use requires `allow_visual_contact_proxy=True` and a
separate interaction estimate. Diagnostic and readiness records label the
basis `visual_exposed_edge_proxy`. Physical sharpness, hardness and 3D edge
orientation remain unavailable; absent or unmatched part detections remain
unknown. Existing physical-contact behavior is unchanged by default.

Held-out PACO replay accepted four edge masks on three parent objects, producing
nominal proxy values 0.258, 0.273 and 0.354 with distinct confidence/bounds.
Six ambiguous and three unmatched masks were rejected. This validates the
image-to-proxy path on that replay, not general restaurant contact accuracy,
negative absence, tips, nails or jagged surfaces. Local checkpoint and image
hashes are recorded in `outputs/contact_visual_proxy/diagnostic.json`.

## Fixed-camera RGB plume component (2026-10-02)

The classical RGB baseline subtracts an observed clean reference, retains
bright low-chroma components above a supplied source region, and measures their
spatial alignment. It supplies image-derived masks to the existing area,
vertical-extent and persistence estimator. Thresholds, source-relative search
size, background-change rejection and confidence cap are configurable. Empty
masks become low **visible-cue** evidence only with explicit bounded-domain
negative validation and confidence; defaults still leave them unknown. Low
visible cue never asserts that an object is cold.

Eighteen actual Isaac RGB captures (six per condition) gave continuous thermal
proxy values 0.000, 0.294 and 0.968 for no cue, intermittent small plume and
persistent large plume. The existing cartoon fixture appearance is unchanged.
Plume masks are inferred from pixels; the configured container region and
open-container context are **priors**, so the final thermal score also retains
prior provenance. The generic object detector returned no container detection
on this fixture. No oracle segmentation is used by the estimator.

This method requires a fixed camera and a clean reference. Global background
changes reject the reference; grey foreground objects can still confound it.
It is a controlled demonstration, not a general steam/smoke detector, moving
camera solution, temperature measurement or steam-identity classifier.
Artifacts and overlays: `outputs/rgb_plume_isaac/`; 27 relevant tests and Ruff
passed. Restaurant source detection and moving-camera validation remain pending.

## Object evidence to CRA and Safe-SAGE (2026-10-02)

The per-track profile provider accepts explicit visual contact proxies and
separate contact interactions. The stationary fusion demo enables this with
`CRA_USE_VISUAL_CONTACT_PROXY=1`, associates accepted masks only to one matching
LiDAR track, and explicitly uses geometric route crossing as a bounded contact
interaction proxy. This is not 3D edge orientation or collision probability.
Ambiguous parent/track matches and absent masks remain unknown. Factor-specific
records retain the proxy basis as well as their input provenance.

A fresh stopped-robot restaurant capture matched three chair tracks, drifted
0.00144 m / 0.00275 degrees, and sent zero nonzero wheel commands. All three
had inferred mass and kinetic inputs; thermal, contact and potential remained
unknown. Conservative aggregate phi stayed 1.0 and was **not accepted as a
validated four-factor sensor result**. No factors from independent fixtures
were spliced into the chairs. Replay through the actual occupancy, configuration
space, guidance and h_full solvers yielded a finite 3 x 24 x 28 field with
lambda 1.25. This validates the evidence-to-policy/field connection, not robot
navigation or universal safety. Artifacts: `outputs/stationary_capture_safe_sage/`.

Integration validation: 1,305 unit tests passed; repository Ruff and whitespace checks
passed. One earlier full-suite invocation had temporary-directory setup errors;
the corrected run completed without errors. All simulations remained stationary.

Next: enrich the restaurant with the agreed factor-specific actors and validate
their visibility/source association stationary. The bounded plume demo can
initially use clearly labeled fixed-camera/source priors; moving robot-camera
plume perception remains unsupported. Check relative heaviness on the selected
rendered objects rather than demanding precise kilograms. Wheel trials still
require the scenario's stationary readiness records. Fixed-class and continuous
CRA baselines retain matched settings and primary lambda 1.25.

Final support-coverage review: a surrounding small support patch must not
establish zero local drop when farther surfaces are unseen. Zero-drop evidence
now requires support-height returns reaching the outer 80% of the configured
search neighborhood in every sampled direction. The fraction is configurable;
this remains a bounded local observation, not a general toppling claim. The
0.30117 m positive sensor replay is unchanged. The new small-platform regression
passes alongside the existing ground-support and positive-drop cases.
Final full regression: 1,306 tests passed; Ruff and whitespace checks passed.
