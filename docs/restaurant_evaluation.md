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
Actual sensor trials still need bounded relative-heaviness validation,
positive fall-surface evidence, an explicit visual contact-proxy adapter and
a validated continuous-plume estimation method. Wheel motion should only begin after
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
  sharpness/hardness remain future work. The current physical contact adapter
  still requires those physical inputs; an explicit proxy path must be added
  and tested rather than populating missing inputs with invented values.
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
