# CRA repository guidance

- Keep waypoint nominal control obstacle-unaware; MPC and analytical CBF create online intervention.
- Runtime CRA uses visual/physical estimation without language or VLM reasoning.
- Preserve measured, inferred, prior, oracle, fallback, and unavailable provenance. Missing evidence is not zero. Object height and center height are not fall height.
- Evaluate factor-specific diagnostics when unknown contact/thermal inputs saturate aggregate `phi`; never mark unsupported hazards absent to improve results.
- Keep lambda fixed at 1.25 for the primary comparison. Treat risk-dependent lambda as a separate ablation.
- Verify candidate weights, licenses, datasets, and compatibility before claiming a visual model is integrated.
- Treat contact risk as general hazardous geometry: exposed edges, pointed tips,
  protrusions, and jagged surfaces can occur beyond restaurant utensils. A
  knife/fork specialist is a bounded demonstration, not the full contact model.
  Keep unsupported sharpness, material consequence, exposure, and route
  orientation evidence unavailable rather than inferring safety from class names.
- The current primary thermal factor is an explicitly named object-associated
  visible-plume proxy. The user's later decision permits validated low/no-plume
  observations to produce low cue scores for any clearly observed object.
  Hot soup is one example; open-container classification is not required in this
  mode. Preserve the older contextual container mode separately. Generic smoke
  is not hot-soup evidence, and a plume is not temperature or aerosol identity.
- Validate perception stationary before enabling new wheel-motion experiments. Report oracle/prior demonstrations separately from sensor-derived results.
- Preserve user changes. Do not force-push, rewrite history, bypass checks, upgrade dependencies silently, or merge unvalidated work.
- The accepted thesis scope is sufficient support for the selected final
  restaurant scenarios. Treat broad generalization, complete 3D reconstruction,
  crowded/occluded multi-object motion, general outdoor contact perception, and
  comprehensive physical calibration as future work rather than prerequisites.
- Mass acceptance prioritizes continuous relative heaviness and sensible
  potential/kinetic risk ordering in a small independent test set. Exact
  kilograms and a large general-object mass study are secondary.
- The primary contact demonstration may use an explicitly named continuous
  visual exposed-edge geometry proxy. Preserve the general hazard taxonomy for
  future work, and never relabel that proxy as physical sharpness or hardness.
  Implement and validate its explicit adapter/readiness path before using it
  in an aggregate runtime experiment; do not fill unsupported fields with zero.
- Validate components first in controlled scenes or held-out inputs outside
  the restaurant, then a stationary combined benchmark, then enrich the
  restaurant and run matched navigation comparisons. Acceptance is bounded to
  those scenarios, not every possible object or condition. A working plume
  estimation method is still required for a sensor-derived thermal claim.
- The implemented visual contact proxy requires explicit adapter opt-in and
  separate interaction evidence. Preserve its `visual_exposed_edge_proxy`
  basis in diagnostics; physical sharpness, hardness and 3D orientation stay
  unavailable. An unmatched part mask is not negative contact evidence.
- The current RGB plume baseline is a bounded fixed-camera reference method.
  Preserve configured container regions/context as priors, reject invalid
  references, and do not claim moving-camera support. Validated low visible
  plume cues do not establish that an object is cold.
- The simple-object demo separates oracle factor profiles from its measured
  LiDAR occupancy and unchanged unknown sensor assessment. Require actual
  waypoint completion for motion acceptance; positive clearance alone is not
  a successful pass-by. See `docs/simple_object_test.md` for its bounded scope.
- The default simple-object test must use the algorithm's observed evidence
  assessment unchanged, including conservative phi saturation. Supplied low/high
  profiles require explicit `--oracle-comparison`. Record complete perception
  readiness separately from uncertainty-aware controller acceptance: unknown
  evidence may drive conservative control, but is never a validated low factor.
- The simple-object controller may enable bounded online turn seeds to escape
  nearly stopped MPC horizons. Keep the nominal obstacle-unaware, retain the
  same objective and exact DTCBF/dense geometry acceptance checks, and report
  this as a unicycle implementation adaptation rather than global convergence
  or exact paper equivalence. Other MPC callers retain the legacy default.
- The accepted static simple-object gate does not establish real-time control,
  monotonic risk-to-clearance ordering, or enforced passing side. Its configured
  side is soft guidance; retain the observed side and planning latency in later
  evaluations. Validate selected restaurant actors stationary before new motion.
- A resolved silhouette may support the explicitly named
  `visual_silhouette_geometry_proxy`; it establishes only visible-outline cues,
  not hidden-surface safety or physical sharpness/hardness. Preserve stronger
  accepted part cues and reject ambiguous/frame-mismatched associations.
- Fitted spherical size may support rough heaviness only with explicit effective
  density evidence. A configured density keeps mass PRIOR; there is no universal
  ball density. Keep the sphere-shape extrapolation and uncertainty explicit.
- Object-plume negative evidence requires a validated clean reference/domain,
  distinct stationary frames, usable air/source regions and unambiguous fresh
  association. Invalid/missing observations stay unknown. Other sources' air
  regions may be excluded from reference-background checks only while enough
  independent background remains; shared positive plumes remain ambiguous.
- The simple sphere test may explicitly opt into a bounded local flat-ground
  continuity PRIOR when its mounting creates a LiDAR blind region. Require an
  observed floor plane and a consistent fitted sphere bottom; the bottom is a
  consistency gate, never fall height. Keep inferred plane evidence distinct
  from assumed unseen support/landing continuity, reject elevated/uncertain
  fits, and preserve the strict sensor-only surface path and its rejections.
- Ground-prior trials use a separately labeled worst-case harmful-fall
  interaction PRIOR with upper bound 1. They must pass four-factor readiness
  in PRIOR mode before motion and must not be reported as sensor-only results.
  High phi from low-confidence evidence is an uncertainty-policy finding; do
  not change confidence, aggregation or fallback values to make a sphere safe.
- Paired plume fixture results are bounded fixed-camera RGB demonstrations.
  Preserve the documented low-contrast puff miss; valid observation gates do
  not guarantee plume recall or establish absence of heat. Rendered puffs never
  supply factor scores, and endpoint navigation freezes the stationary field.
- Seed-color outline refinement requires explicit validated uniform-color and
  separated-object scope. A class-trained body mask can omit a real protrusion;
  preserve that documented miss. The optional spherical-body mass hypothesis
  needs an explicit whole-assembly density PRIOR, never discards occupancy, and
  does not bound appendage mass by its fraction of LiDAR returns.
