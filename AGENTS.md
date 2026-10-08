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
- Keep the bounded thermal demonstration to a visible steam/smoke-like plume
  spatially associated with an open soup container. Generic smoke alone is not
  hot-soup evidence, and a visible plume is not a temperature measurement or
  proof that the aerosol is steam.
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
