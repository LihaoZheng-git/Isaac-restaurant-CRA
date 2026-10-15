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
- The original 2017 Image2Mass weights have an offline numeric comparison, not
  a runtime integration. Keep its legacy-unit conversion, published-output
  conformance gate and license gap explicit. Both mass networks are category-free
  internally; the existing runtime class gate remains a domain safeguard.
  See `docs/image2mass_comparison.md`; oracle/catalog benchmark results must
  not remove current uncertainty or domain gates.
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
- The paired positive-drop fixture is potential-only: target location/height
  selection and matched mass/interaction are configured priors, other factors
  stay unknown and aggregate phi stays conservative. A farther observed landing
  band is not an exact impact site. Require a separate complete raised-object
  perception/platform-occupancy gate before its wheel experiments; older reports
  predating current planar-support checks are not current validation.
- The paired raised-sphere test has a separate explicit RGB instance-geometry
  mode: class-trained YOLO misses remain documented. Its opt-in LiDAR-seeded
  chromatic proposal requires a separated matte uniformly colored object and
  contrasting background; class stays unknown, heuristic confidence stays .5,
  and no authored color/mask/height enters estimation. Exact-mask geometry and
  unchanged sphere-fit gates separate object mass from support returns.
- Retain all unassessed support/environment occupancy with conservative risk.
  In a planar field, unknown support can completely dominate a raised object's
  computed phi. Report per-yaw risk-cell diagnostics and keep matched-geometry
  computed-policy replay explicitly offline, with excluded support/environment
  and no wheel motion. Neither replay nor differing platform paths establishes
  a potential-only causal clearance effect. See the paired potential test report.
- The opt-in SAM 2.1 Tiny adapter uses an observed LiDAR pixel prompt and an
  audited local checkpoint through the existing Ultralytics backend. Class
  remains unknown; accepted instance association is distinct from a known class.
  Keep model mask quality separate from heuristic evidence confidence, reject
  ambiguous/boundary/contaminated proposals, and retain all unmatched occupancy.
  See `docs/category_free_perception.md`; the first stationary glossy sphere was
  recognized by YOLO and must not be described as a detector-miss demonstration.
- Category-free mass uses the separate opt-in `estimate_instance` API with
  a fresh accepted SAM association and live LiDAR geometry. The legacy class
  guard stays intact. No density or category is a network input; retain the
  domain-extrapolation marker, original confidence cap and engineering bounds.
  Unknown class does not mean missing instance association. These live results
  establish rough-heaviness dataflow, not calibrated general-object mass.
- Runtime readiness/report mode follows actual factor-input provenance, including
  fall-interaction priors, rather than only the ground-continuity fixture flag.
  Selecting prior mode never accepts unavailable, fallback or oracle inputs.
  The SAM raised matte fixture is an actual YOLO mask miss; its support/landing
  drop uses the explicitly recorded 2.5 m search scope, not authored height.
  Larger ground fixtures can fail the unchanged bottom-uncertainty gate even
  when mass inference succeeds; keep potential unknown and block full motion.
- Category-free ground, raised (actual YOLO miss), and positive-plume wheel
  trials completed under unchanged computed phi and fixed lambda. This is a
  stationary-estimation/frozen-field endpoint gate, not continuous moving-camera
  estimation, calibrated mass, physical sharpness, temperature or a causal CRA
  advantage. Require actual arrival and safe stopping in later tests; the
  category-free report's `--require-motion` rejects stationary-only records.
  Preserve failed larger-ground/shorter-landing diagnostics and the documented
  conservative uncertainty/support dominance. See `docs/category_free_perception.md`.
- The next accepted target is live perception/CRA/field updates during continuous
  navigation, without scheduled observation pauses. Keep safety stops and the
  stationary development gate. Epoch alignment uses acquisition-time ego pose
  history without extrapolation; asynchronous results retain their source time.
  Camera metadata currently lacks an independent acquisition clock, so the
  simulation read-time approximation and oracle pose/extrinsic references must
  remain explicit. That epoch-only gate established neither per-return deskew
  nor a real-time guarantee; the later opt-in scan audit is described below.
  See `docs/live_cra_navigation.md`; frozen-field trials are not live validation.
- The moving-view plume component is a separate registered clean-reference RGB
  method. It requires explicit clean-reference validation, background/source/
  coverage/photometric gates and a validated negative-observation scope; a failed
  registration stays unknown. Static visible plume may count; camera/plume
  motion is not itself temperature. Synthetic image shifts and stationary
  sensor replays do not establish actual moving-camera acceptance. Preserve the
  original fixed-camera method and their separate domain/provenance records.
- The live four-factor development branch has a successful prior-assisted
  conservative navigation trial, not completed moving-view perception
  acceptance. In `live-positive-v3`, 548 fresh fields and actual arrival were
  recorded, but only 39 updates had an accepted visual instance and only one
  had all four factor inputs. Preserve those failures and unknown/phi=1 results;
  do not report live field commits as complete physical estimation. Keep that
  branch unmerged until its intended perception acceptance is sound. See
  `docs/live_cra_navigation.md` for the remaining gate.
- Live static-scene occupancy memory is an explicit PRIOR, retaining positive
  geometry with original source times through sensor blind regions. It does
  not refresh missing factor evidence, infer free space or support moving humans.
  Gravity-aligned filtering uses the declared oracle pose/ground reference;
  scan-interval occupancy unions are not per-ray deskew. The renderer history
  warning remains unresolved even when its requested setting is 128.
- The live plume opt-in may rectify calibrated fisheye pixels, use above-source
  independent background and recover registration from the originally validated
  clean reference. Preserve the separate coverage/valid-pixel gates and legacy
  defaults. Do not silently accept poor registration, use an arbitrary new clean
  reference or reinterpret absent plume as measured low temperature. A SAM retry
  may select another actually observed owned 3D return, keeping the original
  mask quality, boundary, ambiguity and LiDAR association gates.
- The user approved bounded temporal retention of earlier mass/contact cues.
  `TemporalObjectEvidence` is separate from static occupancy memory: default
  source-age limits are 3 s for mass and 1 s for contact, with identity,
  point-count, competitor and contact-viewpoint gates. Borrowed values retain
  original timestamps/frame IDs, become persistence PRIORS, lose confidence
  and widen bounds; repeated borrowing never renews them. Preserve stronger
  contact cues only until their original expiry. Do not extend this allowance
  to motion, plume, route interaction, support or fall-height inputs.
- The temporal gap diagnostic uses real stationary sensor frames with explicitly
  injected missing SAM outputs. It is not natural occlusion, moving-view
  perception or sensor-only acceptance. The demo separates navigation gates,
  fresh-instance counts, per-factor readiness and retained-property counts.
  Keep unknown potential/thermal handling intact even when temporal mass or
  contact is available. Crowded identity merges/crossings remain outside this
  bounded separated-object validation.
- The live demo now audits copied RTX per-return offsets and explicitly opts
  into ego-motion deskew for audited SENSOR/NONCOMPENSATED/identity-transform
  scans. Keep original header/return epochs and oracle pose/extrinsic provenance;
  reject unsupported compensation, mixed/missing poses, excessive gaps and
  future returns. GMO start/end poses describe a final tick in the observed
  accumulated scan, not full-scan endpoint poses. Deskew does not correct moving
  actors or establish an independent RGB acquisition clock. Preserve the legacy
  whole-scan baseline and the unresolved renderer-history warning.
- Preserve detailed sphere/ground/support and SAM candidate rejection metrics.
  High model mask quality does not override truncation/foreign-return gates.
  Close approaches can legitimately lose LiDAR curved-surface coverage; turns
  can move a sphere out of the front camera. Independent oracle-selected scan
  diagnostics never supply geometry or factor inputs. The eight-second moving
  audit is not waypoint-completion acceptance; moving plume registration remains
  unresolved and its rejected alternatives must not be described as integrated.
- Optimizer candidate age is separate from verified command execution. The
  live demo's legacy 0.2 s age cap has an explicit opt-in up to 0.5 s; never renew
  original optimizer source times or physical input/field freshness. Every
  accepted candidate must pass full exact DTCBF/dense geometry validation from
  the current state against the current field, again on field changes and after
  0.15 s. Verified execution expires after the first 0.2 s interval; analytical
  CBF checks remain per physics step. Preserve failed/future/expired stops and
  report this as a bounded runtime adaptation, not paper equivalence/real time.
  The no-plume and positive-plume reuse trials reached their waypoints but each
  had only one complete factor assessment; navigation success is not moving-view
  perception acceptance. The proposed replacement plume method is a pending
  research choice, not an integrated model or a granted training/data decision.

- The user's later policy decision supersedes the automatic unit-risk handling
  for the new opt-in persistent multi-object experiment. Every current LiDAR
  track gets four numeric control scores using accepted evidence or configured,
  explicitly labeled factor priors. Default uncalibrated priors are .25 per
  factor (all-unknown noisy-OR phi .68359375); retain the legacy strict policy
  separately. Numeric policy completeness is not complete visual estimation,
  and unavailable physical inputs must remain visible in diagnostics.
- Persistent multi-object memory may retain mass, visible contact geometry,
  support/drop and the initial plume cue under an explicit unchanged-scene
  session PRIOR (default original-source expiry 120 s). Complete the explicit
  stationary plume acquisition window before freezing its cue; a clean reference
  alone is not an initial source assessment. Preserve source epochs/confidence,
  reject ambiguous/broken identity, and invalidate support after displacement or
  a visible vertical change. Height is only an invalidation cue, never fall
  height. Motion and route interaction remain current; heat/support changes
  outside the declared scope require reassessment.
- The new all-track path retains exact accepted-mask occupancy ownership. Its
  opt-in static 3D bounds support a separately labeled positive association
  PRIOR through brief mask gaps; unmatched support/environment and unclustered
  returns receive their own factor priors. A LiDAR cluster need not be one
  physical object. Do not claim every environmental object is visually assessed
  or every return lies within the bounded field. Report per-object policies,
  prior uses, readiness and discarded/out-of-domain geometry separately.
- After initial acquisition, the new map freezes accepted intrinsic property
  values under its explicit unchanged-scene prior. Vision retries use an
  explicit one-second per-track cadence and a two-track per-update budget with
  oldest-attempt fairness; all tracks still get current motion and numeric
  policies each processed scan. Support invalidation/expiry remain effective.
  Connected occupied regions with identical phi/flux/lambda/side may share a
  field component to avoid arbitrary identity seams. Preserve differing
  policies, disconnected regions, original track metadata and legacy defaults.
- A temporal source gap in the new map may record an unpublished numerical
  baseline for recovery; it must return no control field and leave accepted
  tracker/evidence state uncommitted. Resume only from a real fresh source pair
  within the unchanged derivative limit, retaining both epochs and all field/
  command freshness and exact verification gates. Never substitute missing
  temporal evidence with zero during motion or renew source clocks.
- The separated multi-object fixture opts into 3D connected clustering with
  planar association/velocity, sparse projected-depth visibility and a separated
  foreground mask depth-layer assumption. Keep original return IDs and all
  physical occupancy; nearby foreign/ambiguous masks still fail unchanged SAM
  quality/coverage/boundary gates. These are bounded association adaptations,
  not complete 3D segmentation or directly measured camera visibility.
- The opt-in equal-height landing mode may infer vertical support-to-landing
  drop when all candidate lower landing bands agree, while impact direction/site
  remains unknown. Keep the spherical bottom consistency gate, explicit 2.5 m
  search scope and worst-case fall-interaction PRIOR separate; height is never
  fall height. Preserve strict directional defaults and mismatched-band failures.
  See docs/restaurant_readiness_gate.md; numeric completeness alone cannot pass
  its independent fixture-evidence or combined navigation acceptance.
- The moving-sphere opt-in may translate accepted instance bounds using current
  observed centroid displacement only after persistence identity/competitor
  checks. Preserve original epochs and the unchanged-shape planar-translation
  PRIOR label, legacy static bounds, unmatched occupancy and moved-support
  invalidation. Moving-actor diagnostics use .15 s positive-occupancy expiry;
  never enable permanent static occupancy memory for them. Scripted .12/.24 m/s
  motion is oracle evaluation truth only, never estimator input. Accumulated
  scans/centroid filtering can underestimate speed; report transition exclusions
  and errors, not exact dynamic velocity or crowded tracking acceptance.
- The combined readiness experiments preserve failed tight mixed-risk layouts:
  positive clearance and stopping do not establish arrival or global unicycle
  convergence. A separately spaced fixture is a different bounded scenario,
  never a retroactive pass. A no-progress watchdog can only reject a trial.
- Vectorized configuration expansion preserves exact occupancy and legacy
  risk/flux/known-ID/first-source metadata priority. Eight-connected equal-policy
  regions and all nearest-distance ties retain the same numerical field (tested
  through h_full/gradients), not paper equivalence or a new safety theorem.
- Calibrated plume-view FOV is explicit (legacy 140 degrees; separated fixture
  opt-in 120). Keep native source pixel counts through remapping: upsampling
  cannot launder an unresolved native source. Preserve all confidence, source
  epoch, clean-reference, background, coverage and ambiguity gates; no claim of
  temperature or moving-view plume acceptance follows from this stationary view.
