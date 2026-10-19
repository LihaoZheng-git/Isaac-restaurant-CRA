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
- Large live diagnostic histories may use explicit append-only JSONL logging;
  in that mode live-fields.json is only the latest-five checkpoint. Preserve
  complete records and source times in the stream. The existing fair visual
  attempt budget may be explicitly set to one for paced development, while all
  tracks still receive current motion/geometry and numeric prior-aware policies.
  Never report deferred vision or late rejected results as fresh perception.
- The low moving-sphere trial passed relative-velocity error checks but failed
  navigation after its geometry left the LiDAR near blind region and short
  occupancy retention expired. Numeric factor priors do not repair missing
  obstacle geometry. The moving-actor demo now stops on unresolved nearby
  positive-track loss before publishing that field. Its .15 s/1.2 m watch,
  2 m enrollment and unique .25 m new-ID association are explicit bounded
  engineering assumptions, not a dynamic belief map or general occlusion
  guarantee. The .333 m-clearance expected-stop regression is not arrival.
- The upright capsule is a separate LiDAR-visible kinetic proxy. Its authored
  height is rendering/monitoring metadata only; the sphere ground prior is
  disabled. Its stationary kinetic-component gate permits explicit unknown
  potential/thermal priors and must not be described as complete four-factor
  perception, human mass calibration or physical contact/temperature estimation.
  Preserve the high smooth-capsule silhouette score as a proxy limitation;
  never lower it to improve navigation. See docs/restaurant_readiness_gate.md.
- Euclidean waypoint distance can increase during a valid wide avoidance
  detour. An optional progress-watchdog rejection is a failed trial, not proof
  of global nonconvergence. Bounded retries may disable that optional watchdog
  while retaining the overall time limit, actual-arrival requirement and every
  freshness, continuity, exact MPC/DTCBF, analytical CBF and stopping gate.
- The live caller leaves the optional world-grid lateral recovery hint unset:
  a constant negative-y final-progress constraint can prevent a return toward
  the goal after passing below an obstacle. Keep per-obstacle Safe-SAGE guidance,
  the obstacle-unaware nominal and observed passing-side diagnostics. The live
  MPC may opt into exact command-cost pruning of its recovery pool; unchanged
  ties retain the first feasible candidate and every winner still passes the
  same exact checks. Legacy callers retain exhaustive recovery by default.
  Neither change establishes convergence, paper equivalence or real time.
- A separate opt-in may reject recovery candidates with already failing exact
  DTCBF margins before dense geometry work; every feasible candidate still gets
  all geometry checks. Keep infeasible main-solver references for restoration.
  Both recovery optimizations are disabled for legacy callers. Worker-compute
  diagnostics retain original source epochs and never make late results usable.
  Preserve the failed 90 s capsule retries separately from velocity acceptance;
  positive clearance, CBF margins and stopping alone do not establish arrival.
- The demo's live QP iteration budget is explicit (legacy 4,000; opt-in bounded
  lower budgets). A lower limit changes numerical search and may use exact-safe
  recovery more often; it never relaxes constraints, freshness or stopping.
  Report the budget in matched comparisons and preserve failed/short diagnostic
  trials separately from endpoint acceptance. Worker timing identified exhausted
  QP calls, but does not establish that all historical latency tails share that
  cause. Keep fixed lambda and factor priors unchanged in these runtime trials.
- The live demo also exposes an optional nonlinear-rescue iteration budget
  (legacy 300). A 1,000-QP-only diagnostic still exhausted 300 rescue iterations
  and returned an unusably late rotation recovery. Report both solver budgets
  and the overall simulation bound; lowering them changes numerical search,
  never the objective or exact acceptance. Preserve the original 90 s failures
  if a later separately bounded retry succeeds. Timing fixtures with differing
  trajectories are not matched-input benchmarks.
- The bounded readiness gate now has accepted final-budget upright-actor and
  three-object waypoint arrivals, plus stationary factor and kinetic gates.
  The actor retained one positively owned track across 547 fields; mass/contact
  stayed labeled property priors, motion remained current, and potential/thermal
  used configured priors. This permits selected restaurant scene construction,
  with stationary validation of each new asset before matched navigation tests.
  It is separate from PR #117's unmet fresh moving-view perception gate and
  establishes neither real time nor calibrated human mass/sharpness/temperature.
  See docs/restaurant_readiness_gate.md for exact budgets, scope and failed trials.
- The first restaurant service build uses audited CC0 Kenney meshes in a
  separate additions layer; preserve original restaurant/robot files. Authored
  placement, asset names and bounds are rendering/evaluation metadata, never
  mass, fall-height, heat, contact or desired-phi inputs. No humans/payload or
  emitted plume are included yet. Its opt-in 640x400 acquisition matches the
  camera's native aspect; legacy acquisition/projection and SAM gates remain.
- The new FoodBowl stationary component gate and 2.2 m restaurant waypoint
  trial passed with labeled potential/thermal priors and fixed lambda. Only one
  fresh visual instance was accepted during motion; keep identity fragmentation
  and prior reversion explicit. The independent authored raster is abort/
  evaluation information only. This wide-aisle result is neither full fresh
  four-factor perception, the proposed 30.88 m route, human/payload validation,
  real time nor causal CRA superiority. Do not extend source clocks to cover
  the long route. See docs/restaurant_service_build.md. PR #117's main-merge
  hold remains separate and unchanged.
- Additional `--restaurant-view` pot/counter/front observations failed their
  stationary component gates (occlusion, merged support and ambiguous ownership).
  Preserve these failures and their factor priors; no asset-name/placement input
  or score override repairs them. These extra views remain zero-wheel only.
  A missing first-frame instance may be logged through the restaurant observation
  window, but the final owned-instance/mass/contact/motion gate remains mandatory.
  See docs/restaurant_prop_validation.md; no experimental plane-cut method was
  integrated or used to remove physical occupancy.
- The opt-in restaurant `service` route reached three waypoints over 3.9466 m
  with 204 live fields and verified stopping. Active-waypoint CRA context and
  obsolete-goal planner-result rejection preserve original source clocks and
  safety checks. It stays inside the unchanged fixed field/expiry limits; it
  is not the proposed 30.88 m route, rolling coverage or real-time acceptance.
  Only two fresh visual instances occurred during motion; keep missing-factor
  and retained-property priors explicit. See docs/restaurant_service_route.md.
- The actual restaurant bowl's registered plume gate failed because fragmented
  instance IDs lacked clean references. Preserve this failure and runtime
  thermal priors. A separate explicit fixed-camera replay passed continuous RGB
  cue ordering using a retained observed source box as a source-region/identity
  PRIOR. Its source timestamp remains recorded; it never repairs live identity
  or commands wheels. Its factor-specific phi also uses labeled unit-interaction
  and .25 other-factor priors. Keep the first five positive-window misses and
  no-temperature/no-moving-camera limits. See docs/restaurant_visible_plume.md.
- The historical Kenney restaurant server is an audited CC0 idle-pose mesh with
  project apron geometry, in a separate scene layer. Authored height/role/pose
  never supply factor inputs. Source/UV/collision and zero-wheel previews pass;
  its stationary perception gate remains failed (profile-view final ambiguity,
  frontal silhouette interior boundaries). No server wheel/crossing acceptance
  follows. Its approximately 1.59 kg catalog-network output is not credible
  calibrated human mass or validated human heaviness ordering. Keep unknown
  contact/potential/thermal priors and the exact failed records. Static memory
  cannot support a moving server. See docs/restaurant_server.md.
- The explicit observed-LiDAR SAM context-box retry uses only the current
  track's projected owned pixels and configurable symmetric image padding.
  Keep unknown class, confidence cap, mask quality/coverage/foreign/boundary/
  ambiguity gates, original source epochs and unmatched occupancy unchanged.
  Do not retry an ambiguous result or treat box padding as physical geometry.
  Restaurant robot setups require the independent authored footprint-clearance
  guard; that oracle abort/evaluation information never enters CRA or control.
- The user now prefers the official Isaac Sim human. The primary replacement
  references the audited People/F_Business_02 asset; NVIDIA geometry/textures
  are external, not vendored. Preserve native scale, project-authored static
  arms-down pose, separate grounding transform and guide-purpose collision
  capsule. Authored pose/bounds/collision never enter CRA. The checked idle
  clip has a mismatched skeleton and is not integrated. Asset/render acceptance
  does not establish perception, calibrated human mass or walking/crossing
  acceptance. Retain the old Kenney diagnostics separately. See
  docs/restaurant_isaac_human.md; PR #117's main hold remains unchanged.
- The official Isaac human's stationary component gate passed with 18/18
  accepted instances, fresh relative motion and a visual silhouette proxy.
  Potential/thermal stayed .25 configured priors; final phi was .622961.
  The 1.921 kg catalog-network result is not credible human mass or validated
  human heaviness ordering. This gate establishes bounded dataflow only; strict
  fresh four-factor readiness is false and no wheel/walking trial was run.
- Native behavior retargeting now passed separate parked-robot idle/walk/stop
  and faster-locomotion component gates in the restaurant. Observed medians
  were .692865 and 1.691414 m/s with actual arrival, articulated feet, grounding,
  PhysX collider readback, independent clearance and decoded RGB video. The
  source rig/animation and requested motion remain oracle scene/evaluation
  information, never CRA inputs. The faster trial does not identify a selected
  sprint clip. In this installed Kit build the animation extension must load at
  application startup for RGB capture; late activation failed an isolated
  two-camera probe. Preserve renderer-history warnings and failed early
  recordings. No moving-human perception, human mass prior, kinetic validation
  or robot crossing was implemented by this actor-motion branch; no static
  memory or unchanged-shape bounds prior may be assumed for articulating
  humans. See docs/restaurant_human_locomotion.md. PR #117's main hold remains.
- The separate native-human parked-robot sensor gate passed with 55 fields,
  27 fresh visually associated human mass PRIORS, one human track, 10 matched
  stationary and 8 steady moving rows; median velocity error was .084934 m/s.
  The audited COCO YOLO11n-seg person cue selects configurable 70 [45,100] kg
  adult-demo mass only after fresh unambiguous SAM mask association. It is not
  a learned/measured human mass or an authored-role input; retain the original
  catalog output separately. This fresh-only articulation mode disables all
  property/bounds borrowing and static occupancy memory; .15 s positive
  occupancy remains bounded. Potential/thermal remain .25 priors, contact is
  a sometimes-high visible geometry proxy, and first-frame contact nominal 1
  with zero combined confidence still saturated phi under the unchanged
  conservative policy. Preserve that factor-specific finding and failures.
  A copied scan may wait at most .05 simulation seconds for its declared return
  poses, without extrapolating or changing acquisition epochs. Retry rejected
  acquisitions each physics tick; never relax freshness to fix scan-cycle
  phase locking. The maximum completed field age was .416667 s; this is not
  navigation/control freshness acceptance, real time or robot/human crossing.
  Keep next wheel experiments behind their own current-field/track-loss gates.
  See docs/restaurant_human_perception.md; PR #117's main hold is unchanged.
- The opt-in native-human navigation mode selects the human as its stationary
  component gate; unrelated bowl diagnostics remain recorded with their actual
  failures/priors and physical occupancy. It rejects static occupancy memory,
  uses .15 s positive scan retention, and may retain only intrinsic mass for
  3 original-source seconds with 1.5 s confidence half-life. Old body bounds,
  contact, support, plume and motion cannot be borrowed for articulating humans.
  Preserve .35 s inference/field caps, nearby-track-loss stops and exact current
  execution checks. A longer stationary window does not lower SAM's .85 quality
  gate. Standing/crossing control acceptance is separate from parked perception,
  complete four-factor sensing, carried bowls, crowds and PR #117's main hold.
  See docs/restaurant_human_navigation.md for current executed results.
- The first standing-human navigation trial failed on a lost limb/body fragment
  near a still-observed body, just before arrival; retain it as failed. The new
  human-only guard opt-in requires unique current positive geometry to resolve
  a rejoined fragment: at least 3 returns within .10 m, existing .25 m centroid
  association and recent coexistence. Ambiguous/unclustered/stale geometry or
  true disappearance still stops; no identity/property evidence transfers.
  Native standing-idle is allowed bounded measured sway, never a forced zero
  motion score. The transverse actor-only zero-wheel walk passed separately;
  coupled retries still need actual arrival and stopping. No carried-bowl or
  moving-view thermal acceptance follows from these checks.
- The original service middle waypoint can overlap the recorded native idle
  envelope; preserve failed/stalled original-route runs. The separate fixed
  `human-service` centerline route removes its 20 cm aisle-side corner, without
  changing online nominal/safety logic. Its independent layout precheck is not
  motion acceptance; both standing-idle and crossing must complete this same
  route with unchanged constraints and their actual measured/inferred motion.
- The centerline standing trial passed separately; the first coupled crossing
  failed and had zero supported moving-human kinetic rows because close RGB
  truncation rejected fresh full-instance mass associations. Preserve both
  findings. Do not relax SAM boundary/quality gates or infer mass from cropped
  bodies. Explicit camera mount experiments retain stock defaults, unchanged
  lens calibration, session-only changes and re-read actual extrinsics; require
  a zero-wheel close-view perception gate before new crossing wheels.
- Human-only moving-fragment continuity may explicitly use a bounded stable
  current-parent velocity hypothesis (not oracle actor or lost-limb velocity),
  at most .35 s query age/.25 m translation/2 m/s speed, .15 s previous velocity
  age and .3 m/s change. Unique actual current returns and existing .10/.25 m
  geometric gates remain mandatory. It transfers no factors or semantic identity
  and establishes neither free space nor general occlusion safety. A focused
  copied-scan replay is not a successful navigation trial. See
  docs/restaurant_human_navigation.md and the prospective final experiment plan.
- Close full-human RGB coverage is a documented limitation: tilt-only and raised
  camera attempts failed their moving/stationary perception gates. A separate
  parked crossing-approach view at y=7.4 with stock camera height/+15 degree pitch
  passed a .5 m/s native walk (10 stationary/nine moving fresh human priors).
  The .7 m/s version had only three moving rows and remains failed. Explicit
  actor trigger/speed options change the authored scenario only, never runtime
  evidence or control. A farther/slow crossing cannot establish close-range,
  faster, crowded, complete four-factor or main-merge acceptance.
- The farther .5 m/s coupled crossing (robot-y trigger 7.0, stock-height RGB
  pitch +15) passed robot/human arrivals, stopping and the live audit: nine
  supported moving-human kinetic rows with positive current ownership, seven
  simultaneous rows, all using the configured adult mass prior. Potential and
  thermal stayed priors; a later validator rejects catalog human mass. Preserve
  the median-vs-maximum velocity error distinction and the nine old-centroid
  geometry merges (no velocity-query merge was exercised in that wheel run).
- Keep the factor-specific uncertainty finding: observed relative energy and
  nominal kinetic risk increased during crossing, while confidence-adjusted
  policy scores need not increase because the .25 uncertainty prior/confidence
  changed. This is not a causal comparison or monotonic phi/clearance result.
  Do not silently alter confidence, priors or energy scales to improve the
  figure. Treat calibration/sensitivity explicitly in the final study. See
  docs/restaurant_human_navigation.md for actual counts and failed attempts.
- The same-mount standing repeat also passed (234 fields, .222086 m minimum
  independent clearance, .002281 m stopping drift); its stricter mass-source
  audit passed. Both bounded examples retain potential/thermal priors, transient
  velocity errors and rejected stale/failed optimizer results. Their arrivals
  are not replicated baseline results, full fresh physical estimation or the
  PR #117/main acceptance gate.
- Initial native carried-bowl work was a separate zero-wheel draft component. The
  official hand attachment/actual PhysX prop motion passed, and one parked
  carrying-human perception benchmark passed (three stationary/six moving
  adult-prior rows). Separate bowl ownership is not validated. Four visual
  trials and the sensor trial failed strict actor stopping; SDK task success
  and zero reported velocity did not mean a stationary root/collider. Keep
  failed reports and the unchanged .03 m stop gate; do not freeze/teleport or
  relabel attachment success as navigation acceptance. The sensor component's
  .416667 s maximum completion age is not the .35 s live-control gate. Preserve
  those failed reports. The user subsequently selected
  continuous server crossing; its separate acceptance must verify actual
  crossing, sustained motion, articulation and independent body/bowl geometry,
  without claiming human endpoint stopping. Robot waypoint arrival and safe
  stopping remain mandatory. Only the later explicitly validated single-crossing
  upper-continuation case is accepted below; repeated lower-return cases remain
  failed and must not be reported as successful navigation.
  See docs/restaurant_native_carry.md; main remains unchanged.
- The installed Isaac python.bat may return zero after script RuntimeError.
  Simulation acceptance requires saved validation flags, actual arrival/stop
  and error-log inspection; a launcher exit code alone is insufficient.
- The native-human guard may retain actual simultaneous-observation epochs for
  enrolled geometric track pairs inside the existing .35 s merge window.
  Parent-only updates never renew a missing fragment's original evidence.
  This corrects a continually advancing parent timestamp truncating recovery;
  it does not retain body shape, factor values or inferred free space. Require
  fresh stable current-parent velocity, existing .10/.25 m/three-positive-return
  and ambiguity gates, and save exact positive geometry for retained-pair
  events. Preserve the .12/.18 m carried-server failed arrivals and their safe
  track-loss stops alongside the later separately configured accepted case.
- The single-crossing carried server with upper-aisle continuation passed the
  original 3.9 m robot route: three waypoints, .999485 m minimum independent
  clearance, .494319 minimum CBF margin, .002247 m stopping drift, 225 live fields
  (219 translating), 31 supported moving-human/30 simultaneous kinetic rows.
  Use its explicit upper-continuation/.18 m grouping/+15 degree camera/60-frame
  stationary setup, fixed lambda 1.25 and unchanged safety gates; see
  docs/restaurant_native_carry.md. The independent retained-pair audit verified
  one .3 s current-parent velocity query with three actual returns and .244295 m
  centroid distance. This is bounded geometric recovery, not occlusion safety.
  Human mass and potential/thermal remain priors; contact is a visible proxy;
  separate bowl perception/fall/plume evidence remains unvalidated. Median live
  velocity checks passed but world/relative error maxima reached 1.640/1.586 m/s.
  Preserve those transients, failed repeated-return cases, oracle ego references,
  read-time RGB clock approximation, renderer warning and main's existing hold.
- The opt-in general object-height path fits a local LiDAR floor in an explicitly
  declared frame/reference and reports visible elevation and adjacent support
  separately from strict local drop. Ground-supported potential baseline priors
  and uncertainty priors may both be .25 but must retain their distinct basis;
  no missing landing, hidden bottom, center/body/hand height or unowned cluster
  becomes fall height. Reject competing support levels; retain original source
  times, all occupancy and legacy sphere/bottom/continuity gates separately.
  Ground/raised zero-wheel component gates passed with an explicitly wider .80 m
  neighborhood and existing .20 m gap, not new runtime defaults. The ground
  strict-drop result remains unavailable; raised drop was .300817 m. These are
  configured target-selection and oracle-frame-assisted sensor diagnostics,
  not moving-human support, carried-bowl ownership or new wheel acceptance.
  Preserve sparse-support/startup failures and unchanged calibration/main hold.
  See docs/object_height_estimation.md for its bounded component gate.
- The opt-in carried-object component separates current positive human/object
  RGB/SAM/LiDAR ownership before independent mass, motion and factor assessment.
  Person mass remains the adult prior; the prop uses its own rough-mass network.
  Confirmed groups apply maximum member phi over both members' occupancy only;
  never mix human mass with prop height or spread the maximum to other obstacles.
  RGB COCO wrists are association cues, not measured joints or grasp evidence.
  Short identity continuity does not publish missing member factors or motion.
  The bounded rotation-invariant relation requires both members to translate,
  stable separation and limited net relative centroid motion; preserve its source
  gaps/ambiguity gates. Wrist-depth motion was rejected after observed jitter.
  The two-wrist compact mode must never silently fall back to the broader
  one-wrist scene-component prototype; its observed background-mask error is
  preserved. Only the explicit two-wrist mode has the later component gate.
  Keep failed side-view trials and their independently wrong/missing partitions.
  Separate bowl fall/plume evidence and new grouped wheel acceptance remain
  unvalidated until their own gates pass. Main's existing hold remains unchanged.
  See docs/carried_object_groups.md for the separate parked component results.
- The diagonal zero-wheel carried-group component passed with 167 fields, 26
  separate assessments and three confirmed moving groups (7.6/7.8/7.9 s), .158226 m
  maximum bowl-centroid error and .183333 s group completion age. Its explicit
  `(26.5,7.4)`/atan2(1.5,2.8) parked view and crossing-facing actor initialization
  are authored scenario choices, not estimator inputs. This brief interval is
  not continuous carrying coverage or grouped robot navigation. Preserve the
  side-view failures and blocked direct-frontal startup. Human contact silhouette
  proxy saturation made all three group phis 1.0; potential/plume stayed .25
  uncertainty priors. Keep that factor-specific finding, the adult mass prior,
  rough bowl mass, oracle frame references and main hold explicit; do not tune
  confidence/scales/priors to make the group less risky.
- The user's later decision authorizes a separate explicit human-body contact
  model: default .10 nonzero blunt-body PRIOR after fresh unambiguous person/SAM
  association. This supersedes the generic outline policy only in that opt-in
  scope, not physical sharpness/hardness, accepted part cues or carried objects.
  Preserve raw outline scores/old phi=1 results and their original source times;
  never raise measurement confidence to select the prior. Retained human mass,
  class text or an authored role alone cannot select it. Keep other factors,
  unmatched occupancy, group maximum, primary lambda and all safety checks
  unchanged. A separately accepted edge uses its original uncertainty policy
  and wins over the body prior when stronger. This new declared model is not
  physical contact calibration or new wheel acceptance; retain the legacy mode
  by omitting the explicit config/diagnostic flag. See docs/human_body_contact.md.
- The .10 body-prior zero-wheel component passed with 167 fields, 108 fresh
  human-prior rows (92 moving), no nonhuman assignment and preserved raw scores
  (15 were saturated). Three confirmed groups used bowl phis .822696/.821154/
  .822236 over person phis about .609. This is prior-assisted same-frame policy
  dataflow, not new grouped navigation, continuous association coverage or
  calibrated body contact. Preserve the remaining view/height/plume limitations.
  Keep the chosen human contact model fixed in matched controller experiments;
  show contact-prior sensitivity and risk-dependent lambda separately.
- The bounded restaurant comparison opt-in preserves raw CRA assessments and
  records the separately applied field policy. Fixed-class uses only the fresh
  person association or unknown fallback; an accepted SAM mask never invents a
  low-risk generic class. The circular velocity-aware geometric CBF is a
  different controller reference with conservative footprint/cell allowances,
  source-age rejection, inferred tracker motion and labeled zero-motion priors;
  its shadow risk field is never its wheel controller. Compare fixed-class and
  CRA under identical MPC/CBF settings and lambda 1.25. Geometric CBF margin units
  differ from h_full and its shadow perception overhead is not pure CBF timing.
  The prespecified crossing variants use elapsed navigation time, independent
  of robot progress; preserve the legacy robot-y trigger separately. Keep all
  failed/missing reports and signed inflated-monitor overlap, distinct from
  measured physical collision. See docs/restaurant_baseline_comparison.md.
