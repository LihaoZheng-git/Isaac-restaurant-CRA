# Crate-only confirmed evidence refresh

The user requested a crate-only test in which better evidence can update saved
mass/contact, while uncertain views leave the saved values intact. This separate
default-off experiment uses `--confirmed-prop-refresh` and
`configs/restaurant_crate_evidence_refresh.json`. It does not change the V3 demo,
bin, human, controller, priors, route, lambda, robot limits or acceptance gates.

The archived full-demo failure has two separate paths: ordinary raw-track RGB
assessment succeeded, while repeated image-group acquisition failed. A 34-point
reference at source37.0 was followed by a 182-point candidate at45.0, all original
samples supported but new surface up to0.2653m away. The old6cm bidirectional
support and0.5m view-change scope rejected it. Original failures stay preserved.

The experiment permits a new repeated proof of a genuinely expanded observed
surface. It requires one unambiguous original reference, every historical anchor
within the original6cm distance of a current point, and every reprojected anchor
inside the current mask. SAM quality must be available and no lower than the
reference. More sampling density alone is insufficient: at least8 new3cm world
voxels beyond the old6cm support, at least25% relative voxel gain, and at least6cm
span gain are required. The original90cm object extent/ownership/foreign-point/
competing-mask gates remain. These are explicit engineering assumptions for
isolated static props, not proof of complete physical identity or calibrated
sharpness. They do not resolve unobserved adjacent objects in general.

The new view must agree across at least3 successive sources spanning0.2s, with
the original0.35s gap and existing calibrated moving-view checks against the
first proposed replacement. Missing/ambiguous/failed proposals reset the pending
proof. Merely seeing a larger point set never replaces saved evidence.

After confirmation, the actual mass/contact estimators run on the fresh accepted
mask. Prior/retained values cannot masquerade as new observations. Replacing a
saved record additionally requires current timestamps, no lower estimator
confidence, no wider evidence intervals, complete previous support and no
competing stored location. Scores can rise or fall; neither score nor object
name selects the update. Both properties and association reference commit only
with the accepted field. A failed assessment or field leaves old state intact.
The refresh decision itself adds no stop command or controller veto: failure
simply retains the existing evidence path and ordinary occupancy policy. Existing
geometry, freshness and control checks can still stop when their conditions fail.
Every old anchor remains for subsequent growth checks, and old/new property
sources are recorded. Reuse alone never renews a timestamp. Kinetic and
interaction stay live; unchanged contact cue does not imply unchanged phi.
The existing initial stationary acquisition phase still takes its latest valid
source; the new guarded replacement applies after properties are frozen for
live use. This experiment does not redefine that initial acquisition phase.

Run the fresh independent60 stationary gate, then own60 and the original40cm
nominal approach with original10cm arrival tolerance. Native outputs belong under
`outputs/crate-evidence-refresh/`; old V3 files must remain byte-identical.
Synthetic positive/adversarial tests do not establish
that natural crate views will supply sufficiently better evidence. No full-demo
repair, general tracking, complete fresh four-factor or safety guarantee follows.

Keep default-off/draft/main/primary/PR117 holds. No merge or full-demo rerun.

Source validation:105 focused tests and3,899 full unit tests (222.38s), Ruff and
diff checks pass. The original long-run archived candidate replay reproduces all
422 source decisions exactly with the option off. A geometry-only development
probe finds one expanded-reference confirmation at41.9s and10 crate-qualified
rows versus zero before. Old archives lack proposal quality, so this probe
explicitly assumes equal SAM quality and omits fresh property assessment; it is
not a model-quality, retention or native acceptance result.

## Native crate-only result at fc49f31

The independent60 and own60 stationary gates PASS with frozen source hashes and
zero wheel commands. The independent gate has58 accepted target assessments of
60 sources. Both initial sensor-epoch rejections remain archived.

The short approach PASS reaches its one goal in1.816667 simulation seconds,
with0.304336m measured forward travel toward the40cm authored goal under the
original10cm arrival tolerance. Minimum independent clearance is0.529563m,
minimum CBF margin0.222635, and stopping drift0.003184m. There are12 live fields,
11 committed while translating. These results do not establish long-route repair.

Original-source/current-point retention passes12/12 live rows, with zero errors.
The saved rough mass stays2.157806kg and shape cue stays0.395161. All selected
points remain within the independent crate mesh bounds; all12 unchanged20cm
whole-center diagnostics pass. Five crate view-growth attempts are rejected:
all lack complete historical support and meaningful new coverage, two also have
lower segmentation quality, and two omit projected historical anchors. Stored
values and source timestamps remain unchanged throughout motion.

There are ZERO native geometric confirmations or property replacements. Thus
native rejection/holding/navigation are validated in this short view sequence;
the positive replacement path is validated only by controlled unit/integration
tests. A natural larger-view replacement still needs a separate observation.
Do not promote this result into a claim that full-demo crate acquisition is fixed.
Exact archived proposal-mask/point/SAM-quality replay passes all60 stationary and
72 own-stationary/live sources, including every refresh decision.

Kinetic and interaction remain current. Two new-fragment updates lack speed
history and use explicit fallbacks, raising final contact risk to0.495161 and
phi to0.787021 despite the unchanged shape cue. The full phi range is
0.617117-0.787021. This existing limitation remains; no claim of constant phi,
calibrated static-object velocity or generic four-factor sensing is made.

Reports, rejection reasons, source audits, native video and the plot are under
`outputs/crate-evidence-refresh/`. The chart is `crate-evidence-result.png`;
`motion-v1/service-risk-camera.mp4` contains the native camera presentation.
All408 previous final-demo files remain byte-identical. Draft PR237 stays
unmerged/default-off; V3 is unchanged.
