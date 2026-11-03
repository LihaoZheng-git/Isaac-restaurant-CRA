# Unknown obstacles instead of track-loss termination

The user approved representing uncertain tracks as ordinary planning obstacles,
with a fixed unknown phi, before retrying service demo v2. This supersedes the
previous request to decide only a narrow furniture correspondence exception.
The separate `--unknown-obstacle-planning` mode is default OFF. The matched
recipe is `configs/restaurant_service_demo_v2_unknown_obstacles.json`.

The waypoint nominal, CRA estimators, fixed lambda 1.25, physical margins,
MPC horizon, exact geometry/DTCBF checks, analytical filter and wheel limits
are unchanged. Unknown geometry contributes positive occupancy before solving
the field. A lost ID no longer ends a trial when that geometry is correctly
represented. Invalid sensors, invalid field construction or absence of a
verified moving command can still require stopping. Never claim universal
safety, arrival, or pause-free navigation.

## Geometry and risk contracts

- The fixed unknown phi is configurable (`--unknown-obstacle-phi`), default
  **0.68359375**, numerically equal to aggregation of the existing four 0.25
  factor priors. It is a policy prior, not four newly measured factors. Retain
  a higher previously applied risk/flux. Lambda and passing-side policy are
  preserved. The existing current unassessed-return policy is unchanged.
- Store original positive surface points, centroid, frame, source time and
  applied policy for nearby observed tracks. An unknown obstacle cannot erase
  current raw returns, supply a new class/mass/contact/heat/height observation,
  or renew visual identity and factor clocks.
- Already qualified stationary surfaces use the existing 5cm + 0.005m/s
  position allowance for up to ten seconds of support age. Losing qualification
  or an ID no longer immediately erases the old surface from planning.
- A separate explicit **10cm position/geometry-association prior** permits
  whole-surface correspondence to current geometry. This is not the legacy
  6cm observation test. For long gaps on previously qualified static surfaces,
  current supporting points must themselves have repeated 6cm support across
  three real sources spanning 0.2s. Source gaps above 0.35s clear that history.
  Person masks/body/uncertainty regions exclude stationary support. Original
  query/factor epochs remain; a geometric support timestamp is separate.
- Short-gap whole-surface replacement requires all original points within
  10cm of a unique current owner, at least three distinct current returns and
  no competing whole owner or sufficient unclustered alternative. It supplies
  current geometry continuity, not a transferred visual identity. Already
  validated short human-fragment whole-envelope coverage can likewise replace
  an old geometry query, explicitly labeled as a body prior.
- Each replacement retains the old positive surface in its admission field.
  The guard retires the old query only after checking that field's complete
  local occupied cells, risk/flux and original epoch. The current replacement
  track remains watched. Retained risk is recorded separately from factors.
- Without supported stationary geometry, use an all-direction 1m/s motion
  prior plus the full 1.55s horizon/field-age allowance. After a stationary
  lease expires, expand from that lease's endpoint instead of discarding the
  surface. Increasing uncertainty can constrain motion; it is not free space.
- Rasterization is bounded by the local grid. Clipped portions remain unknown
  outside the field; existing out-of-domain/rolling-edge command checks still
  apply. A clipped region is logged, rather than automatically ending the trial
  solely because its full extent exceeds the local field.
- Memory and geometry replacements commit transactionally with an accepted
  field. Rejected proposals cannot update geometry, factor or guard clocks.

These are declared engineering priors for the existing separated walking-human
and stationary-restaurant setting. Fast arbitrary objects, crowded occlusion,
moving furniture, and permanent disappearance without positive replacement are
not validated. No ray-based free-space clearing is introduced. A long unresolved
gap can eventually fill the local field and leave no verified moving command.

## Validation

Tests cover unknown phi and higher risk, geometry growth, original clocks,
whole-surface replacement, ambiguous/partial/unclustered returns, human
exclusion, actual current-field coverage, local clipping, strict default
behavior, real worker transaction rollback, and real CLI recipe parsing.
Native provenance audits distinguish fixed phi, inferred positive support,
retained stationary prior and generic motion prior. Independent native human
truth scores static exclusion only; it never supplies control inputs.

Fresh actor circuit, moving-human zero-wheel component, route-start and near
stationary gates are required before the full mission. A v2 release also needs
all eight route guides, actual final arrival/stopping, unchanged sources and
camera presentation audits. Preserve every failed attempt and accepted v1.
The existing plume/prop diagnostics remain failures: rendered steam does not
establish accepted thermal evidence. Keep main/primary/PR117 release holds.

The first full regression run passed **3,477 tests in 123.16s**; Ruff passed.
The archived v2 failure replay preserves the exact 626 original points, source
26.1s, current raw scan32.3s and original phi0.68359375. All199 local uncertainty
cells are occupied at the original8.1cm allowance; the guard continues while
all29,650 raw points remain. This is a geometry-only newly built grid replay
with archived qualification, not a complete preceding qualification, field/MPC
replay, or native navigation result. Failed report SHA remains
`d7b3c8c1d76eea94af65effb2dac0f3b5f7f91a2c2c83d0fbd7eb0381d54cfd9`.

## First native attempt, e7a1364

All four fresh prerequisites passed with unchanged sources: actor circuit
670.88s wall; moving-human component324.35s/111 sources; startup60/279.15s;
near60/282.32s. The component exercised28 unknown-region rows including14
geometry replacements; four short-human field proofs passed. Own60 stationary
startup also passed before the wheel trial.

The wheel trial was **interrupted after a diagnosed stall**, not accepted.
Its105 committed fields ran from source8.2 to23.2s; final saved local position
was approximately(.10543,.00018)m. No final arrival or stopping report exists.
The original ID-loss guard did not terminate it. Instead, two unqualified
surfaces received the1m/s generic motion prior and expanded across the robot's
field, preventing progress. At their first losses,8,703/8,706 and6,738/6,740
original points had current support within10cm; maximum gaps were.13203 and
.19625m. The whole-surface/all-point qualification was too brittle to establish
a stationary hypothesis for these partially observed large surfaces. Phi stayed
.68359375. Preserve outputs/unknown-obstacle-planning/long-motion-v1.

The user then approved a separate evidence-based stationary-background prior
for partially observed non-human surfaces. Whole geometry remains occupied;
people/observed movers retain dynamic uncertainty. This follow-up requires new
unit and native validation; it does not turn this interrupted trial into a pass.

## Supported-background follow-up

The explicitly approved, separate `--unknown-supported-background-prior` option
is default OFF and requires unknown-obstacle planning. Its matched recipe is
`configs/restaurant_service_demo_v2_supported_background.json`. The original
all-point furniture qualifier and all earlier recipes remain unchanged.

The new hypothesis requires at least99% of the original points within6cm of
current points that themselves have support in two earlier real scans, with
three sources spanning at least0.2s and individual gaps no larger than0.35s.
At least three distinct physical support coordinates are required. Duplicated
returns do not count as separate supporting positions. Every remaining point
must lie within25cm of repeatedly supported geometry. All original points,
including that unsupported tail, remain occupied with a25cm position allowance,
plus0.005m/s growth during the existing ten-second geometric support lease.

These configurable limits are engineering assumptions for separated restaurant
background surfaces. They do not establish that every retained point was seen,
that a surface is a recognized piece of furniture, or that it cannot move.
The25cm bound was selected after the three-source archived diagnostic found a
23.8627cm maximum gap to repeated support; the initial20cm candidate rejected
that surface. The same25cm allowance is added to occupied geometry. The native
validation will be fresh, rather than treating that diagnostic as an independent
navigation test.

Current person masks, validated body regions and uncertain-human regions exclude
stationary qualification. A current track above0.2m/s whose geometry lacks the
dominant fixed-world support excludes its6cm neighborhood as possible motion.
Centroid motion with dominant fixed-world support can be cluster resampling;
it does not change kinetic evidence. Contradictions remove the stationary
hypothesis and retain dynamic uncertainty. Whole-source gaps clear unpublished
support history. No identity, visual factor or original source timestamp is
renewed. A separate support timestamp records real current geometry evidence;
unobserved geometry expires into dynamic uncertainty after ten seconds.
When a conflict invalidates a previously supported stationary hypothesis,
motion uncertainty starts at its last real geometric support epoch and retains
that epoch's position allowance. This timestamp is logged separately from the
unchanged original query/factor source; it is never advanced by the conflict.

Archived three-source geometry replay qualifies both diagnosed surfaces:
track59 retains8,706 points with8,677 repeatedly supported (99.6669%);
track176 retains6,740 with6,738 supported (99.9703%). Original epochs8.2s and10.0s
remain unchanged. This replay uses actual archived raw geometry/current track
metadata, without seeding qualification or using actor truth. It does not replay
the complete earlier perception, field/MPC, or wheel mission. The first source
for track59 is transformed from its recorded robot pose and obstacle scan.

Tests exercise partial support, oversized tails, insufficient support fractions,
source gaps, human exclusions, moving surfaces, duplicate returns, lease expiry,
original source preservation, actual-grid coverage, transactional rollback and
real native CLI dependencies. Fresh actor, moving-human parked, start/near
stationary and full navigation gates are still required before any v2 release.

Pre-native checks:3,500 unit regressions passed in125.03s. A final independent
risk/flux-floor correction then passed51 focused tests in6.05s; Ruff and diff
checks passed. Higher old boundary flux is retained even when the current risk
is higher. Native validation is pending at this revision.

### Native follow-up at875a564

The exact frozen revision passed3,501 unit tests in143.63s and all four fresh
native prerequisites: actor704.49s wall, moving-human component320.82s/108
sources, route-start60/283.48s and near-human60/286.66s. The component exercised
28 retained-region events:14 current whole-surface replacements, two body-prior
replacements,11 retained supported-background regions and one generic motion
region. All original component and new provenance/human-exclusion audits passed.

The wheel trial passed its own60 stationary frames, progressed approximately
3.93m, then stalled. It was deliberately interrupted after184 fields, source
8.2..41.9s, with final saved local position(3.93107,-.00072)m. No final arrival
or stopping report exists. Preserve outputs/supported-background-prior/long-motion-v1.
Six-point background263 at(10.43973,1.89368)m had original/support time17.4s.
Its ten-second support lease expired; generic1m/s uncertainty then grew to
6.75m at source32.3s despite unchanged phi.68359375 and no observed motion
trigger. This is a geometry-lifetime failure, not an automatic lost-ID stop.

The exact final six-point source was not archived; saved earlier same-ID data
contains15 points at16.9s and cannot substitute for it. Nearby current returns
remain, but no exact whole-six-point replay is claimed. Add exact retained
geometry logging before another trial.

The user then explicitly approved a separate persistent stationary-background
prior: retain every qualified original point with a capped0.30m allowance until
positive motion/person contradiction, keeping original clocks and unknown phi.
People/observed movers remain dynamic. This assumes unobserved restaurant
background stays stationary and does not validate arbitrary hidden movers.

## Persistent qualified-background option

`--unknown-persistent-background-prior` is a separate, default-OFF policy. It
requires the supported-background mode and uses the matched recipe
`configs/restaurant_service_demo_v2_persistent_background.json`. Only a surface
with the existing repeated-support certificate can persist. Its entire original
point set stays occupied with `min(0.30, 0.25 + 0.005 * support_age)` metres of
position allowance. After ten seconds its diagnostic mode explicitly becomes
`persistent_stationary_background_prior`; the support/source/qualification
clocks do not advance. Actual new repeated support can update the separate
support epoch, while the original query and factor epochs remain unchanged.

This is the user's approved static-map assumption: previously qualified,
unobserved restaurant background stays stationary until positive person or
motion contradiction. It is not inferred free space or a universal motion
bound. Current people, validated body regions, uncertain-human regions and
observed movers retain their original exclusions and dynamic handling. A
contradiction removes the stationary certificate. Unqualified surfaces still
use generic motion uncertainty. Legacy recipes still expire after ten seconds.

All raw occupancy, risk/flux floors, fixed lambda and exact control checks stay
unchanged. The cap can prevent an old stationary piece from filling the whole
field merely through elapsed time; it cannot promise progress through genuinely
blocked geometry or invalid sensing.

Every accepted native field now archives exact original geometry for every
retained unknown region, including pieces outside the nearby guard's watch.
Content-addressed NPZ files hold the original points, center, source time/frame
and ID; per-field references include geometry and file SHA-256 hashes. Evolving
radii, support certificates and applied risk remain in the corresponding field
diagnostics. Archives are evaluation output only, never runtime inputs. They
resolve the missing-six-point diagnostic gap for future runs without pretending
to recover the old missing sample.

Focused validation:72 tests passed in6.37s across unknown obstacles, supported
background and persistence. Full regression:3,522 unit tests passed in129.42s;
Ruff, compilation and diff checks passed. Fresh native gates are pending.

### Persistent-option native outcome at9036e8b

All four fresh prerequisites passed: actor660.62s wall; parked moving-human
component323.11s/107 sources; startup60/281.17s; near-human60/284.50s. Exact
geometry archive integrity passed for the component (30 references/20 files)
and both stationary gates. The wheel run's own60 stationary frames passed.

The original expiry failure was cleared: the analogous seven-point fragment254
retained original source/support17.4s at radius0.30m beyond the ten-second lease,
and the robot passed3.93m. However, it later stalled at5.3875m. The run was
interrupted after214 committed fields through source41.3s (the interruption
request recorded213 fields at41.1s); there is no final
arrival/stopping report. Preserve outputs/persistent-background-prior/long-motion-v1.

At34.9s the whole-cluster motion exclusion removed the stationary certificate
from original3474-point surface642 (source33.2s, last support34.6s). Generic
uncertainty then grew from2.10m to5.20m at38.0s. Exact archived geometry now
exists. A source-subset replay using real34.3/34.4/34.6/34.9 raw geometry found
cluster679 (32 returns, centroid speed0.9354m/s) overlapping the old surface
only through22 repeatedly fixed-world points. Its10 unconfirmed points were
at least0.18599m from the old surface, outside the unchanged0.06m exclusion.
The full cluster therefore supplied a false motion contradiction through its
already stationary portion. This is a grouping defect, not a failed phi change.
The subset replay is not a complete native tracker/control replay.

Follow-up within the approved persistent policy will distinguish repeated
stationary overlap from unconfirmed moving portions for ALREADY-qualified
background only. New enrollment, human exclusions, all current raw occupancy,
unknown-motion handling and every physical/control check stay unchanged.
Require fresh gates and full arrival after this correction; no v2 release yet.

### Correcting whole-cluster motion contradictions

Within the separate persistent-background option, moving-cluster exclusions
now distinguish the current points with actual repeated fixed-world support
from the unconfirmed remainder. The same three-real-source/0.2s/0.06m support
and source-gap rules apply. Only an already-qualified retained background
surface can use this distinction. New enrollment and legacy modes retain
whole-cluster exclusions. People, body regions and uncertain-human regions
still exclude unconditionally. A source gap removes the point-level proof.

If any unconfirmed point of any current moving cluster lies within the original
0.06m exclusion distance, the whole old prior still becomes dynamic. If overlap
is confined to repeatedly stationary points, a centroid shift of the larger
cluster does not supply a positive motion contradiction at that old surface.
All current points remain occupied, current velocity and kinetic evidence are
unchanged, and every retained old point and original clock remains. The change
implements the approved persistent static-map policy rather than qualifying
unseen geometry as measured or declaring the unconfirmed part stationary.

Each source logs the old/current IDs and original/current epochs, repeated and
unconfirmed counts, support epochs and nearest unconfirmed distance. The
provenance auditor checks these separately from native human exclusion and
actual field coverage. Fixed-world correspondence is still an engineering
stationary hypothesis; it does not establish a general hidden-mover detector.

Archived component replay at34.9s preserves all3474 original points, source33.2s
and support34.6s at radius0.2515m; whole-cluster exclusion produced2.10m dynamic
uncertainty instead. It uses actual34.3/34.4/34.6/34.9 scans, a subset of the
native history, with the recorded original certificate. No actor truth enters
this replay.85 focused tests pass in5.45s. Full regression/fresh native gates
remain pending for the correction. Full3,535 unit tests passed in122.30s;
Ruff and diff checks passed.
