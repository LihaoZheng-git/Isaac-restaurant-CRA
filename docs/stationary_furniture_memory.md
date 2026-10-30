# Bounded stationary furniture memory

The user approved continuing through brief furniture tracking gaps using
remembered positive geometry and freshly verified commands. The previous
long-human run stopped before its encounter: eight returns in fragment2045
were absent from the next scan's original10cm correspondence region. That
failure remains valid; this experiment adds a different, explicit geometry prior.

`--stationary-furniture-memory` is default OFF. The separate recipe is
`configs/restaurant_furniture_memory_v1.json`; global static occupancy memory
remains OFF. No authored layout, asset name or actor truth selects furniture.

## Contract

- Every original point must match observations from at least two earlier scans,
  spanning at least .2s, within .06m in the world frame. Source gaps above .35s
  reset qualification. Matching is geometric and may cross track-ID changes;
  it never transfers an object's visual identity or factor evidence.
- An unpublished candidate may also complete that same three-scan proof after
  its original track ID disappears, using real subsequent scans. Its maximum
  confirmation window is the existing two-gap history window (.7s total, each
  actual source gap at most .35s). Every point must have current real support.
  The original geometry/policy epoch and10s expiry stay unchanged; qualification
  time is separate. Pending candidates never supply occupancy or guard coverage.
- Qualified surfaces with their track center within2.5m are eligible for a10s original-source
  lease. A missing surface is rasterized with .05m uncertainty plus .005m per
  second of age. These configurable engineering priors assume stationary
  background furniture; repeated support alone does not prove permanent rigidity.
- Resolved, untruncated current person masks, even without accepted instance association, and validated
  body centers exclude a1m neighborhood. Tracks with at least three observations
  and speed above .2m/s without complete repeated fixed-world surface support
  exclude their own .06m surface neighborhood. A centroid-speed disagreement
  with EVERY-point static support is recorded, not applied to CRA/kinetic values.
  This narrower
  generic-motion region avoids confusing changing furniture-cluster centroids
  with a one-metre moving body. Both exclusions purge old records AND support
  history. Unrecognized people and crowded motion remain outside acceptance.
- Retained geometry uses the strongest originally applied policy among all
  owned/support returns in the original track. Its source frame, time and risk
  remain recorded. This is retained control policy, not newly measured mass,
  shape, heat, motion or fall height. Current tracks still get their normal CRA.
- The guard checks the original ID, source time and centroid, then verifies
  every uncertainty-expanded surface cell against the actual current occupancy
  grid, including risk and boundary flux. Clipping, expiry or missing/lower-risk
  coverage do not permit continuation. No guard clock is renewed by memory.
- Memory is staged with the field and discarded with a rejected proposal.
  Existing field-age, exact MPC/DTCBF/dense geometry and per-step analytical CBF
  gates remain. The nominal route, speed/margins and lambda1.25 are unchanged.

The prior fills missing observed surfaces, not entire inferred solid furniture
volumes. It does not infer free space or provide a general moving-object map.
Known human identity-loss handling remains unchanged. Geometry is planar and
world alignment retains the project's declared simulator pose provenance.

## Validation

Focused tests cover source-preserving retention/expiry, changed IDs, unsupported
surfaces, human/motion invalidation, complete current-grid coverage, risk/flux
floors, rotated/rolling coordinates, staged rollback and an actual all-object
field with a missing track. Independent native-human root audits are evaluation
only; retained raw surfaces must stay outside its .4m reference disk.

The archived pre-encounter replay uses431 real sources from the failed v3 run.
It has gaps up to2.8s and therefore is not a full scan-contiguous replay. Version1
over-excluded neighboring furniture using noisy moving-cluster centroids.
Version2 separates person-region from generic surface exclusion and retains
the actual fragment2045 at its original90.7s source. No old field/result is
rewritten. This is a module replay, not a wheel or general occlusion claim.

At the initial source freeze,113 focused tests and3,297 full unit tests pass
(111.45s full suite), with Ruff and compilation checks passing. Native validation
and the full long mission are pending at this source freeze.
All previous accepted route-only and failed long-human records remain available.
Primary/main/PR117 holds remain; no real-time or complete fresh four-factor claim.

## First native attempt and visibility correction

At64037a9 all four fresh prerequisites passed: the parked native walking-human
component (116sources,62with retention),180s zero-drift idle/release,60-frame
route-start and60-frame near-human views. Their sources remained unchanged.
The first long mission nevertheless FAILED nearby_track_lost10.833333s into
navigation,75fields, before the person encounter. Clearance.579947m and
stop drift.005066m do not establish arrival. Preserve `long-motion-v1`.

Fragment17's visible returns changed from1174 to614, shifting its centroid and
creating an inferred speed of about.78m/s. Its earlier source had ordinary low
speed. The first qualifier treated this as physical motion and excluded memory.
The corrected qualifier checks EVERY current raw world point against the same
three-source/.2s/.06m support contract before interpreting centroid motion.
It also applies the2.5m enrollment limit to the track center, retaining genuine
outer-point history rather than creating an artificial evidence boundary inside
a large furniture cluster. Newly moving/unmatched surfaces and people still
reject; current kinetic evidence remains untouched. Neither the correspondence
threshold nor the guard's whole-current-field coverage requirement is relaxed.
New focused regressions exercise partial-visibility centroid shifts, one changed
surface point, actual translated surfaces and large-cluster enrollment.
Fresh native gates and a new full mission remain required after this correction.
The correction passes116 focused and3,300 full unit tests (167.45s), Ruff and
the same431-source replay, which still retains original fragment2045. The
archived long-gap replay cannot validate the new motion attempt by itself.

## Second native attempt and person-mask boundary correction

At `f77d522` all four fresh prerequisites passed again:115 parked component
sources (49 with retention),180s zero-drift idle/release, and both60-frame views.
The mission passed the previous stopping location but FAILED at88.3s with
613 committed fields, two passed route guides, .391863m minimum clearance and
.004587m stopping drift. The guard used1316 furniture-coverage proofs; all
independent human-memory audits passed. These are partial results, not arrival.
Preserve `outputs/furniture_memory/long-motion-v2` and its failed report.

Fragment1930 was correctly qualified and retained from its original90.7s source.
At96.2s, YOLO produced two false person masks on the robot's visible image edges
(confidence .573 and .540). Projecting these masks excluded1597 returns and
purged that furniture record. The actual human was distant in the image. This
was a raw visual-cue quality bug, not missing memory eligibility or a CRA score.

The exclusion adapter now rejects masks or continuous detector boxes that reach
the image boundary, consistent with the existing resolved person-association
requirement. Checking the detector box catches resizing slivers that leave a
single empty mask pixel at the border. Rejected detections, confidences, boxes
and affected return counts are logged as unresolved cues. They never assert
that a person is absent. Current validated body centers and IDs still exclude
unconditionally, and resolved raw person masks still exclude without SAM.
No physical occupancy is removed; no mass, motion, other factor or source clock
is changed. New partially visible/unrecognized people are outside this bounded
scenario, and this is not a general detector false-positive solution.

The archived actual-image/LiDAR replay reproduces both masks and the1597-return
exclusion, and the corrected adapter rejects both. That is an offline component
diagnostic only. Fresh stationary and moving-human gates, followed by a new
full navigation attempt, remain required.
The boundary correction passes125 focused and3,305 full unit tests (116.92s),
Ruff, compilation and diff checks. These do not replace the pending native gates.

## Third native attempt and confirmation across an ID change

At `76e18b8` all four new prerequisites passed, including116 parked component
sources with89 retained-surface observations. The full mission passed both
previous failure locations, then FAILED nearby_track_lost at110.133333s,
758 fields and four passed route guides, before human walking began. Minimum
clearance was .391843m and stopping drift .004592m;1542 furniture coverage proofs
and all independent memory-human audits passed. Preserve `long-motion-v3`.

Fragment2429's last observation at112.2s contained13 points. Ten had already
completed temporal support; three had not. Its entire surface was still directly
observed inside the larger cluster1959 at112.4 and112.7s: maximum original-to-current
distances .057262m and .033380m. The qualifier only examined the old fragment
while its ID existed, so it missed this later confirmation.

The corrected implementation keeps the original candidate unpublished within
the existing three-scan history window. Real subsequent scans can complete the
same all-point stationarity proof across cluster IDs. This does not transfer
identity/factors, add geometry before qualification, renew the original lease,
or change correspondence, uncertainty, human exclusion or safety thresholds.
Qualification time is logged separately from the original source time. Spatial
indices are reused within an update to avoid rebuilding the same scan index for
each pending fragment. The original failed result remains a failure; new native
acceptance is still required.

The corrected source passes132 focused and3,312 full unit tests (142.67s),
Ruff, compilation and diff checks. The506-source archive replay confirms
fragment2429 at112.7s while retaining its original112.2s source. That archive
has gaps up to4.5s: its later117.4s history reset purges the fragment, whereas
the original live diagnostic records show the covering cluster continuously
qualified at116.7--118.0s. Keep both facts; the replay is confirmation of this
data path, not a complete native navigation pass.

Review found a further sequence edge case: clearing support history alone could
leave an unpublished candidate eligible for three later scans across a source
gap. A gap now also discards those candidates, while previously qualified
records retain their original expiry. A regression supplies three fresh scans
after a broken sequence and checks that only the new current surface qualifies.
The v4 parked component at `4b8801e` was interrupted during this review, before
wheel trials. Its partial output is preserved and provides no acceptance.
This source-gap correction passes133 focused and3,313 full unit tests (116.98s),
with Ruff and diff checks passing. Fresh native validation remains pending.
