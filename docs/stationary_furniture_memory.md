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
- Qualified surfaces with their track center within2.5m are eligible for a10s original-source
  lease. A missing surface is rasterized with .05m uncertainty plus .005m per
  second of age. These configurable engineering priors assume stationary
  background furniture; repeated support alone does not prove permanent rigidity.
- Current person masks, even without accepted instance association, and validated
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
