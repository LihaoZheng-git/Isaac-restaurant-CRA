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
- Qualified positive surfaces within2.5m are eligible for a10s original-source
  lease. A missing surface is rasterized with .05m uncertainty plus .005m per
  second of age. These configurable engineering priors assume stationary
  background furniture; repeated support alone does not prove permanent rigidity.
- Current person masks, even without accepted instance association, and validated
  body centers exclude a1m neighborhood. Tracks with at least three observations
  and speed above .2m/s exclude their own .06m surface neighborhood. This narrower
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
