# Bounded positive-surface continuity

The closer 20 s observed-only crossing stopped on fragment 617 while parent
75 remained current. Both centroid queries lacked three positive returns within
the unchanged .10 m gate. The former guard retained only a centroid, not the
original point set, so the natural fragment's coverage cannot be reconstructed.
See [the preserved failed trial](observed_crossing_encounter.md).

## Additional measured-geometry proof

`--human-positive-fragment-surface` explicitly enables a bounded additional
association route, requiring the existing bare-human positive-extent mode.
Legacy behavior remains unchanged without it. Existing successful centroid/
bounded-velocity proofs remain; no distance or age threshold is widened.

The new route retains the latest positive planar point set of an enrolled
track for association diagnostics only. It does not add those old points to
occupancy. If that track disappears, the new proof requires:

- An actual recent co-observation with the current parent, preserving the
  original .15 s pairing and .35 s association-age limits.
- The original centroid still within .25 m of the current parent centroid.
- At least three distinct original points, with **every** original point within
  .10 m of currently copied positive points owned by that parent.
- At least three distinct current supporting points and no competing supported
  track/unclustered geometry; the existing three-return ambiguity gate remains.
- The existing one-to-one missing-fragment/current-parent association rule.

No old-surface advection, inferred free space, circular body coverage, factor
transfer or semantic relabeling is introduced. Original point/source epochs
remain intact; parent-only updates never refresh a missing fragment. Histories
expire at the unchanged association age, while the unresolved loss clock remains.
Missing/partial/ambiguous evidence still stops. The proof concerns sampled planar
positive geometry, not complete 3D body coverage, articulated motion guarantees
or hidden surfaces. RTX accumulation/ego-only deskew limitations remain.

Accepted proofs retain original/current point coordinates and epochs. Rejected
proofs retain the original coordinates and maximum nearest-parent distance.
Merge/stop archives now include source-frame XYZ and pose metadata; stop logs
also retain the preceding accepted packet under its original epoch. These are
diagnostic exports, not controller inputs or refreshed observations.

## Validation scope

The controlled ring fixture reproduces an empty centroid even though all old
surface samples rejoin one current track. The legacy guard stops; the new
whole-point proof resolves that association with .10/.25 m gates unchanged.
Tests reject a single uncovered point, shifted geometry, competitors, unclustered
returns, absent co-observation, expired geometry and competing missing fragments.
They also check unchanged input arrays, original clocks and transactional rejection
of missing source geometry. These are unit fixtures, not a claim that native
fragment 617 is repaired. Its next guarded trial must supply that evidence and
pass its own 60-frame stationary component before wheels.

All 2,310 unit tests passed in 93.05 s, including 70 focused continuity/coverage
checks; repository Ruff and whitespace checks passed. This validates the
software contracts, not natural native recovery or a completed bypass.

The 20 s native fixture, all control/physical policies, fixed lambda 1.25 and
all independent clearance, freshness, exact MPC/DTCBF/dense geometry, analytical
CBF, arrival, stopping and passing checks remain. Main/long-service are held.

## Native result at f087533: still failed

`surface-human17-observed-motion-v1` passed 60 stationary frames, then stopped
at 25.233333 navigation seconds on `nearby_track_lost`: 1/3 waypoints,
.231092 m minimum independent clearance, .246328 minimum CBF margin and
.004404 m stop drift. It published 249 fields (201 translating), with 91
all-track visual instances and zero complete-factor assessments. Human motion
passed with nine supported kinetic/two simultaneous rows, .156494/.013460 m/s
median world/relative errors and actual crossing. Full arrival/arc/behind failed.

There were three legacy merge events and **zero new surface-proof events**.
Passing the former fragment-stop time does not establish natural fragment-617
recovery. This run instead stopped on old fragment 350, last observed at source
17.7 s near (27.752845,9.017058), missing for 15.6 s when source 33.3 s brought
the robot within 1.190842 m. The nearest current positive return was .712571 m
away. No positive overlap proof could clear it; absence is not observed free
space. Its original surface history had expired, and no early rejection event
was saved because the caller logged only merges/stops. Preserve that diagnostic
gap rather than invent its old 3D shape or declare its location empty.

The run froze source and preserved all guard/exact checks. It establishes neither
a successful bypass nor a native exercise of the new proof. The next step
captures exact source packets at first disappearance and checks the same fixture
with optional call profiling off, leaving wall timers and all control gates on.
