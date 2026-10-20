# Bounded missing-fragment coverage prior

The user explicitly approved this separate geometry uncertainty policy after
the 13 s forecast diagnostic stopped on fragment 490. That failed run is
preserved in [the forecast report](forecast_guidance_reference.md). A centroid
inside a human disk never suffices. The existing strict three-positive-return
merge path and all legacy defaults remain unchanged.

## Opt-in policy

`--human-fragment-coverage-prior` requires the standalone validated human-body
envelope and continuous CRA mode. It does not enable static occupancy memory.
`--human-fragment-maximum-point-speed` defaults to **1 m/s**, a configurable,
uncalibrated point-motion PRIOR for this separated adult fixture. Centroid
measurements check consistency with this assumption; they cannot establish a
bound on articulated point speed. Faster limbs or unmodeled motion remain a
limitation, not measured coverage.

The adapter stores every track's complete last positive planar LiDAR bounding
region, point count, risk and source time. A fragment needs at least four
observations, two recent velocities within the configured speed bound and a
unique simultaneously observed body envelope containing its whole region.
When the fragment disappears, expand the whole original region by a disk of
radius `point_speed_bound * original_source_age`. Require this entire rounded
region inside the same current validated body disk, without a competing body
envelope. Parent observations must remain <=.15 s apart and parent velocity
change <=.3 m/s. The original missing-fragment age may never exceed .35 s.
The established human adapter still supplies its own identity, positive-return,
centroid, source-gap and competitor checks.

Before field construction, raise the applied envelope risk, if necessary, to
cover the old fragment risk using the unchanged .5..1.7 flux map and lambda
1.25. This is an explicitly audited **control-risk floor PRIOR**, not a change
to any physical factor, the object's CRA assessment, or perception readiness.
The independent navigation continuity guard repeats the geometric/motion
calculation from its own positive history and checks all conservatively
intersecting cells of the actual source occupancy grid. Unoccupied cells,
lower risks or clipping outside the bounded field reject coverage.

Successful coverage leaves the missing fragment unresolved in the guard with
its original epoch. It is not an observed merge or a transfer of person
identity, mass, contact, support, fall height, plume or free space. Repeat the
check every field; expiry or ambiguity still stops. Only real positive returns
may update the fragment source record. Worker memory commits transactionally
after a complete field succeeds. Field/command freshness, exact MPC/DTCBF,
dense geometry, analytical CBF and stopping requirements remain unchanged.

## Catalog mass domain correction

A fresh accepted small SAM fragment may overlap most of a person mask while
having insufficient whole-mask IoU for the adult mass prior. A separate
same-frame positive person-containment guard (default .5 of instance pixels)
now blocks catalog-object mass inference. Even truncated/competing positive
person masks may reject that domain, but they cannot assign adult mass,
person identity or body contact. Missing/stale person pixels cannot prove a
nonhuman class. The original unambiguous whole-person association and adult
mass expiry stay intact. This correction does not retroactively validate the
.0247 kg fragment result in the old diagnostic.

## Validation

Unit checks cover full-region versus centroid containment, motion/identity
epochs, competing bodies, warmup, original expiry without renewal, lower-risk
and clipped/holey fields, rotated/degenerate regions, unchanged strict merge
tests, catalog-domain rejection without adult mass, and field transaction
rollback. Native stationary and motion results will be recorded after running
the committed implementation. No proactive crossing or real-time acceptance
is claimed from these unit checks. Main and long-service remain held.

All **2,251 unit tests** passed (76.96 s), including 125 focused policy,
perception and guard tests. Repository Ruff and whitespace checks passed.

### Native diagnostic at 0ed0f3c

`outputs/guidance_progress/native-fragment-coverage-13s-v1` preserves the
same 13 s encounter, 20 s diagnostic limit, .025 pacing and .20 m centroid
uncertainty. All 60 stationary frames passed the component/envelope gates with
zero wheel commands. Four stationary field updates used region-coverage priors.

The moving run ended at the intentional 20 s duration limit: **not passed**,
1/3 waypoints, 199 live fields (137 while translating), .170846 m full-sweep
clearance, .0000027 m stop drift and minimum CBF margin .377741. It had no
track-loss stop, seven strict observed fragment merges and ten worker field
coverage rows. The independent guard accepted one complete ten-cell coverage
proof for fragment 499 at source 22.9, original source 22.8. Its age was only
.1 s: **no guard continuation beyond the strict .15 s loss threshold occurred**.
Do not attribute the absence of a loss stop causally to this prior or describe
the old fragment-490 failure as repaired. Main/long-service remain held.

There were 58 fresh visual instances across 13,879 object assessments, zero
complete four-factor assessments, zero supported moving-kinetic rows and zero
simultaneous motion rows. The passing-selection and curved-arc gates failed.
Fresh human SAM proposals during the crossing were cropped at image boundaries;
high model quality and good LiDAR coverage did not override that rejection.
The new positive-overlap domain guard was exercised on the known body, but this
run did not establish a natural unknown-fragment mass-domain correction.
The specific small-fragment correction remains unit-tested.

One of the 199 independent moving body-envelope checks failed: at source 22.5,
center error .2050068 m exceeded the .20 m uncertainty prior by .0050068 m.
Preserve this failed .60 m-radius record. The user explicitly approved a
separate **.25 m centroid uncertainty PRIOR / .65 m total radius** for the next
test, with stationary validation before additional motion. This is a declared
geometry policy change, not a tolerance change or retroactive pass. The
configured 1 m/s point-speed bound remains uncalibrated. The 850.3 s navigation
wall time, startup epoch rejection and renderer-history warning remain explicit;
launcher exit zero again accompanied a full-acceptance exception.
