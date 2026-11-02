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

Native results remain pending.
