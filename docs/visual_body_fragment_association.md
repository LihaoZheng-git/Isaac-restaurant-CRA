# Current visual association of small body fragments

The profiling-off `df93d52` trial in
`outputs/paper_flux/ray-human17-observed-no-profile-v1` failed the continuity
guard at 19.65 navigation seconds, before its declared 20 s crossing. It reached
one of three waypoints, with .178136 m independent clearance including stopping,
.730048 minimum CBF margin and .000711 m stop drift. There were 195 published
fields, 185 while translating. Optional call profiling was off. This is neither
arrival nor bypass acceptance; the deliberate gate-failure traceback is retained.

Its exact first-loss archive identifies fragment 369: four measured lower-leg
returns at 17.6 s became two unclustered returns at 17.7 s. These positive returns
remained occupancy, but the four-point cluster ID disappeared and later blocked
the robot at its old location. No empty-space observation follows. The old
centroid/source clock was correctly retained by the strict guard.

Replaying the original RGB and calibration with the audited existing models
produced an accepted SAM mask (quality .901285) and person/SAM IoU .866958.
All four original fragment points project into both masks. Sparse depth-visible
projection retains only one of them; the complete calibrated projection is
therefore needed to diagnose the whole fragment. Other background returns behind
the person remain separate, including the large 343 cluster. The original image
annotation is `outputs/paper_flux/sparse-fragment-v1.png`.

## Opt-in method

`--human-visual-fragment-association` is limited to the separated, bare native
adult, observed-occupancy ablation. A same-frame **fresh RGB person mask** and
**current positive 3D returns** may associate a still-observed small cluster with
the currently validated person parent. Parent identity can be the existing
explicit identity prior; its original camera epoch is preserved. The method
uses neither the circle's radius nor actor truth. It adds no control geometry.

Requirements:

- Current verified parent LiDAR support; at least eight projected parent returns,
  >=90% inside one untruncated, confident person mask (confidence >=.5).
- Every child return projects inside that same mask, no competing person mask;
  at least three distinct positive returns and <=10% of the parent's return count.
  The first version also had an absolute 32-return cap; the native crossing
  exposed that cap's dependence on sampling density (see result below).
- Every child return is within .30 m in **3D** of actual current parent returns.
  This is a bounded association threshold, not body radius or clearance.
- Same camera frame/epoch and <=.15 s camera-to-scan age; unique nonnested parent.

The continuity guard records the inferred current part association, watches the
parent, and does not independently enroll that child. It cannot retrospectively
clear a missing child using an old mask. Parent loss still blocks. No clustering,
occupancy, factor score, human class/mass acceptance, passing-side selection or
clock is changed. Unclustered/foreign returns retain their original policies.
Legacy callers leave the option disabled. This is an explicitly bounded
association hypothesis, not proof of physical attachment, a crowd tracker,
carried-object solution or free-space clearing.

## Validation

Unit cases cover complete association, whole-fragment projection, one outside
point, one unprojected point, excessive 3D gap, duplicate returns, competing or
truncated masks, weak/missing/stale parent, frame mismatch, original identity
epochs, unchanged arrays, parent enrollment, missing-parent stop and rejection
of old proofs. Exact sensor replay (`visual-parts-replay-v1.json`) associates
369 and several other small leg fragments while rejecting the large chair/wall
losses. It preserves original capture timestamps via `captures.json`; no authored
human geometry enters the replay.

The initial complete unit run passed 2,324 tests; 104 focused checks passed on
the final initial source, including its added enrollment/metadata checks. Ruff
passed. A sandbox temporary-directory setup failure was resolved by using a new
workspace-local test directory; no dependency/environment upgrade was needed.

## Native result at 60bb923: initial cap failure preserved

`outputs/paper_flux/visual-parts-human17-observed-motion-v1` passed 60 parked
frames, then stopped on `nearby_track_lost` at 21.1 navigation seconds. It passed
the earlier failure time and began the crossing, but did not complete the route:
1/3 waypoints, 209 fields/192 translating, .117140 m independent clearance,
.705905 minimum CBF margin, .000931 m stop drift. Human motion had two supported
kinetic/one simultaneous row, below the full gate. Behind projection -.251958 m
passed alone; walking arc -.012572 m and arrival failed. This is not bypass or
fresh four-factor acceptance.

The new blocker was 620, source 28.9 s, missing at 29.1 s. Its original 69 returns
all project into one confident (.863202) fresh person mask; all 69 also pass the
existing .5 m foreground-depth layer. Parent 71 has 1,006 returns, .999006 mask
coverage, and the maximum actual child-to-parent 3D distance is .296001 m.
Thus the child is only 6.86% of parent support and passes every association gate
except the original absolute 32-return cap. Its original returns still have
current positive neighbors within .070512 m, owned as unclustered occupancy;
this is not observed free space or a measured disappearance of the body.

## Density-independent follow-up

Remove only the absolute 32-return cap, keeping >=3 distinct returns, <=10%
relative size, fresh whole-mask membership, unique current validated parent,
unchanged .30 m 3D point gap and all clock/ambiguity/occupancy rules. An absolute
sample count should not change identity when the same physical surface is
sampled more densely. A component exceeding 10% remains rejected. Controlled
tests compare the same surfaces at 40/3 and 800/60 samples and the measured
1,006/69 count case. No mask dilation, wider physical uncertainty, absent-hazard
assumption or retrospective clearing is introduced. The exact failed original
record remains separate; a new own stationary/native trial is required.

The follow-up passed all 2,327 unit tests in 81.81 s, 65 focused continuity/
association tests in 4.79 s and repository Ruff. An exact original-frame replay
now accepts 620 -> 71 at the original 28.9 s LiDAR / 28.983333 s RGB clocks,
preserving the parent's original 28.883333 s identity cue. This is a component
replay result, not natural crossing acceptance. The initial native trial had
16 accepted part-association rows across six child IDs, 78 all-track visual
instances, 14,347 object assessments and zero complete four-factor assessments.

No bypass or four-factor fresh-perception pass follows from component results.
Keep fixed lambda 1.25, .25 factor priors, forward-only motion, original
clearance/braking/freshness/exact checks, all failed attempts and main holds.
