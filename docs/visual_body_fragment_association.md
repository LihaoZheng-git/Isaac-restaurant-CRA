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
  3--32 distinct positive returns and <=10% of the parent's return count.
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

Native stationary/motion validation of this change is pending. No bypass or
four-factor fresh-perception pass follows from these component results. Keep
fixed lambda 1.25, .25 missing-factor priors, forward-only motion, original
clearance/braking/freshness/exact checks, all previous failures and main holds.
