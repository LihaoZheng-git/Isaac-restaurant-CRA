# Bounded person-identity passing-selection ablation

The user approved this optional policy on 2026-10-20. The primary observed-human
selector still requires a fresh accepted same-frame person/SAM association for
a new selection. This separate experiment may use the original person identity
for at most .5 seconds when the body-envelope adapter has **already verified
current positive LiDAR continuity**. The earlier body-envelope trial's qualified
.2-second identity gap motivated the bound; it did not demonstrate bypass success.

Enable `--human-passing-identity-prior-max-age .5` together with observed-human
passing and the explicit body-envelope prior. Zero is the legacy default. The
worker rejects this mode without the envelope adapter; the CLI rejects ages
above .5 seconds. The envelope's longer geometry identity lease never extends
selection eligibility. Its support, identity, competitor, source-gap and motion
residual gates must first pass transactionally. Original image/frame and current
LiDAR epochs are kept separate, including the camera read-time approximation
and oracle ego-pose/extrinsic limitations.

Fresh visual IDs and retained identity eligibility are reported separately.
An identity-prior selection is not marked fresh. Each new selection records
its original image timestamp/frame, selection-camera timestamp, current LiDAR
timestamp and support count. Held preferences keep that original record and the
unchanged eight-second policy expiry; new images cannot renew an existing
selection. The .5-second cap applies to making a new selection, not to relabeling
the resulting controller policy as current visual evidence.

This changes only passing-side eligibility. Nominal waypoints remain obstacle
unaware. Mass, motion, contact, potential/support, plume, factor priors, occupancy,
lambda 1.25, planner/field freshness and exact MPC/DTCBF/analytical CBF acceptance
are unchanged. It supplies no missing physical factor. Primary comparison
aggregation rejects enabled identity-prior variants even when guidance tracking
is off. Native results must pass stationary gates before wheels and retain
arrival, containment, human motion/mass, selection identity, visible arc and
full braking-sweep checks separately. It does not promise to solve the planner's
previous near-zero-progress behavior.

All 2,183 unit tests and repository Ruff passed before native execution. Tests
cover prior/fresh separation, original-source and eight-second latch clocks,
expired/future/invalid evidence, required current envelope updates, unchanged
physical factor dataflow, transactional rejection and independent provenance
validation. Native execution retains the prior recipe's 90-second overall bound
and adds the existing optional 12-second no-progress rejection; this only rejects
stalled trials and is not a relaxed success criterion. The original 90-second
failed trial remains separate. Native outcomes will be recorded after execution.
Main and long-service remain held.

## Native result at 1b9bd2b

The separate 8-second-delay crossing ran after both 60-frame stationary human/
envelope gates passed. It **failed** `progress_timeout` at 52.166667 navigation
seconds: only 1/3 waypoints, 1.511750 m endpoint goal error. The original 90-second
limit was retained; the explicit 12-second optional progress watchdog ended this
trial earlier. This is a rejected development trial, not proof of global
nonconvergence or a matched causal CRA comparison.

Minimum independent monitored clearance including the full sampled braking
sweep was .514525 m (stopping-only minimum .567527 m). Minimum sampled CBF
margin was .276074; translation stop drift .000360 m and yaw drift .013789 rad.
All 519 live envelope containment rows passed, maximum centre error .165031 m,
minimum containment margin .034969 m. There were 519 fields, 237 while translating.
These are bounded independent monitor results, not universal geometry guarantees.

Passing-policy identity and original eight-second expiry checks passed across
79 held-policy rows. The single selection at LiDAR source 19.4 s used a **fresh**
accepted person cue. There were **zero retained-identity selections**. Thus this
trial exercises the enabled runtime path and its current-envelope checks, but
does not establish natural native acceptance of the new prior-based selection.
The .5-second path is covered by unit/field tests; the separate historical
eligibility audit admitted the one .2-second gap and rejected all six .7--1.8 s
gaps, without replaying a new controller trajectory.

The visible left arc failed (-.094111 m maximum during the moving encounter);
behind-passing passed. Human validation also failed its unchanged minimum of
five supported moving-kinetic rows: only three were available, with one
simultaneous-motion row. Adult mass-source, original mass-age and median world/
relative velocity checks passed (.156372/.017524 m/s median errors). Retain this
sparse positive-ownership/fresh-association limitation rather than changing
the gate or retaining mass longer. The 34,132 object assessments had 144 visual
instances, zero complete four-factor assessments and 135,823 configured factor-
prior uses; this is still prior-assisted navigation development.

An exact-time join found 74 accepted-plan/executed-command pairs after 40 s,
with zero extra CBF command changes (1e-3 tolerance). Median planned and executed
forward speeds both were .0001606 m/s; 67 pairs used an active guidance reference
with zero horizon goal progress. These records locate late nonprogress in the
planner/reference choice, not additional final-CBF clipping in those samples.
They do not prove the cause of every stall or authorize relaxing exact safety
checks. Further bounded work should inspect local-reference progress and
fresh supported moving-kinetic coverage before another full-bypass claim.

Artifacts are in outputs/identity_prior_selection/native-crossing-8s-v1:
report.json, review-summary.json, motion-passing-validation.json,
live-human-envelope-validation.json, stopping-sweep.json, control-join-audit.json,
navigation-result.png and human-navigation.mp4. The complete field history is
live-fields.jsonl; live-fields.json is only its latest-five checkpoint. The
launcher returned zero despite the failed gate; execution.json and the saved
validation reports preserve that distinction. PR #143 remains a separate draft;
main and long-service remain unchanged.
