# Static-prop evidence for the bin and broken crate

The user approved a simpler demo method: remember verified mass/contact at a
static observed location, preserve those properties through visual gaps, and
keep kinetic estimation live. This opt-in supersedes the old immediate deletion
rule only for this explicitly scoped mode. It supplies no object names, authored
centers, mesh bounds, manual scores or ground-truth velocities to control.

`--static-prop-evidence` currently requires the original gated short approach.
Initial acquisition still requires the existing three-source image/LiDAR group
proof, mask/ownership gates and actual inferred or measured mass/contact. The
initial observed world bounds and positive samples remain fixed. Later valid
stationary acquisitions update properties chronologically until the existing
acquisition freeze; no score-based selection occurs. Acquisition is bounded to
2.5 m and the existing .9 m separated-prop extent. Competing locations cannot
enroll overlapping regions.

During motion a current sample can receive retained evidence only within the
original observed bounds plus an explicit .06 m location allowance. Require at
least eight current returns, eight distinct original nearest correspondences,
60 percent of current returns within the original .06 m support distance, and
80 percent coverage of each contributing raw fragment. This is a static-location
association PRIOR: newly sampled points between old rays are not claimed to have
passed complete original-surface correspondence. Conflicting locations reject;
absent support publishes nothing but keeps private evidence until original expiry.
No stored bounds or inferred surface become collision geometry. All original
raw returns remain occupied, including points outside the association region.

Retained mass/contact keep their original values, bounds, source timestamps and
frame. They become explicit priors with the existing 120 s maximum property age
and 60 s confidence half-life. No borrowed observation renews its clock. Raw
LiDAR velocities, robot velocity, route interaction and CRA aggregation remain
current. Potential/thermal are not supplied by this cache and retain their
existing unavailable-input priors in these two prop tests. Contact risk and phi
may change with current interaction/confidence; no final score is frozen.

Presentation uses current projected LiDAR points labeled `retained evidence /
LiDAR points`. It does not paint a saved SAM silhouette onto a different image or
claim a fresh visual measurement. Ordinary current accepted masks retain their
existing presentation path.

Recorded association replay of all15 live fields from each previous failed
native run passes the new location match for both targets, with exact raw
fragment-membership replay. This is development evidence, not native acceptance.
The initial focused run passed38 tests but two temporary-directory fixtures
errored inside the sandbox; the repeat with workspace basetemp passed40 tests.
Fresh independent/own60 stationary gates, short navigation, exact source/point
audits and stopping are required before v3 integration. Earlier whole-center,
mask and complete-perception failures remain separate and preserved.

Keep this mode default-OFF and the current draft/main/primary/PR117 holds.
At frozen runtime `674c893`, both fresh independent60 and own60 gates PASS.
Both native short approaches PASS: .303904 m measured travel,16 live fields,
15 translating fields,.550631 m minimum independent clearance and .002686 m
stop drift. Minimum CBF margins are .694667 crate/.687801 bin. Both retain
their original target evidence in16/16 live updates, with exact recorded-point
association and original value/bounds/source/time/frame audits. Kinetic is live:
relative speed ranges .0053--.2630 m/s crate and .0126--.2010 m/s bin.

Retained rough masses are2.71491 kg crate/.44670 kg bin; contact cues are
.34156/.15138. Confidence/prior handling gives contact factors .15812/.14385
and phi ranges .61528--.61585/.60876--.60930. These are modest proxy contrasts,
not calibrated physical sharpness or complete fresh four-factor perception.
Potential/thermal remain .25 priors. The old complete-center .20 m audit still
passes only1/16 crate rows versus16/16 bin rows; partial observed surfaces are
not full object centers. All crate selected points lie inside its independent
mesh box; maximum bin box excess is .00204 m, retained in the report.

The first audit incorrectly compared the sensor frame with the camera frame;
that failed audit is retained separately. The corrected audit uses the actual
original camera-frame provenance and passes with zero source/point errors.
Full unit suite3816PASS194.19s plus the final CLI test1PASS; focused40PASS,
Ruff/compile/diff PASS. All4604 protected prior files remain unchanged.

The short-test condition is met: proceed directly to separate v3 integration.
No full v3 acceptance is implied here. Artifacts: `outputs/static-prop-evidence/`.
