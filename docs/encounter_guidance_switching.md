# Encounter-dependent guidance (separate controller variant)

The user approved steps 1–3: observed encounter classification, per-human lambda
switching and controlled navigation comparisons. Rear-pass planning is deferred.
The primary comparison retains fixed lambda 1.25; encounter-dependent switching
is a separate ablation, not risk-dependent lambda or a replacement safety theorem.

## Step 1: current motion classification

`EncounterMotionClassifier` uses only current LiDAR track centroids/inferred world
velocities, validated human track IDs and the obstacle-unaware active waypoint
bearing. No actor pose, authored route, speed or phase enters it. The existing
human identity adapter supplies current positive ownership and original class
clocks; the classifier neither creates nor renews those identities.

Default engineering gates are configurable: >=20 current points, >=3 consistent
observations, <=.35 s source gap, <=.35 m/s velocity change, <=.2 m position
prediction residual, <=15 degree route-bearing change, >=.2 m goal distance.
Speeds <=.12 m/s may classify stationary; moving classification requires
.2..2 m/s. The interval between is uncertain. Absolute velocity-route angle
distinguishes parallel (entry <=30 degrees, hold <=40) and transverse crossing
(entry >=60 degrees, hold >=50). Parallel direction is reported separately for
same/opposite travel. The middle band is uncertain; no state is carried through
missing identity, invalid continuity or unresolved motion. Every result retains
its actual scan epoch, consistency count and state-start epoch. This policy
hysteresis is not physical evidence persistence or a guaranteed collision model.

Step 1 unit tests cover both crossing directions, both parallel directions,
stationary actors, rotated world frames, angular hysteresis/turns, source gaps,
missing/replaced identities, sparse returns, excessive speeds, changed waypoint
bearings and invalid inputs. Runtime integration and navigation results are
pending on subsequent bounded branches.

## Step 2: per-human field integration

The opt-in `--encounter-guidance-ablation` requires the verified bare-human
object-map/passing path, explicit body prior and yield strategy, without extra
guidance-tracking cost or the global-zero option. Every current verified human
is classified; lambda0 eligibility additionally requires the original current
qualified passing selection (including its original eight-second expiry).
Thus the new classifier does not bypass the existing selection/identity gates.

Only the selected stable crossing parent, its body-prior cells and positively
proven current visual fragments get lambda0. Unmatched support/environment,
unverified auxiliary returns, unrelated tracks and all fallback class policies
retain lambda1.25. Parallel, stationary or uncertain classification restores
normal guidance and disables intentional yielding even if an older passing-side
preference is still latched. It does not force unsafe waiting: original exact
temporal DTCBF, dense geometry and analytical zero-command checks still apply.

The classifier is copied with the other worker state and committed only after a
complete field succeeds. A temporal gap cannot advance its accepted history.
The applied per-row audit retains classification and selection epochs; physical
CRA/lambda1.25, risk/flux/side, all raw occupancy and factor provenance are intact.
`fixed_lambda=null` explicitly identifies the mixed per-obstacle field; it is
never reported as the primary fixed-lambda experiment. Legacy defaults and the
always-zero ablation remain separate.
