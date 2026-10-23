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
