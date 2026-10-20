# Human body-envelope occupancy prior

The user approved this separate bounded geometry prior after the 7791a5a
native guidance ablation reached the independent human envelope despite positive
sampled CBF margins. The failed trial and post-stop endpoint limitation remain
in docs/guidance_tracking_ablation.md. Main and long-service stay held.

## Runtime contract

This opt-in adapter receives existing current LiDAR tracks and the same-frame
accepted person/SAM association already audited for human mass. It reads no
actor state, simulator radius or authored body pose. A configured .40 m body
radius plus .20 m centroid uncertainty forms a **PRIOR** filled occupancy disk
around the observed LiDAR centroid. Neither radius is measured anatomy or a
universal bound. Both are explicit engineering assumptions for the separated
adult scenario; independent containment checks must pass.
LiDAR/world conversion retains the declared oracle ego-pose/extrinsic references
and the existing camera read-time approximation; this is not sensor-only geometry.

Each current disk needs >=8 positive track-assigned LiDAR returns. Retention requires the same
matched track, increasing observation count, a <=.25 s LiDAR gap, <=.25 m centroid
residual against the previous observed velocity, and no substantial competing
track within .35 m (competitor >=half the target point count). These are bounded
identity gates, not crowd/crossing identity guarantees. The original visual
identity/frame is retained for at most 30 s. A distinct fresh accepted camera
frame may refresh identity; borrowed/duplicate frames cannot. This is a separate
identity PRIOR, not a current class measurement or an extension of factor clocks.
A missing, expired or ambiguous known identity blocks publication of a new
control field until a fresh accepted visual association can resolve it. It never
silently erases the prior body or transfers its factors to another track.

The prior disk fills grid cells through the existing occupancy builder, retaining
all original/unmatched/support returns. It uses a separate auxiliary track ID
and explicit provenance. Its CRA policy is at least the configured unknown
factor policy and the parent's computed risk. Fresh-person class cues and passing
preferences may apply through the parent mapping, without inventing current
class evidence from retained identity. The geometric CBF baseline uses current
parent LiDAR velocity for these same cells. Lambda remains 1.25. Comparisons must
match the full envelope configuration.

Known person identity also guards the catalog mass model's domain. Generic ABO
inference cannot overwrite the human mass on that track after a person-mask
miss. Only a fresh accepted person association supplies a new configured adult
mass interval. Otherwise the existing original mass retention/3 s expiry applies,
then the configured factor prior. Body identity never supplies or renews mass,
contact, motion, height, support, plume or route-interaction evidence.

## Enable and validate

`--human-body-envelope-radius .4` explicitly enables the prior;
`--human-envelope-centroid-uncertainty .2` and
`--human-envelope-identity-max-age 30` configure its assumptions. Defaults leave
the prior disabled. The current adapter requires standalone persistent human
object-map mode, the audited person detector/mass-prior association, and no
carried-object partition or static occupancy memory. Broader identity cases and
carried-human geometry remain separate work.

Unit tests cover original clocks, duplicate frames, gaps, missing/reused tracks,
competing geometry, expiration, raster/frame transforms, actual field integration,
transaction rollback, catalog-domain rejection and geometric-baseline velocity.
The independent validator checks source-time alignment and containment of the
configured actor envelope, without supplying those oracle values to runtime.

A new 60-frame stationary human/envelope gate is required before wheel tests.
Every braking frame is now monitored, including rear swing during rotation;
translation drift alone is insufficient. Native results must be recorded here
before claiming successful bypass or sensor-derived full-factor acceptance.
The previous 7791a5a native failure had zero complete four-factor assessments;
this geometry prior does not solve moving-camera plume or general perception.


## c048777 native result

All 2,156 unit tests and repository Ruff passed. The new native crossing passed
both 60-frame stationary gates. Its minimum stationary containment margin was
.008289 m; the completed moving audit then passed all 894 current envelopes,
with maximum observed-centroid/reference-centre error .170730 m and minimum
containment margin .029270 m. Earlier in-run concern about the moving centroid
bound was provisional; the completed source-aligned audit passed. No correction
to the centre model was needed or integrated.

The human navigation/mass/motion checks passed, including six supported moving
kinetic rows and two simultaneous-motion rows. No supported human kinetic row
used catalog mass. Full-frame mass/contact/thermal estimation is still not
established: there were 149 accepted visual instances, 59,454 object assessments,
zero complete four-factor assessments and 236,864 configured factor-prior uses.

**The full bypass failed**: duration_limit at 90 navigation seconds, 2/3
waypoints, .896943 m endpoint goal error. Minimum independent clearance was
.450916 m, including the braking audit (.556785 m stopping minimum); translation
stop drift .00002496 m and yaw drift -.0002834 rad. There were 894 fields/349
while translating. The .325369 m encounter-specific arc and behind test passed,
but no passing-policy row was selected, so this is not a validated deliberate
human bypass or an algorithmic improvement over the primary arrival.

A source-history audit found qualifying left-side human candidates without a
same-frame accepted person cue. Some had older identity ages .7--1.8 s; one had
.2 s. The separate proposed <=.5 s identity-prior selection mode remains a user
research decision, **not implemented**. Existing same-frame selection, original
side-policy clocks and all physical evidence expiry rules remain unchanged.

An exact-time join of accepted plans/commands found 305 valid rows after 40 s,
with zero additional CBF command changes and median planned/executed forward
speed .0001989 m/s. The planner itself selected near-zero progress. These data
do not support blaming the final CBF projection or unicycle mapping alone.
The strong beta=100 reference candidate requires further progress/policy review.

A separate offline ORACLE-selected track replay of 21 saved 7791a5a source poses
rejected a proposed uniform-semicircle centre correction: maximum error worsened
from .156757 to .229682 m, with five corrected samples exceeding .20 m. It is
not integrated. The runtime remains the explicitly named centroid-centred prior.

Artifacts: outputs/human_body_envelope/native-crossing-8s-v1/report.json,
review-summary.json, stopping-sweep.json, control-join-audit.json, and
live-human-envelope-validation.json; native-development-traces.png compares
engineering trials with changed settings, not a causal CRA baseline comparison.
The launcher returned zero despite the failed gate; do not use it as acceptance.
Main/long-service remain held and the experimental PRs remain drafts.
