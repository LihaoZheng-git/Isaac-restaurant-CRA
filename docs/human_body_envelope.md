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
