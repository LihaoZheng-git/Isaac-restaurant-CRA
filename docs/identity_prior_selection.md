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
