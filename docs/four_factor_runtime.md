# Four-factor runtime mapping

Sensor-derived `PhysicalAttributes` can now be adapted to all four CRA factors
and exposed through the existing track profile-provider protocol. Potential and
kinetic retain their energy proxies. Contact uses a configurable weighted
sharpness/hardness severity and exposure-times-orientation interaction. Thermal
uses a heat-source cue plus a separately supplied contextual interaction.

For an associated thermal track, the stationary runtime can derive that
interaction from the same track's waypoint-corridor score. This reuses measured
or inferred route geometry only; it does not infer carriage, spill,
temperature, or burn severity. Missing or stale heat/route evidence leaves the
thermal interaction unavailable.

These normalized mappings are uncalibrated hypotheses. Missing, stale, future,
or disabled-prior inputs leave only that factor unknown. Steam alone cannot
create thermal severity. Factor diagnostics should be inspected alongside
aggregate `phi`, especially when conservative unknown fallbacks saturate it.

The default Safe-SAGE policy maps `phi` to normal flux and keeps lambda fixed at
1.25. Risk-dependent lambda is reserved for a separate evaluation ablation.

The live evidence builder accepts explicit per-track contact and thermal
specialist providers alongside mass and fall-surface providers. Their returned
attributes retain their own measured or inferred provenance. If either
specialist is absent, that group stays unavailable; the fusion layer never
turns a missing detector result into a zero hazard score.

The camera/LiDAR diagnostic can optionally load a local two-label checkpoint
through `HOT_SOUP_DETECTION_CHECKPOINT`. It runs plume/container inference on
the stationary RGB frame, associates the selected container with exactly one
projected LiDAR track, attaches the resulting continuous thermal attributes,
and writes a thermal diagnostic. With no configured checkpoint the thermal
group remains unavailable. No project checkpoint currently satisfies the real
held-out data gate, so this optional path is integrated but not claimed as a
validated sensor model.

The stationary camera/LiDAR fusion demo now performs this mapping for every
live geometric track after optional visual-mass inference. It prints each
factor's accepted values or exact rejection reasons, followed by nominal and
conservative `phi`, boundary flux, and lambda. These parameters are diagnostic
only: they do not enter the field, CBF, MPC, or wheel controller in this demo.
The diagnostic now waits for a second, strictly newer LiDAR timestamp before
building the evaluation frame. Its tracker uses a configurable 0.15 m/s
stationary gate and requires two consecutive motion updates before publishing
nonzero obstacle velocity. A matched two-frame track can therefore contribute
sensor-derived stationary motion evidence without turning a one-frame centroid
jump into motion. Fall height remains unavailable; contact and thermal also
remain unavailable without their specialist evidence. Conservative saturation
is reported explicitly rather than treating missing factors as zero.

The first end-to-end stationary run associated a live chair track, inferred
5.495 kg with the visual-mass adapter's wide cross-domain bounds, and retained
fall height and relative speed as unavailable. Potential, kinetic, contact,
and thermal were therefore all unknown. CRA reported nominal `phi=0.0` and
conservative `phi=1.0`, producing boundary flux -1.7 with lambda fixed at
1.25. The robot drifted 0.000136 m and 0.0004 degrees, and no wheel command was
issued. This run validates plumbing and unknown propagation, not calibrated
factor accuracy or closed-loop safety.

The subsequent two-frame stationary run inferred motion for the same live
chair track. With 7.781 kg mass and stationary-gated track velocity, kinetic
CRA accepted severity 0.001 and route interaction 0.150 at confidence 0.25.
Potential stayed unknown because neither fall height nor harmful-fall
interaction was observed. Contact and thermal also stayed unknown, so
conservative `phi` remained 1.0 while nominal `phi` was near zero. Robot drift
was 0.000234 m and 0.0003 degrees. This establishes temporal sensor plumbing;
moving-object accuracy and the 0.15 m/s stationary threshold still require a
separate controlled calibration.
