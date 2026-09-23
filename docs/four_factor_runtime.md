# Four-factor runtime mapping

Sensor-derived `PhysicalAttributes` can now be adapted to all four CRA factors
and exposed through the existing track profile-provider protocol. Potential and
kinetic retain their energy proxies. Contact uses a configurable weighted
sharpness/hardness severity and exposure-times-orientation interaction. Thermal
uses a heat-source cue plus a separately supplied contextual interaction.

These normalized mappings are uncalibrated hypotheses. Missing, stale, future,
or disabled-prior inputs leave only that factor unknown. Steam alone cannot
create thermal severity. Factor diagnostics should be inspected alongside
aggregate `phi`, especially when conservative unknown fallbacks saturate it.

The default Safe-SAGE policy maps `phi` to normal flux and keeps lambda fixed at
1.25. Risk-dependent lambda is reserved for a separate evaluation ablation.

The stationary camera/LiDAR fusion demo now performs this mapping for every
live geometric track after optional visual-mass inference. It prints each
factor's accepted values or exact rejection reasons, followed by nominal and
conservative `phi`, boundary flux, and lambda. These parameters are diagnostic
only: they do not enter the field, CBF, MPC, or wheel controller in this demo.
On a single frame, motion and fall height remain unavailable; contact and
thermal also remain unavailable without their specialist evidence. The
resulting conservative saturation is reported explicitly rather than treating
missing factors as zero.

The first end-to-end stationary run associated a live chair track, inferred
5.495 kg with the visual-mass adapter's wide cross-domain bounds, and retained
fall height and relative speed as unavailable. Potential, kinetic, contact,
and thermal were therefore all unknown. CRA reported nominal `phi=0.0` and
conservative `phi=1.0`, producing boundary flux -1.7 with lambda fixed at
1.25. The robot drifted 0.000136 m and 0.0004 degrees, and no wheel command was
issued. This run validates plumbing and unknown propagation, not calibrated
factor accuracy or closed-loop safety.
