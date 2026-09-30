# CRA factor confidence review

This review checks how evidence confidence and provenance reach the four CRA
factors. Confidence describes support for an estimate; it is not the factor's
hazard magnitude. Severity and interaction remain separate continuous values.

| Path | Accepted sensor evidence | Conservative handling |
| --- | --- | --- |
| Potential | mass, explicit fall height, and harmful-fall interaction | Missing inputs stay unknown. Object or center height never substitutes for fall height. Prior and fallback dependencies retain their status and confidence caps. |
| Kinetic | mass, relative speed, and route-crossing likelihood | Unconfirmed-motion fallbacks remain fallback evidence. The nominal controller remains obstacle-unaware. |
| Contact | sharpness, hardness, dangerous-part exposure, and route orientation | General mask geometry does not become physical sharpness or hardness. Contact masks retain their capture timestamp. |
| Thermal | continuous heat-source evidence and same-track route interaction | Plume magnitude is separate from model confidence. Mask, context, and association cues each pass freshness checks. No temperature or hot/not-hot claim is made. |

The adapter applies the weakest contributing confidence after the same
configurable prior and fallback caps for all four factors. Analytic energy,
volume/density mass, learned visual mass, and other derived values preserve any
prior, fallback, or oracle dependency. Sensor-domain readiness accepts only
measured or inferred inputs; a positive numeric fallback is not promoted to
sensor evidence.

Derived evidence cannot acquire a fresh timestamp merely because a calculation
ran later. Fall height, mass fusion, contact masks, and contact geometry retain
their oldest contributing observation time. The visual-mass estimator and
continuous plume pipeline also reject stale or future inputs before publishing
a current estimate.

These checks validate contracts and conservative propagation, not calibration.
The energy scales, normalized contact mapping, continuous plume mapping, visual
mass error bound, confidence caps, route interaction, and CRA fallback risks
remain engineering hypotheses to evaluate. Unknown factors continue to uplift
the separately labeled four-factor aggregate and may saturate `phi`; factor-
specific reports must identify those unknown fallbacks rather than replace them
with zero.
