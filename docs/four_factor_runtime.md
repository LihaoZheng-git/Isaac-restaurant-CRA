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
