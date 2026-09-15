# CRA repository guidance

- Keep waypoint nominal control obstacle-unaware; MPC and analytical CBF create online intervention.
- Runtime CRA uses visual/physical estimation without language or VLM reasoning.
- Preserve measured, inferred, prior, oracle, fallback, and unavailable provenance. Missing evidence is not zero. Object height and center height are not fall height.
- Evaluate factor-specific diagnostics when unknown contact/thermal inputs saturate aggregate `phi`; never mark unsupported hazards absent to improve results.
- Keep lambda fixed at 1.25 for the primary comparison. Treat risk-dependent lambda as a separate ablation.
- Verify candidate weights, licenses, datasets, and compatibility before claiming a visual model is integrated.
- Validate perception stationary before enabling new wheel-motion experiments. Report oracle/prior demonstrations separately from sensor-derived results.
- Preserve user changes. Do not force-push, rewrite history, bypass checks, upgrade dependencies silently, or merge unvalidated work.
