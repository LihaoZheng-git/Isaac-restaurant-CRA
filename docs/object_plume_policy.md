# Object-associated visible-plume thermal policy

The user clarified that the thermal factor should describe continuous visible
plume evidence for each object. Valid low/no-plume observations can support a
low cue score without proving coldness. Hot soup is an example, not a class to
predict. This supersedes the previous open-container requirement for the new
explicit `visual_object_plume_proxy` mode; the old contextual mode is retained.

`robot/perception/object_plume_evidence.py` reuses existing RGB foreground,
temporal plume features and continuous score mappings. It returns a thermal
record for every track in the observation window. Class names never change the
score or create heat evidence. Records can remain unavailable with specific
rejection reasons; this does not make every detected object a valid physical
thermal estimate.

## Valid low evidence versus missing evidence

The caller supplies an observed clean reference, its validation assertion and
an explicit negative-validation scope. The reference must precede at least
three distinct frames spanning 0.15 s or more. Defaults reject evidence older
than 0.5 s, moving cameras, stale/future track geometry, unmatched/ambiguous
object masks, small/low-confidence or image-truncated regions, unstable source
boxes, insufficient visible air above the source and changed backgrounds.
The RGB search region extends 1.5 source heights above the object in this mode.
This is a bounded visible-region assessment, not complete visibility of heat.

Zero detected foreground in a valid, domain-validated window supplies continuous
area/extent/persistence features with nonzero uncertainty. Positive foreground
strength and persistence raise the cue. Source association attenuates unrelated
evidence. Detector confidence affects epistemic confidence, not cue magnitude.
Configured regions remain PRIOR; detected regions remain inferred. Steam
identity, temperature, open-container state, carriage and spill stay unknown.

Observed source search columns are excluded from the common reference-background
check so a plume on object A does not invalidate a stable object B. At least 10%
independent background remains mandatory. Global lighting/camera changes still
fail validation. If the same positive foreground substantially overlaps two
object regions, both results remain unknown rather than arbitrarily assigning
that plume. A separate thermal interaction estimate uses the same track's
waypoint-corridor evidence; camera visibility is not interaction probability.

The physical adapter/provider/readiness paths accept this new thermal basis
only with explicit `allow_visual_thermal_proxy` opt-in and the correct evidence
source. Default physical/contextual behavior is preserved. No unknown inputs
are filled with zero; CRA uncertainty, aggregation and lambda 1.25 are unchanged.

## Stationary diagnostics, 2026-10-08

`scripts/demos/object_plume_isaac_diagnostic.py` captures the unchanged approved
USD plume fixture. RGB supplies foreground and temporal features; source
regions/track bindings are explicit fixture PRIORS, not successful object
detections. No robot is created, no authored plume mask supplies a score, and
no thermal physics is simulated.

The final rendered controls in `outputs/object_plume/isaac_controls_v2` gave:

| Condition | Continuous plume cue | Interpretation |
|---|---:|---|
| No visible plume | 0.000 | Validated low cue |
| Small intermittent plume, 2/3 frames | 0.354111 | Intermediate evidence |
| Persistent dense plume, 3/3 frames | 0.762231 | Stronger evidence |
| Distant configured region with the plume elsewhere | 0.000 | Not assigned to unrelated region |
| Unvalidated reference | unavailable | Rejected |
| Moving-camera flag | unavailable | Rejected |

An initial control run correctly exposed that the original global background
test was invalidated by the unrelated plume. Its failed report remains in
`outputs/object_plume/isaac_controls`. Independent-background exclusions plus
regression tests fix this without disabling camera/lighting checks or lowering
the acceptance requirement.

The robot demo's `--object-plume-policy` captures a fresh three-frame stationary
Hawk RGB window, invokes the existing object detector and camera-LiDAR fusion
on each frame, and assesses every observed track. The static tracking geometry
is at most 0.4 s old, with a measured simulation-pose stationarity guard and
explicit frozen-geometry scope. Its clean/no-plume validation scope is the
controlled sphere fixture; this is not general real-camera negative validation.

The fresh live sphere capture produces a thermal cue of 0.0 with interval
[0.0, 0.013125], confidence 0.55 and inferred provenance. Its route-conditioned
thermal pre-uplift risk is 0.013125; CRA's existing uncertainty policy raises
this to 0.457219. This is a low observed cue, not a zero conservative risk.
The final capture in `outputs/object_plume/live_sphere_v2` contains 27 track
records, three with accepted plume evidence; the target sphere is one of those.
The others retain explicit rejection reasons. Its 754 target LiDAR returns and
1.44 mm stationary drift pass the bounded capture/dataflow gates. Complete
sensor-only four-factor readiness does not pass, including because mass is PRIOR.
Mass remains prior-dependent, contact uses the visible silhouette proxy, and
potential still lacks justified support/landing evidence. Aggregate phi remains
1.0 due to that remaining unknown factor. The saved restaurant scene and nominal
waypoint/MPC/CBF control stack are unchanged; no new wheel motion was performed.

The baseline requires a stationary camera and a clean reference. It can confuse
bright low-chroma foreground with a plume and miss faint/non-bright aerosol.
Moving-camera deployment, general thermal hazards and all-object coverage are
not validated. The existing detector also rejects conflicting/fragmented object
associations; such tracks receive unknown records, not low scores.

Independent tests cover valid no/faint/dense/unrelated cues, class independence,
shared-source ambiguity, multi-source background changes, missing validation,
low visibility, stale/future/duplicate frames, source motion, region provenance,
adapter opt-in and readiness. Run relevant tests plus repository unit/Ruff checks
before extending the accepted observation domain.

Validation on this revision: **1,406 unit tests passed** in 35.35 s, repository
Ruff checks passed, and `git diff --check` passed. The two stationary Isaac
diagnostics above passed their stated component/dataflow gates. No wheel-motion
test, general thermal calibration or complete four-factor acceptance is claimed.
