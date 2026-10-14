# Persistent multi-object CRA map

The later user decision changes the live experiment's uncertainty policy:
estimate every observed geometric obstacle's CRA policy, retain previously
observed properties, and keep kinetic estimation live. A failed factor no
longer automatically forces the complete object's phi to one in this explicit
opt-in mode. The legacy single-target/strict uncertainty path stays available.

## Dataflow and scope

`LiveObjectPerception(..., persistent_property_config=...,
object_factor_prior_config=...)` runs the existing SAM/mass/contact/support/
plume components for every currently camera-observable LiDAR track. SAM image
encoding and the contact frame are shared across tracks. Camera qualification
still requires eight projected owned returns; association/quality/boundary
checks are unchanged. Other tracks also receive numeric policies through
configured priors. No class, density or VLM reasoning is introduced.

The all-track result exposes `objects`, each with physical attributes, adapter
diagnostics, numeric CRA control policy and independent perception readiness.
Legacy top-level attributes describe only the first track in this mode. The
field uses every track's own normal-flux mapping and fixed lambda 1.25. MPC,
analytical CBF, collision geometry, field freshness and optimizer execution
verification are unchanged. Nominal waypoint control remains obstacle-unaware.

An accepted mask owns only its selected LiDAR returns. New returns may acquire
a retained object's policy through an explicit static, axis-aligned 3D bounds
association prior with 5 cm padding and continuous geometric identity. This is
not hidden geometry or robust dynamic-actor association. Unmatched support,
environment and unclustered returns keep separate occupancy and prior policies.
The positive occupancy history remains separate from property memory. Track
clusters can merge multiple physical objects; all-track numeric scoring does
not establish all-instance visual segmentation. The field remains spatially
bounded and reports discarded cell representatives.

## Numeric priors and uncertainty

`ObjectRiskPriorConfig.factor_risks` configures four normalized engineering
priors. Initial defaults are .25 each, rather than a fitted probability or
measured hazard magnitude. With the unchanged noisy-OR aggregation an entirely
unassessed profile has phi `1 - .75^4 = .68359375`. Valid evidence retains its
original confidence and the existing uncertainty-uplift formula, using these
explicit priors rather than unit-risk fallbacks. High observed risk can still
produce phi one. Factor-specific .10/.25/.50 sensitivity comparisons are useful
future evaluations; this change is less conservative than the strict baseline.

Adapter input failures remain failures. Each numeric score reports its prior,
source mode, confidence, nominal value, basis and original input rejections.
Numeric control readiness is separate from physical perception readiness.
Missing data is not claimed to be a measured zero, grounded object, smooth
surface or cold object. Two different objects can share phi legitimately.

## Property persistence

The opt-in property store retains original mass, fall-height/support interaction,
visible contact and visible-plume inputs for a configured session. It never
caches velocity or route interaction. Default original-source lifetime is
120 s and confidence half-life is 60 s; these are scenario assumptions.
Borrowed values become PRIORS, retaining source frame/time, original status and
bounds. After acquisition, accepted intrinsic values are frozen; new
view-dependent mass/contact estimates do not renew them. Support invalidation
and original expiry still apply, and newly available missing properties can be
acquired. Repeated borrowing never refreshes their age. Oracle/fallback estimates
cannot enter the observed-property store.

Track observation continuity and a 25 cm prediction-error gate are required;
a competitor within 30 cm invalidates reuse. A move beyond 15 cm from the
support observation or a newly observed center-height change above 10 cm
invalidates the support/drop cache. Center height only invalidates a previous
support hypothesis; it never supplies fall height. Identity loss and original
expiry remove stored inputs. Unobserved heat/support changes remain outside the
unchanged-scene assumption and require a new acquisition phase.

Plume acquisition remains enabled during the initial stationary observation
window, including observations after the clean reference. Explicit
`freeze_properties()` ends that acquisition window. Navigation can then reuse
the last accepted object-associated plume cue without successful moving-camera
registration. This validates persistence of an initial cue, not moving-view
plume detection or temperature measurement.

After acquisition, visual reassessment is attempted at most once per second
per observable track by default, with at most two visual tracks per update.
Oldest-attempt scheduling prioritizes never-assessed tracks and prevents
starvation. Newly visible tracks receive numeric priors while awaiting vision.
All tracks still receive current kinetic/context estimates and numeric policies
on each processed scan, with priors while visual work is deferred. This cadence
is explicit in diagnostics, not an evidence timestamp renewal, discarded
occupancy or scheduled observation stop. The pending trained moving-view plume
model is not required for this initial-cue-persistence scope.

## Diagnostics

Run the recorded stationary sensor check with existing audited weights:

```powershell
python scripts/demos/cra_object_risk_map_replay.py --source outputs/live_cra/verified-plan-reuse-none-live --output outputs/object_risk_map/no-plume-final --visual-gap-after 6
```

The positive-plume source is `verified-plan-reuse-positive-live`. Both source
stationary gates and model hashes are verified. Copied RGB/LiDAR frames are
real simulated sensor observations; pose/floor references remain oracle and
ground continuity remains prior. Injected missing SAM outputs are explicitly
synthetic failures, not natural occlusion. Replay never starts Isaac or moves
wheels. It is not fresh sensor acquisition or motion acceptance.

The live Isaac demo exposes `--persistent-object-risk-map` and configurable
`--object-factor-prior-risk`. Its separate stationary numeric-policy gate
requires 12 constructed fields and accepted visual observations, and reports
individual perception readiness separately. Passing that gate does not certify
complete visual estimation for every track. Navigation uses initial plume
persistence and current kinetic/context/field updates; it is a different
experiment from the previous moving-plume-registration target.

Validation results and motion limitations are recorded below after diagnostics.

The new map opts into connected equal-policy field regions. Geometry with
identical phi, normal flux, lambda and passing side shares one connected field
component regardless of arbitrary LiDAR track fragmentation. Different policies
and disconnected physical regions stay separate. Per-object identities,
occupancy, scalar policy values and diagnostics remain unchanged. A synthetic
fragmented/joined-region test checks identical boundary vectors and full solved
guidance fields. This is a field representation adaptation, not exact paper
equivalence or proof of global navigation convergence; legacy callers keep
their previous grouping default.

Temporal gaps require a separate unpublished recovery baseline. If consecutive
perception fields are farther apart than the unchanged .35 s derivative limit,
no control field is returned and tracker/evidence/accepted-field state stays
uncommitted. The next actual field can recover only using a real baseline-to-
current source interval within that limit; source epochs are not renewed and
missing derivatives are not replaced with zero for motion. Field/command age
checks and full current-state MPC/CBF verification still apply after recovery.
