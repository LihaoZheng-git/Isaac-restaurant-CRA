# Positive-return fragment continuity

This is an explicit bounded bare-human continuity option, not a new physical
factor model, person identity guarantee or free-space inference method.

The late-crossing v2 stop in `docs/guidance_control_audit.md` concerned a fragment
last observed before the actor began walking. A body can split and rejoin as
the camera/LiDAR viewpoint changes. The existing .25 m parent-centroid gate
can reject a connected larger cluster even when its actual returns cover the
old fragment location.

An independent cross-trial geometry diagnostic illustrates that distinction:
the v1 `coverage-merge-1026.npz` scan at 18.2 s has six current track-73 returns
within .10 m of v2's former fragment centroid; the current parent centroid is
.298892 m away. This is a counterfactual cross-trial query, not proof of v2
identity continuity or a runtime input. Both original failures remain failures.

The live opt-in is `--human-positive-fragment-extent`. The legacy centroid
route remains unchanged. A farther parent may use the new route only with:

* an original actual co-observation no more than .15 s before the fragment's
  last source time and a fragment age no greater than .35 s;
* current epoch-aligned positive geometry, at least three returns within .10 m
  of the old centroid, and the original one-to-one uniqueness/competitor gates;
* the direct old-location query. A farther velocity-advected query is ineligible.

The association is inferred from observed planar geometry. The event preserves
the original source time, co-observation time, actual query, supporting return
count and parent-centroid distance. Native logs save the matching raw positive
geometry. No mass/contact/height/plume value or class transfers; no confidence
or factor clock renews. All current occupancy remains in the field. Missing,
old, ambiguous or uncovered locations still block. This is outside the carried-
object and crowded/occluded multi-person scope.

Recovery-candidate geometry also permits an early rejection after one actually
unsafe query, in four-interval batches under the existing feasibility-pruning
opt-in. A safe candidate receives every original endpoint, dense and padding
query. The full public validation path remains unchanged. Tests compare the
winning horizons and query counts and deliberately exercise inter-sample
collisions; no safety bound or objective is relaxed.

Native acceptance must be recorded separately from these software tests.
The implementation passed 2,082 unit tests and repository Ruff. The first
prespecified native gate uses the previously approved primary 8 s crossing
timing, .5 m/s left-to-right actor, original camera and .90 turn-search recipe.
It is a separate case from the retained 13 s stress failures, not a matched
causal comparison of the continuity option. Its own 60-frame parked gate must
pass before wheels. No native acceptance is claimed before that result.
Use the original priors, lambda 1.25, RGB read-time approximation and oracle
ego-reference provenance. The previous main/long-service holds remain.
