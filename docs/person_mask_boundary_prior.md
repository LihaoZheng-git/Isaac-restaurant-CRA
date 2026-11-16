# One-pixel person-mask association prior

The6d0254a raw source23.8 s frame has fragment522 with37 currently observed
LiDAR returns:36 project inside the fresh .831622 person mask; one projects
exactly1 source-image pixel outside at(290,124). All37 project into the image,
parent-mask coverage is1 and max child-to-parent3D gap is.277696 m. The strict
helper rejects the whole association. The navigation trial consequently stops
at24.0 s source; its failure is retained. This observation alone does not
identify segmentation error versus acquisition/projection uncertainty.

`--human-fragment-mask-boundary-pixels 1` is an explicit, bounded **association
PRIOR**, default0. For the separated bare-adult experiment it requires:

- >=95% child returns inside the original current person mask; every remaining
  return inside its one-pixel axial neighborhood (diagonals are farther than1).
- All original current return/projection, >=3 distinct points, <=10% parent size,
  <=.30 m actual3D gap, >=90% parent-mask coverage, untruncated/confident unique
  person, parent identity and original RGB/LiDAR/identity source gates.
- No child overlap with any competing person's similarly bounded mask.

Only still-observed parts qualify. The proof is labeled `status=prior` and
records the original mask coverage, boundary-return count, configured source
pixel prior and original identity/frame epochs. Strict full-mask proofs remain
inferred. The continuity guard requires its matching explicit one-pixel opt-in;
legacy guards reject prior proofs. A missing child cannot be cleared with an
old proof. The parent stays under the existing current-positive loss watch.

The original masks, raw occupancy, physical factor inputs/scores, body extent,
source clocks and all exact control/clearance gates are unchanged. This is not
a calibrated camera-error bound, hidden-body reconstruction, generic occlusion
recovery, SAM acceptance or negative contact/thermal evidence. A separate object
touching the person remains outside the declared separated-adult scope.

Exact original frame replay creates the labeled522 prior while the strict
path rejects, retaining original23.8/23.883333 scan/camera and23.583333 identity
epochs.84 focused tests passed in4.49 s; all2,402 unit tests passed in81.19 s;
Ruff and diff checks passed. New stationary/native acceptance is pending, with
the same N6 selected-tangent controller,8.4 m actor-only event,.5 m/s human,
all physical/flux/lambda policies, observed occupancy and .02 m clearance guard.

At bfe507d the new native trial passed the parked and moving-human component
gates and reached3/3 waypoints in23.033333 navigation seconds:229 fields/218
translating,.028942 m minimum clearance,.001771 m drift and.032018 CBF margin.
Full bypass FAILED: -.000288 m left arc and-.021533 m behind projection. There
were25 active selected references, but no visible detour. The committed-field
audit contains **one boundary-prior proof**,519->72 at23.8 s, among15 current
part rows. An earlier event-only tally incorrectly reported zero: the sparse
coverage-event stream omits successful part-only associations. This establishes
native use, not a causal attribution of arrival to the tolerance. All15,105
object assessments had zero complete four-factor sensor assessments(65 accepted
instances), while human motion had5 supported and2 simultaneous rows.

The first selected22.9 s SHA-checked snapshot admits no full reference from
the bounded4800-candidate search (both original and selected-tangent soft
rankings). The offline fixed turn-pattern probe admits only rightward-.4 rad/s
rotation among its tested patterns; no tested left pattern passes unchanged
exact geometry/DTCBF. This is not a global infeasibility proof. Keep the tight
case and evaluate any earlier lead-time fixture separately, without changing
algorithm, safety thresholds or the failed result.

The same field audit exposes three policy inconsistencies: accepted parts519
(23.8 s) and567(25.9/26.3 s) retain generic right passing/b=-1.3203125 while their
current human parent72 has selected left/b=-1.7. Part association currently
affects only the continuity guard. A separate control-policy consistency option
should preserve physical child CRA/unknowns and all source clocks rather than
copy the parent's physical factors or add geometry.
