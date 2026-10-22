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
