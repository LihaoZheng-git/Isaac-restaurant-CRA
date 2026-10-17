# Restaurant prop observation audit

The first food bowl remains the accepted stationary component and short-route
example. Additional zero-wheel views on 2026-10-17 tested the existing pot,
counter knife block and front-table cup. **None of those additional views
passed the component gate.** Their placement and asset names never supplied
factor inputs, and no motion was enabled for them.

`--restaurant-view bowl|pot|counter|front` selects an explicit robot setup pose
and an independently matched report target. It does not select a target for
SAM or mass inference; all eligible observed tracks are still assessed.
The original bowl remains the only motion-enabled view in this branch.

| Observation | Result | Recorded reason / scope |
| --- | --- | --- |
| Pot, first front-aisle view (`props-pot-v1/v2`) | Rejected | No unique pot row; chairs/table edge obscure the object. v1 stopped at the legacy first-frame guard; v2 collected the full 18-frame diagnostic. |
| Pot, rear aisle gap (`props-pot-v3`) | Rejected | 16 unique proximity matches, two ambiguous; zero accepted pot instances. The 856-return track includes table/chair structure. Prompted masks fail unchanged quality/ownership gates. |
| Counter knife block (`props-counter-v1`) | Rejected | No unique target match across 18 frames; the higher counter lip obstructs this low-camera view. |
| Front cup (`props-front-v1`) | Rejected | Two unique proximity matches, 16 ambiguous; zero accepted cup instances. Proximity alone does not establish visual ownership. |

The rear pot view's policy remained four .25 configured factor priors, phi
.68359375. This is numeric policy dataflow, not sensor-derived pot mass,
contact, fall height or thermal evidence. No score was reduced to make a prop
appear benign, no shape/mass was taken from USD, and no support occupancy was
removed. Offline experimental plane cuts did not supply runtime inputs and
were not integrated; table/object separation needs a separately validated
method before stronger perception claims.

Restaurant diagnostics now permit a first RGB frame with no accepted instance
so the complete observation window can be recorded. They have no validated
negative-plume scope. The unchanged final component gate still blocks wheels;
legacy fixture clean-reference guards remain unchanged. Reports use generic
`target_*` check names and require the explicitly requested asset, never a
different conveniently observed object. Unit regressions cover this behavior.

The camera mounting, acquisition aspect, SAM thresholds, physical adapters,
priors and fixed lambda are unchanged. Raw records/images are under
`outputs/restaurant_service/props-*`; test logs use
`outputs/restaurant-service-props-*.log`. The proposed next steps are bounded
route coverage and an independently validated server asset, while these
unresolved tableware observations remain explicit priors.

Validation: 1,745 unit tests passed, including the named-target and finite-view
regressions; repository Ruff and code/document diff checks passed.
