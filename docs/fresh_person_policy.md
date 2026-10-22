# Fresh person-mask policy association

The approach-triggered N6 fixture reached all three waypoints and passed the
moving-human perception component, but failed the visible curved/behind gates.
Its policy audit qualified human track76 at23.0 s but rejected it as visually
unverified until24.1 s: current motion qualification did not imply fresh SAM
physical-instance readiness. Preserve that1.1 s class-eligibility delay.

The already-installed audited `yolo11n-seg.pt` detector was running each frame
for person attention/domain guards. Exact original RGB23.283333 s replay
(scan23.2/frame212) gives one confident(.918585), untruncated person mask covering
all1,786 projected parent foreground returns, before the old24.1 selection.
The parent identity's original SAM/person camera epoch is22.583333 s. The new
cue preserves that epoch and supplies only a **new current inferred class cue**.

## Explicit alternative, no physical evidence substitution

`--human-fresh-person-policy` is opt-in for the separated bare-human,
observed-occupancy/person-selected passing experiment. It reuses the existing
audited detector and calibrated current projection; no new model/weights,
language reasoning or VLM is added. A current class cue may qualify passing
without waiting for a new physical SAM assessment when all these hold:

- Current validated human parent identity/positive LiDAR support, matched track,
  original identity provenance retained and scan epoch unchanged.
- Same RGB source frame/time; camera-scan age<=.15 s under the existing declared
  read-time clock approximation.
- At least20 projected parent returns, >=90% person-mask and foreground-depth
  coverage, confidence>=.5, >=200 mask pixels and no image-boundary truncation.
- Unique person/parent ownership; competing masks, two parents sharing a person,
  inconsistent source arrays, sparse/occluded background or stale sources reject.

The foreground filter's separated first-depth-layer hypothesis/gap(.5 m) is
recorded explicitly. It is association evidence, not directly measured camera
visibility or hidden-body geometry. No point, physical score or control margin
is manufactured or removed. The helper permits tighter configurable coverage/
point/depth thresholds while preserving these bounds.

The passing policy keeps its existing current velocity, consistency, point,
prediction and original selection-age gates. Same-frame accepted SAM/person
evidence takes precedence. Diagnostics record both legacy SAM and direct-detector
eligible IDs and retain the original selected visual cue unchanged while policy
hysteresis holds. Transactional field failures do not commit that cue.

This source is explicitly **inferred class-policy evidence**, not fresh SAM
acceptance. Mass/contact/support/fall-height/plume values and clocks, unavailable
physical inputs, factor priors, aggregate CRA, lambda and occupancy remain
unchanged. Only earlier eligibility can change the existing passing-side field
policy. Original parent-domain identity clocks are not renewed by borrowing or
by this detector path. Complete four-factor perception is still a separate gate.

## Validation scope

Component tests reject stale/misaligned frames, duplicates, weak/partial or
truncated masks, observed foreground occlusion, competing people/parents, sparse
support, unmatched/old identities and invalid provenance. They also preserve
input arrays/identity epochs and demonstrate same-source eligibility only after
the unchanged motion gates qualify. Exact original sensor replay accepts76 at
23.2 s with the original22.583333 identity epoch and explicitly reports no
physical-evidence refresh. This does not establish selection at23.0 s, natural
occlusion recovery, crowd tracking or a native curved-bypass pass.

Validation before native motion: all2,347 unit tests passed(100.25 s); the
67-test person-policy/passing/fragment/flux subset passed(5.74 s), followed by
32 focused person-policy/passing tests after diagnostic-only threshold metadata
was added(5.10 s). Ruff over `robot`, `tests`, `scripts` and `git diff --check`
passed. Native acceptance below remains independent of these component checks.

The next own stationary/native trial keeps N6, the8.4 m actor-only trigger,
.5 m/s human, legacy planar-filtered reference mode and all physical/source/
clearance checks unchanged, isolating this policy-cue path before any additional
direct-guidance ablation. Native latency/selection/arrival/passing acceptance is
recorded below. Preserve all old failed results and primary defaults.

### Native cue acceptance, navigation rejection at75f45ce

`outputs/paper_flux/fresh-person-human17-observed-N6-v1` passed its own60-frame
parked gate with zero wheel commands. It selected the current observed human's
left/trailing preference at23.0 s, the **first qualified motion row**; legacy
SAM person eligibility was empty. The selected .906449 detector cue covered
99.1662% of1,799 projected parent returns and retained the original22.583333 s
identity timestamp. There were142 current detector-cue rows and10 selected rows
using this original cue. The previously observed class delay is resolved in
this run, not established as a universal latency guarantee.

**Navigation FAILED**: the independent clearance guard stopped at15.866667
navigation seconds /1 of3 waypoints, .019087906 m minimum clearance (below the
unchanged .02 m threshold), .000880452 m stopping drift,158 fields/154 translating.
The human-motion gate and curved/behind gates also failed: -.000250 m left arc,
+.273273 m behind projection. All10,976 object assessments had zero complete
four-factor sensor assessments;52 accepted visual instances do not change this.

Planner diagnostics show near-straight checked references before the crossing,
then infeasible horizons and an unsafe current configuration as the person
approaches the nearly stopped robot. The passing choice is not an executable
trajectory, and observed surface occupancy plus a stronger boundary flux does
not reconstruct hidden human extent. Preserve this safety rejection rather
than relax clearance, restore clocks, lower risk or call the run a bypass.
