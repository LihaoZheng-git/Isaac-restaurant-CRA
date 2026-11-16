# Restaurant visible-plume component

The real bowl now has a separate zero-wheel, three-phase diagnostic: explicitly
clean reference validation, unchanged/no-plume observations, and visible
rendered puffs. Puffs are session-layer visual fixtures above the authored bowl
mesh, not steam physics, temperature, aerosol identity or supplied factor scores.
RGB features supply every cue value. The source object need not be classified
as soup or an open container. Original scene/robot bytes remain unchanged.

## Registered live association remains failed

Both `bowl-plume-v1` and `bowl-plume-v2` collected 18 observations per phase and
passed the bowl's mass/contact/motion component check. The **registered object
plume gate failed**: LiDAR/SAM instance identities fragmented between the clean
and later phases, so new IDs lacked valid object-associated clean references.
No fresh thermal cue was accepted in either comparison phase. Unknown thermal
remained an explicit .25 factor prior; no validity/quality threshold was relaxed.
This is not moving-camera plume acceptance or a repair of live identity.

The collector now saves all native RGB phase images, copied stationary poses,
source epochs, accepted per-object box/confidence/mask-pixel diagnostics and
the rejected registered-method result. It never stores authored object masks
or score labels as estimator inputs. Multiple clean-validation frames permit
an observed instance to acquire a reference during that declared clean phase;
the reference flag is disabled during both scoring phases. Reference resets
inside the validation phase are not counted as valid temporal cue windows.

## Separate fixed-source PRIOR component passed

`restaurant_stationary_plume_replay.py` reuses the **existing fixed-camera RGB
method**, with an explicit `--static-source-region-prior` opt-in and separately
confirmed `--clean-reference-validated` scope. It uses the last uniquely matched
accepted SAM source box from the clean phase (native box [444,135,477,148],
original track 67, original timestamp 3.3 s). The match to the authored fixture
is independent evaluation/configuration information only. It supplies no heat,
class or risk score. The box then becomes a declared unchanged source-region /
identity **PRIOR**; it does not establish current LiDAR identity or refresh the
original sensor measurement timestamp.

The replay rejects camera displacement/yaw beyond .01 m/.01 rad and requires
zero-wheel records, distinct source epochs, usable source/air/background
regions and the original fixed-camera temporal/photometric gates. The
configured region binding is a prior evaluated at each current RGB capture;
the original observed source timestamp is separately retained in the report
and scalar evidence. The resulting heat/cue status remains PRIOR. This is a
bounded RGB component with a source prior, not sensor-only thermal estimation.

On the actual 54-frame `bowl-plume-v2` sequence:

| Quantity | Unchanged view | Visible plume |
| --- | --- | --- |
| Available component windows | 16 | 16 |
| Median continuous plume cue | .0000 | .32920 |
| Median thermal factor score | .13156 | .38164 |
| Median factor-specific phi | .63363 | .73913 |

The last two rows are **offline factor-specific evaluations**, holding the
other three factors at the existing .25 missing-factor priors and using a
separately labeled unit-interaction PRIOR with .5 confidence. They are not the
robot's actual route interaction or navigation phi, and were never supplied to
wheels. Source/prior confidence and uncertainty are retained; no unknown
factor is treated as zero. The unchanged-view zero is a visible-cue observation
in this validated scope, not evidence that the bowl is cold.

The positive phase's first five usable windows still returned zero, before
the cue increased through .2125, .2709, .3292 and up to .5033. Preserve those
misses; this successful bounded ordering check does not guarantee plume recall.
The method can confuse bright low-chroma foreground, requires a clean fixed
view, and does not generalize to moving cameras or hidden heat. No new learned
model, dataset, weights, training or environment package was introduced.

## Reproduction and tests

Run the restaurant command in `restaurant_service_build.md` with a fresh output
directory and `--restaurant-plume-diagnostic --stationary-frames 18`, without
`--motion`. The collector preserves its failed registered gate. Then explicitly
run the separate component replay:

```text
scripts/evaluation/restaurant_stationary_plume_replay.py
--source outputs/restaurant_service/bowl-plume-v2
--static-source-region-prior --clean-reference-validated
```

Reports/images are `restaurant-plume-validation.json` (registered failure),
`fixed-source-plume-replay.json` (PRIOR component), native `plume-rgb/` frames
and `stationary-plume-component.png`. Do not substitute the latter acceptance
for the registered runtime gate. The actual controller retains its existing
thermal prior in this scene until a separate association/runtime gate passes.

Software regression: 1,759 unit tests passed; Ruff and code/document diff
checks passed. Tests reject unavailable/prior-as-fresh heat, stale and duplicate
epochs, ambiguous/unowned sources, wrong proxy basis, insufficient windows and
moving/invalid camera poses. PR #117's main-merge hold remains unchanged.
