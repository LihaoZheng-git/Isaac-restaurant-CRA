# Observed shape and rough heaviness

The simple-object estimator can now use an observed round outline as positive
evidence of a low **visible geometry** contact cue, and a curved LiDAR surface
to infer spherical size. Neither a semantic class name nor a failed edge
detection establishes safety. This is a bounded component improvement; the
different-observed-objects -> different aggregate phi -> matched wheel-control
experiment is still pending.

## Inputs, assumptions and runtime path

- `robot/perception/visual_shape_contact.py` scores a resolved object mask using
  circularity deficit, PCA elongation, concavity and large polygon corners.
  Their maximum is a continuous engineering proxy. Minimum pixel count,
  component coverage, border margin, polygon scale, confidence and interval
  allowances are configurable. Small, border-truncated, fragmented, holed and
  low-confidence masks are unavailable. Tiny disconnected mask fragments up to
  the configured 2% allowance increase uncertainty. Occlusion can still be
  missed; a clean outline cannot establish complete visibility or reveal
  interior/hidden hazardous features. Physical sharpness, hardness, edge radius
  and 3D route orientation remain unavailable.
- `build_track_shape_contact_proxies` binds same-frame masks only through
  unambiguous camera-LiDAR associations. A detected exposed-edge part is combined
  by taking the maximum cue and interval endpoints, so a round outline cannot
  suppress it. A rejected outline retains valid part-only evidence. Ambiguous
  associations and mismatched frame IDs/timestamps are rejected.
- `robot/perception/spherical_mass.py` fits an inferred sphere to associated
  LiDAR returns using centered least squares. Point count, conditioning,
  residual, curved-surface coverage and radius scope are checked. No authored
  radius or center enters the fit. The spherical hypothesis extrapolates the
  unseen back surface; it is not complete measured geometry. Radius/volume
  intervals and confidence are engineering assumptions, not calibrated bounds.
  The existing visible-envelope box-volume rejection remains unchanged.
- Mass uses fitted volume times a supplied **effective bulk density** interval,
  including hollow/solid construction and contents. There is no default density
  or material detector. A density prior keeps mass and dependent kinetic
  evidence `prior`, even though size is sensor-derived. The demo only uses this
  explicit option when the learned mass estimator has no supported prediction;
  it does not replace a supported learned estimate.
- The adapter/provider/readiness path now accepts the explicitly named
  `visual_silhouette_geometry_proxy`, retaining the existing visual-proxy opt-in
  and separately supplied contact interaction. The old exposed-edge basis stays
  distinct. Unknown thermal/potential evidence and conservative aggregation are
  unchanged. Primary lambda remains 1.25; the waypoint nominal remains
  obstacle-unaware. No new wheel-motion experiment was performed.

## Stationary Isaac diagnostic, 2026-10-08

Three independent live Hawk RGB/Pandar LiDAR captures used session-layer blue
spheres, existing checkpoints and unchanged saved scene assets. Both supported
captures had 12 distinct scans, 0.10 s camera-LiDAR age, about 1.44 mm robot drift,
finite Safe-SAGE fields and accepted stationary MPC predictions. All physical
factor assessments were computed from captured inputs; no factor scores were
supplied manually.

For this size-ordering demonstration only, the same explicit density prior was
25 kg/m³ with interval [5, 100] and confidence 0.35. This is an illustrative
construction assumption, not measured density or a universal ball prior.

| Authored radius, evaluation only | Fitted radius and interval | Returns | Rough mass and interval, PRIOR | Visible contact proxy | Stationary result |
|---|---|---:|---|---:|---|
| 0.18 m | 0.170724 [0.146474, 0.194974] m | 194 | 0.521088 [0.065817, 3.104697] kg | 0.089968 | Pass |
| 0.25 m | 0.239773 [0.203854, 0.275692] m | 753 | 1.443543 [0.177426, 8.777299] kg | 0.103349 | Pass |
| 0.30 m | Not accepted through combined association | 1,307 | Unavailable in demo | Unavailable | Rejected: competing ball/mouse masks |

The larger accepted sphere yields greater estimated heaviness under the same
density assumption. The mass intervals overlap substantially, so this is nominal
relative ordering, not statistically established mass separation. Both authored
radii fall inside their inferred intervals; their nominal radius errors are
5.15% and 4.09%. Authored dimensions are evaluation labels only. RGB/visible
outline and LiDAR size are observed/inferred; robot pose/extrinsics are existing
simulation-oracle references, floor and nominal approach are configured priors,
and effective density is an explicit prior.

Kinetic pre-uplift risks increase from 0.001831 to 0.005187 under the unchanged
upper-bound adapter and 100 J normalization. Uncertainty raises these to
0.650641 and 0.701697; this improvement does not establish a low conservative
kinetic factor or a calibrated injury scale. Contact pre-uplift risks, using
proxy upper bounds and route interaction, are 0.179968 and 0.196701; after
uncertainty uplift they are 0.343975 and 0.357361. These are distinct from the
raw visual proxy values in the table.

Both aggregates remain **phi=1.0** because thermal prerequisites and justified
fall/interaction evidence are unavailable. Four-factor sensor readiness remains
false, and prior-dependent mass cannot pass a sensor-only claim. No object height
or fitted sphere center/radius is substituted for fall height. No new claim of
temperature, low thermal hazard, or complete contact physics is made.

The 0.30 m failure is preserved, not excluded silently: the segmenter returned
overlapping `sports_ball` (0.572 confidence) and `mouse` (0.554) instances.
Despite 95.4% target mask overlap, the existing ambiguity gate rejected the
association. No thresholds were relaxed. This exposes a real coverage limit
for subsequent multi-object perception work.

## Independent checks and artifacts

Independent synthetic masks test round, pointed, elongated and jagged ordering;
known 3D visible hemispheres test size recovery and eightfold mass scaling when
radius doubles at fixed density. Negative controls cover flat/narrow/non-sphere
surfaces, insufficient points, malformed data, incomplete masks, ambiguous
tracks, stale/future timestamps, provenance and adapter/readiness opt-in.

A fresh CPU replay of existing held-out PACO image `000000040248.jpg` retained
two accepted knife edge proxies, 0.273169 and 0.354419. Their whole silhouettes
were rejected by quality checks, and their positive part cues were preserved.
This is an independent image replay, not live Isaac evidence or a matched
cross-domain effectiveness study. It does not establish real nail/tip/jagged
coverage. No data, weights or dependencies were added or redistributed.

Final regression: **1,378 unit tests passed**, including 27 new shape/heaviness
cases; repository Ruff and whitespace checks passed. Stationary captures are
in `outputs/shape_heaviness/{small,medium,large}`. `summary.json` and
`stationary-comparison.png` contain the combined component report and images;
the original large failure and all camera/point/mask inputs remain intact.
The local `summarize.py` reconstructs the raw proxies from saved detector masks
and reruns the separate held-out replay. Reports from the initial captures
predate the added top-level raw-proxy field; reconstructed values and exact
original report hashes are retained in `summary.json`.

Run the existing stationary demo with its documented model environment and add
`--object-radius 0.18 --sphere-density-prior-kg-m3 25 5 100`, followed by a fresh
output directory and radius 0.25. Omit `--motion`. The radius argument authors
only the fixture/independent evaluation geometry. Omit the density option to
retain unavailable mass when the learned class model cannot predict.

Next: obtain supported thermal and fall/interaction evidence for a coherent
object comparison, and address detector ambiguity where needed. The accepted
gate remains different observed evidence -> different computed aggregate phi
-> matched waypoint/MPC/CBF control. This component result does not fulfill it.
