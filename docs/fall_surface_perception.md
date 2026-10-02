# LiDAR support and landing surfaces

`estimate_lidar_fall_surfaces` estimates fall height only from local surface
geometry. It removes the tracked object's own returns, localizes a support
height band adjacent to the object, and searches radial landing corridors. A
positive drop requires a lower landing band in a direction where the
support-height band ceases. Opposing equally supported directions are rejected
as ambiguous. Object height and center height never enter this calculation.

All geometric thresholds are exposed through `LidarFallSurfaceConfig`. The
defaults require at least six returns per surface band, no more than 0.06 m
between the lowest visible object returns and support, planar support extending
at least 0.10 m beyond the visible object footprint, and a drop of at least
0.08 m. Returned surface heights are inferred evidence with explicit bounds;
they are not direct contact measurements. Harmful-fall interaction remains a
separate input to potential CRA.

Synthetic stationary tests cover a table edge above a floor, an object on a
continuous floor, a table with no visible lower landing surface, missing
support, and ambiguous opposing drops. In the existing Isaac restaurant scene,
an early permissive version falsely inferred a 0.173 m chair drop. The planar
support-overhang gate now rejects that chair because its sparse lower returns
do not establish a supporting plane beyond the object footprint. This negative
control passes while leaving fall height unavailable. A deliberately elevated
Isaac object with visible support and landing geometry is still required before
claiming live positive fall-height accuracy.

The controlled positive Isaac diagnostic in
`scripts/demos/lidar_fall_height_isaac_diagnostic.py` authors a target on a
0.80 m support but supplies only live PandarXT-32 points to the estimator.
Authored heights are read afterward for offline scoring. Because the 32-beam
scan's lowest visible target return is about 0.17 m above its support, this
fixture declares an experimental 0.20 m support gap and 0.30 m local search
margin. The stricter runtime defaults remain unchanged. This configuration
requires additional negative-scene calibration before runtime adoption.

The earlier run reported a 0.726 m drop against a declared 0.800 m oracle.
The 2026-10-02 stationary benchmark found that the fixture scaled a unit USD
cube by half its declared dimensions. Its rendered support and target therefore
did not match the experiment's geometry contract. That result is superseded
and must not be cited as valid positive fall-height accuracy.

The diagnostic now uses full metric dimensions, checks rendered support top,
target bottom and target height, and authors only in the session layer. With
the corrected 0.80 m support, the live scan's object-adjacent band was too far
below the target. Fall height remained unavailable. A separately labeled
0.55 m visibility diagnostic also remained unavailable because the selected
band lacked planar support beyond the object's footprint. Both robots stayed
within 0.005 m / 0.1 degree stationary tolerances. No runtime thresholds were
loosened to pass these cases.

`CRA_FALL_SUPPORT_DROP_M` configures the diagnostic support height; the default
remains 0.80 m. Target association and the height filter use authored fixture
setup. The surface estimator receives live points, and authored heights are
used separately for scoring. Positive fall-height sensor validation remains
pending; support-plane visibility and band selection need investigation before
using this path in a motion experiment.
