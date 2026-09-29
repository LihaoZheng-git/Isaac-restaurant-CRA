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

The validated stationary run associated 15 live LiDAR returns with the target,
inferred a 0.726 m support-to-landing drop against the separately authored
0.800 m oracle, and covered the oracle within its uncertainty interval. Absolute
error was 0.074 m. Robot drift was 0.0012 m and 0.0127 degrees under zero wheel
commands. This is one controlled positive sensor diagnostic, not a general
fall-height accuracy result.
