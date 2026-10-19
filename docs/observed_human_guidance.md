# Observed-human passing qualification and runtime diagnostics

The preceding 8 s component arrived/stopped and passed human perception/motion
checks, but its passing selector later accepted a track that failed independent
human matching (.798841 m maximum centroid error). Geometric centroid motion
alone cannot establish that a track is the intended observed person. The
original failed bypass remains recorded in `docs/positive_fragment_continuity.md`.

`--human-observed-passing-target` is a separate bare-human opt-in. A **new**
passing selection requires the existing fresh, same-frame, unambiguous
YOLO-person/SAM association. The association that already selects the declared
adult mass prior supplies this class cue; mass magnitude is not an eligibility
test or a measured weight. Retained mass, class text alone, stale/missing cues
and unavailable estimates cannot select a new target. Current camera frame ID
and read-time timestamp are recorded separately from the LiDAR policy epoch.

The original eight-second same-track policy latch may continue after that
visual cue leaves view, keeping its original selection time. It retains no
physical evidence or current class claim. All original geometric ambiguity,
source-gap, identity, point-count and encounter-release checks remain active;
an unknown competing crossing can still block/release the preference. Every
other obstacle keeps its occupancy, numeric CRA and online safety intervention.
Both new flags default off, retaining the existing comparison recipes.

`--profile-runtime` instruments individual worker calls and saves `.pstats`
only for calls longer than three wall seconds. It also records the actual loaded
NumPy/SciPy/OSQP/Torch paths and versions and, if already installed, read-only
thread-pool metadata. It installs no packages and changes no thread limits.
A library/environment manifest is saved before motion even without profiling.
Profiling can affect timing and is diagnostic, not native performance acceptance.
The first profile run preserves the old selection mode and primary 8 s fixture
but explicitly ends at 11.5 navigation seconds; missing arrival is expected in
that diagnostic and must not be relabeled successful navigation.

The numerical audit must precede further tuning. In particular, Safe-SAGE v3
explicitly forward-propagates its scaled temporal estimate across the MPC
horizon ([sections III-E/F](https://arxiv.org/html/2603.05497v3#S3.SS6)); our
current implementation also does so. Whether noisy live field changes make
that forecast infeasible is a hypothesis to diagnose, not authorization to
silently cap it, relax exact checks or add another objective. Keep the nominal
obstacle-unaware, lambda 1.25, all evidence clocks and prior policies unchanged.

The optional person gate passed focused selector, actual all-object-field and
provenance tests, including rejection of retained-only mass and preservation of
unverified-object fields. Native selection/bypass acceptance remains pending.
The full unit suite passed 2,100 tests; repository Ruff passed. The optional
thread-pool inspector is absent in the project environment, so that metadata
stays explicitly unavailable. No package was installed; loaded library paths,
build configuration, logical CPU count and selected thread-count environment
variables remain available for the native diagnostic.
