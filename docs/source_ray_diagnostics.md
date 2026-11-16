# Exact source packets at first track disappearance

The surface-enabled retry stopped on a 15.6 s-old unresolved fragment near the
human's initial position. Current positive returns were .712571 m away; there
was no measured overlap. The previous diagnostics could not distinguish a
vacated region, occlusion or a LiDAR blind region. See the failed result in
[positive-surface continuity](positive_surface_continuity.md).

With the existing `--save-sensor-snapshots` option, the bare-human live caller
now archives the immediately preceding accepted source and the newly processed
source whenever an enrolled current track first disappears. Each pair keeps:

- Original frame/header epochs, raw sensor points and per-return offsets.
- Original scene/obstacle points, exact point ownership and RGB frames.
- Sensor extrinsics, source-frame transforms and the recorded ego poses bracketing
  the scan's return epochs; compensation metadata remains explicit.
- Old/current tracking summaries and applied policies, before the guard decision.

The bounded in-memory diagnostic cache keeps four recent packets and protects
the last accepted source. Missing raw archives are labeled unavailable. All
timestamps remain original; neither stored points nor these copies enter control.
Per-return ego poses/extrinsics remain oracle references. GMO start/end metadata
must not replace full-scan pose history. No moving-actor deskew or independent
RGB acquisition clock is introduced.

Surface rejections are now retained even before a stop, so the original samples
are not lost when the .35 s association history expires. A stop still saves its
unpublished current source and preceding accepted source. File names carry both
source frame IDs. This changes diagnostic capture only: **no free-space clearing,
timeout deletion, additional occupancy, factor update or relaxed guard exists**.

## Next same-fixture check

Keep the same 20 s crossing and surface option, with the optional concurrent
`--profile-runtime` instrumentation off. Stage/worker wall timers remain. The
installed cProfile has verified cross-thread attribution/overhead limitations;
this is an instrumentation sensitivity check, not a causal controller comparison.
The own 60-frame stationary gate, lambda 1.25, actuator limits, original 90 s
bound/.025 pacing and all clearance, freshness, exact safety, arrival/arc/behind
thresholds remain. Preserve whichever outcome occurs before choosing any new
visibility or association method. Source copies alone are not evidence that an
old location is free or that a disappeared object is harmless.

Validation: 91 focused inference, continuity, profiling, deskew and epoch tests
passed in 5.11 s. Repository Ruff/whitespace checks pass. The unchanged core
surface/continuity source previously passed all 2,310 unit tests at f087533;
the new diagnostic exports still require native acquisition validation.
