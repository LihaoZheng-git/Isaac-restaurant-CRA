"""Stationary synthetic temporal thermal-cue diagnostic."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.thermal_cues import ThermalCueFrame, estimate_thermal_cues


def main():
    frames = tuple(
        ThermalCueFrame(t, 0.05, 0.85, "synthetic_steam_fixture")
        for t in (1.0, 1.2, 1.4)
    )
    result = estimate_thermal_cues(frames, timestamp_seconds=1.5)
    print("STATIONARY THERMAL DIAGNOSTIC: no camera model or robot motion")
    print(f"steam={result.steam_likelihood.value:.2f}; type={result.source_type.value}")
    print("temperature=unavailable; container/spill state=unavailable")
    assert not result.open_container_likelihood.is_available
    print("PASS: temporal visual cue stayed separate from temperature")


if __name__ == "__main__":
    main()
