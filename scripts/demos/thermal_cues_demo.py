"""Stationary synthetic temporal thermal-cue diagnostic."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.physical_attributes import EstimateStatus, ScalarEstimate
from robot.perception.thermal_cues import ThermalCueFrame, estimate_thermal_cues


def cue(value: float, name: str) -> ScalarEstimate:
    return ScalarEstimate(
        value=value,
        lower_bound=max(0.0, value - 0.1),
        upper_bound=min(1.0, value + 0.1),
        confidence=0.8,
        unit="1",
        source=f"stationary_{name}_fixture",
        timestamp_seconds=1.5,
        status=EstimateStatus.INFERRED,
        evidence=("stationary synthetic visual evidence",),
    )


def main():
    frames = tuple(
        ThermalCueFrame(t, None, 0.85, "synthetic_steam_fixture")
        for t in (1.0, 1.2, 1.4)
    )
    result = estimate_thermal_cues(
        frames,
        timestamp_seconds=1.5,
        open_container_likelihood=cue(0.75, "open_container"),
        plume_source_association_likelihood=cue(0.65, "plume_association"),
    )
    print("STATIONARY THERMAL DIAGNOSTIC: no camera model or robot motion")
    print(
        f"steam={result.steam_likelihood.value:.2f}; "
        f"container={result.open_container_likelihood.value:.2f}; "
        f"association=0.65"
    )
    print(
        f"continuous_heat_proxy={result.heat_source_likelihood.value:.2f}; "
        f"type={result.source_type.value}"
    )
    print("temperature/spill/route interaction=unavailable")
    assert result.heat_source_likelihood.value == 0.65
    assert not result.spill_likelihood.is_available
    print("PASS: contextual visual evidence stayed continuous and non-thermal")


if __name__ == "__main__":
    main()
