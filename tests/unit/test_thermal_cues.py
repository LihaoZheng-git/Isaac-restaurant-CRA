import pytest

from robot.perception.thermal_cues import (
    ThermalCueConfig,
    ThermalCueFrame,
    estimate_thermal_cues,
)


def frames(flame=0.8, steam=0.1):
    return tuple(ThermalCueFrame(t, flame, steam, "fixture") for t in (1.0, 1.2, 1.4))


def test_persistent_flame_is_inferred_but_not_temperature():
    result = estimate_thermal_cues(frames(), timestamp_seconds=1.5)
    assert result.heat_source_likelihood.value == 0.8
    assert result.source_type.value == "visible_flame"
    assert "not a temperature" in result.heat_source_likelihood.evidence[-1]
    assert not result.open_container_likelihood.is_available


def test_persistent_steam_stays_separate_from_heat_source():
    result = estimate_thermal_cues(frames(flame=0.1, steam=0.9), timestamp_seconds=1.5)
    assert result.source_type.value == "visible_steam"
    assert result.heat_source_likelihood.value == 0.1
    assert result.steam_likelihood.value == 0.9


def test_unsupported_steam_is_unavailable_instead_of_zero():
    items = tuple(
        ThermalCueFrame(t, 0.8, None, "fire_smoke_only", smoke_likelihood=0.9)
        for t in (1.0, 1.2, 1.4)
    )
    result = estimate_thermal_cues(items, timestamp_seconds=1.5)
    assert result.heat_source_likelihood.value == 0.8
    assert not result.steam_likelihood.is_available
    assert result.source_type.value == "visible_flame"


def test_too_few_or_stale_frames_are_unavailable():
    assert not estimate_thermal_cues(
        frames()[:2], timestamp_seconds=1.5
    ).steam_likelihood.is_available
    assert not estimate_thermal_cues(
        frames(), timestamp_seconds=4
    ).steam_likelihood.is_available


def test_temporal_median_rejects_one_score_spike():
    items = (
        ThermalCueFrame(1, 0, 0, "fixture"),
        ThermalCueFrame(1.1, 1, 0, "fixture"),
        ThermalCueFrame(1.2, 0, 0, "fixture"),
    )
    assert (
        estimate_thermal_cues(items, timestamp_seconds=1.2).heat_source_likelihood.value
        == 0
    )


@pytest.mark.parametrize("score", [-1, 2, float("nan")])
def test_invalid_scores_rejected(score):
    with pytest.raises(ValueError):
        ThermalCueFrame(1, score, 0, "fixture")


def test_future_frame_and_bad_config_rejected():
    with pytest.raises(ValueError):
        estimate_thermal_cues(frames(), timestamp_seconds=0)
    with pytest.raises(ValueError):
        estimate_thermal_cues(
            frames(),
            timestamp_seconds=2,
            config=ThermalCueConfig(minimum_frame_count=0),
        )
