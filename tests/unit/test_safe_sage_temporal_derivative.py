"""Unit tests for Safe-SAGE motion-compensated temporal differences."""

from dataclasses import replace

import numpy as np
import pytest

from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.safe_sage_h_full import SafeSAGEHFullResult
from robot.safety.safe_sage_temporal_derivative import (
    SafeSAGETemporalDerivativeConfig,
    SafeSAGETemporalDerivativeEstimator,
    SafeSAGETemporalDerivativeResult,
    sample_temporal_safety_derivative,
    validate_temporal_safety_derivative,
)


def make_grid_config() -> LocalOccupancyGridConfig:
    """Return a 4-by-5 cell-center grid."""

    return LocalOccupancyGridConfig(
        resolution=1.0,
        minimum_x=-2.5,
        maximum_x=2.5,
        minimum_y=-2.0,
        maximum_y=2.0,
    )


def make_h_full(
    values: np.ndarray,
    *,
    yaw_angles: np.ndarray | None = None,
) -> SafeSAGEHFullResult:
    """Build one minimal aligned h_full result."""

    array = np.asarray(values, dtype=np.float64)
    if yaw_angles is None:
        yaw_angles = np.linspace(
            -np.pi,
            np.pi,
            num=array.shape[0],
            endpoint=False,
            dtype=np.float64,
        )
    return SafeSAGEHFullResult(
        yaw_angles_radians=np.asarray(yaw_angles, dtype=np.float64),
        values=array,
        spatial_gradients_robot_xy=np.zeros(
            array.shape + (2,),
            dtype=np.float64,
        ),
        yaw_derivatives=np.zeros_like(array),
        dirichlet_boundary_masks=np.zeros_like(array, dtype=bool),
        guidance_divergence=np.zeros_like(array),
        forcing_fields=-np.ones_like(array),
        iteration_counts=np.ones(array.shape[0], dtype=np.int64),
        maximum_residuals=np.zeros(array.shape[0], dtype=np.float64),
        resolution=1.0,
        softplus_beta=1.0,
        minimum_forcing_magnitude=1e-9,
    )


def make_constant_h_full(value: float) -> SafeSAGEHFullResult:
    """Return four constant yaw slices."""

    return make_h_full(np.full((4, 4, 5), value, dtype=np.float64))


def estimate(
    previous: SafeSAGEHFullResult,
    current: SafeSAGEHFullResult,
    *,
    time_delta: float = 0.2,
    egomotion: tuple[float, float] = (0.0, 0.0),
    previous_estimate: SafeSAGETemporalDerivativeResult | None = None,
    smoothing_factor: float = 0.35,
) -> SafeSAGETemporalDerivativeResult:
    """Estimate with deterministic test limits."""

    estimator = SafeSAGETemporalDerivativeEstimator(
        SafeSAGETemporalDerivativeConfig(
            smoothing_factor=smoothing_factor,
            minimum_time_delta_seconds=0.01,
            maximum_time_delta_seconds=0.5,
        )
    )
    return estimator.estimate(
        previous_h_full=previous,
        current_h_full=current,
        grid_config=make_grid_config(),
        time_delta_seconds=time_delta,
        egomotion_translation_grid_xy=egomotion,
        previous_estimate=previous_estimate,
    )


def test_identical_fixed_frame_fields_have_zero_derivative() -> None:
    field = make_constant_h_full(2.0)

    result = estimate(field, field)

    assert np.all(result.values == 0.0)
    assert np.all(result.raw_values == 0.0)
    assert result.valid_fraction == pytest.approx(1.0)


def test_finite_difference_preserves_closing_sign() -> None:
    previous = make_constant_h_full(1.0)
    current = make_constant_h_full(0.8)

    result = estimate(previous, current, time_delta=0.2)

    assert result.raw_values == pytest.approx(-1.0)
    assert result.values == pytest.approx(-0.35)
    assert result.maximum_absolute_derivative == pytest.approx(0.35)


def test_first_order_low_pass_uses_previous_filtered_estimate() -> None:
    first = estimate(
        make_constant_h_full(1.0),
        make_constant_h_full(0.8),
        smoothing_factor=0.25,
    )

    second = estimate(
        make_constant_h_full(0.8),
        make_constant_h_full(0.8),
        previous_estimate=first,
        smoothing_factor=0.25,
    )

    assert second.raw_values == pytest.approx(0.0)
    assert second.values == pytest.approx(-0.1875)


def test_integer_egomotion_alignment_removes_apparent_change() -> None:
    columns = np.arange(5, dtype=np.float64)
    previous_values = np.broadcast_to(columns, (4, 4, 5)).copy()
    current_values = np.broadcast_to(columns + 1.0, (4, 4, 5)).copy()

    uncompensated = estimate(
        make_h_full(previous_values),
        make_h_full(current_values),
        smoothing_factor=1.0,
    )
    compensated = estimate(
        make_h_full(previous_values),
        make_h_full(current_values),
        egomotion=(1.0, 0.0),
        smoothing_factor=1.0,
    )

    assert np.all(uncompensated.values == pytest.approx(5.0))
    assert np.all(compensated.values[compensated.valid_mask] == 0.0)
    assert compensated.valid_fraction == pytest.approx(0.8)


def test_subcell_egomotion_uses_bilinear_alignment() -> None:
    columns = np.arange(5, dtype=np.float64)
    previous_values = np.broadcast_to(columns, (4, 4, 5)).copy()
    current_values = np.broadcast_to(columns + 0.5, (4, 4, 5)).copy()

    result = estimate(
        make_h_full(previous_values),
        make_h_full(current_values),
        egomotion=(0.5, 0.0),
        smoothing_factor=1.0,
    )

    assert np.all(result.values[result.valid_mask] == pytest.approx(0.0))
    assert np.all(result.values[~result.valid_mask] == 0.0)


def test_temporal_result_samples_space_and_periodic_yaw() -> None:
    values = np.zeros((4, 4, 5), dtype=np.float64)
    values[0] = 0.0
    values[1] = 2.0
    values[2] = 4.0
    values[3] = 6.0
    result = estimate(
        make_constant_h_full(0.0),
        make_h_full(values),
        time_delta=0.2,
        smoothing_factor=1.0,
    )

    sampled = result.sample(
        np.array([0.0, 0.0]),
        -0.75 * np.pi,
    )

    assert sampled == pytest.approx(5.0)


def test_sampling_helper_preserves_legacy_scalar() -> None:
    h_full = make_constant_h_full(1.0)

    sampled = sample_temporal_safety_derivative(
        -0.25,
        query_position_grid_xy=np.array([0.0, 0.0]),
        query_yaw_radians=0.0,
        h_full=h_full,
        grid_config=make_grid_config(),
    )

    assert sampled == pytest.approx(-0.25)


def test_sampling_helper_uses_spatial_temporal_field() -> None:
    previous = make_constant_h_full(1.0)
    current_values = np.ones((4, 4, 5), dtype=np.float64)
    current_values[:, :, 2] = 0.8
    result = estimate(
        previous,
        make_h_full(current_values),
        smoothing_factor=1.0,
    )

    sampled = sample_temporal_safety_derivative(
        result,
        query_position_grid_xy=np.array([0.0, 0.0]),
        query_yaw_radians=0.0,
        h_full=make_h_full(current_values),
        grid_config=make_grid_config(),
    )

    assert sampled == pytest.approx(-1.0)


@pytest.mark.parametrize("time_delta", [0.001, 0.6])
def test_estimator_rejects_time_delta_outside_configured_bounds(
    time_delta: float,
) -> None:
    with pytest.raises(ValueError, match="time_delta_seconds"):
        estimate(
            make_constant_h_full(1.0),
            make_constant_h_full(1.0),
            time_delta=time_delta,
        )


def test_estimator_rejects_mismatched_yaw_angles() -> None:
    previous = make_constant_h_full(1.0)
    current = make_h_full(
        np.ones((4, 4, 5), dtype=np.float64),
        yaw_angles=np.array([-3.0, -1.0, 1.0, 3.0]),
    )

    with pytest.raises(ValueError, match="yaw angles"):
        estimate(previous, current)


def test_validate_rejects_temporal_field_from_another_grid() -> None:
    h_full = make_constant_h_full(1.0)
    result = estimate(h_full, h_full)
    other_grid = replace(make_grid_config(), minimum_x=-1.5, maximum_x=3.5)

    with pytest.raises(ValueError, match="grid_config"):
        validate_temporal_safety_derivative(
            result,
            h_full=h_full,
            grid_config=other_grid,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"smoothing_factor": 0.0},
        {"smoothing_factor": 1.1},
        {"minimum_time_delta_seconds": 0.0},
        {
            "minimum_time_delta_seconds": 0.2,
            "maximum_time_delta_seconds": 0.1,
        },
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SafeSAGETemporalDerivativeConfig(**overrides)


def test_result_arrays_are_read_only() -> None:
    result = estimate(
        make_constant_h_full(1.0),
        make_constant_h_full(0.8),
    )

    with pytest.raises(ValueError):
        result.values[0, 0, 0] = 3.0
