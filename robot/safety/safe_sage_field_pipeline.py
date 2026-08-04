"""End-to-end semantic field construction for the Safe-SAGE controller.

This module joins the reusable field-building stages without introducing
Isaac Sim dependencies:

1. Expand one raw obstacle grid into yaw-dependent configuration spaces.
2. Solve the semantic normal-and-tangential Laplace field at every yaw.
3. Convert guidance divergence into spatially varying negative forcing.
4. Solve and stack the periodic ``h_full(x, y, psi)`` safety field.

All intermediate products remain available in the returned result so demos
and experiments can diagnose geometry, semantic guidance, and Poisson
convergence separately.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from robot.safety.local_occupancy_grid import LocalOccupancyGrid
from robot.safety.safe_sage_configuration_space import (
    SafeSAGEConfigurationSpaceConfig,
    SafeSAGEConfigurationSpaceResult,
    build_safe_sage_configuration_space,
)
from robot.safety.safe_sage_guidance_policy import (
    SafeSAGEGuidancePolicy,
)
from robot.safety.safe_sage_h_full import (
    SafeSAGEHFullConfig,
    SafeSAGEHFullResult,
    solve_safe_sage_h_full,
)
from robot.safety.semantic_obstacle import SemanticObstacle
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
    TangentialLaplaceGuidanceFieldResult,
    solve_tangential_laplace_guidance_field,
)


class SafeSAGEFieldPipelineError(RuntimeError):
    """Report the field-construction stage and yaw slice that failed."""


@dataclass(frozen=True)
class SafeSAGEFieldPipelineConfig:
    """Configuration-space, guidance, and safety-function settings."""

    configuration_space_config: SafeSAGEConfigurationSpaceConfig
    guidance_field_config: TangentialLaplaceGuidanceFieldConfig = field(
        default_factory=TangentialLaplaceGuidanceFieldConfig
    )
    h_full_config: SafeSAGEHFullConfig = field(
        default_factory=SafeSAGEHFullConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.configuration_space_config,
            SafeSAGEConfigurationSpaceConfig,
        ):
            raise TypeError(
                "configuration_space_config must be a "
                "SafeSAGEConfigurationSpaceConfig."
            )
        if not isinstance(
            self.guidance_field_config,
            TangentialLaplaceGuidanceFieldConfig,
        ):
            raise TypeError(
                "guidance_field_config must be a "
                "TangentialLaplaceGuidanceFieldConfig."
            )
        if not isinstance(self.h_full_config, SafeSAGEHFullConfig):
            raise TypeError(
                "h_full_config must be a SafeSAGEHFullConfig."
            )

        configuration_count = (
            self.configuration_space_config.yaw_slice_count
        )
        if self.h_full_config.yaw_slice_count != configuration_count:
            raise ValueError(
                "configuration-space and h_full yaw_slice_count values "
                "must match."
            )


@dataclass(frozen=True)
class SafeSAGEFieldPipelineResult:
    """Complete field plus yaw-aligned intermediate results."""

    configuration_space: SafeSAGEConfigurationSpaceResult
    guidance_fields: tuple[
        TangentialLaplaceGuidanceFieldResult, ...
    ]
    h_full: SafeSAGEHFullResult
    semantic_obstacle_count: int

    @property
    def yaw_slice_count(self) -> int:
        """Return the common yaw-slice count."""

        return self.h_full.yaw_slice_count

    @property
    def yaw_angles_radians(self) -> np.ndarray:
        """Return the canonical read-only yaw angles."""

        return self.h_full.yaw_angles_radians

    @property
    def guidance_iteration_counts(self) -> tuple[int, ...]:
        """Return harmonic-solver iterations for each yaw slice."""

        return tuple(
            field_result.iteration_count
            for field_result in self.guidance_fields
        )

    @property
    def tangential_interface_cell_counts(self) -> tuple[int, ...]:
        """Return tangential-interface sizes for each yaw slice."""

        return tuple(
            field_result.tangential_interface_cell_count
            for field_result in self.guidance_fields
        )


def build_safe_sage_field_pipeline(
    source_grid: LocalOccupancyGrid,
    *,
    guidance_policy: SafeSAGEGuidancePolicy,
    config: SafeSAGEFieldPipelineConfig,
    semantic_obstacles: Iterable[SemanticObstacle] = (),
) -> SafeSAGEFieldPipelineResult:
    """Construct all yaw-dependent Safe-SAGE fields in one checked call."""

    if not isinstance(source_grid, LocalOccupancyGrid):
        raise TypeError("source_grid must be a LocalOccupancyGrid.")
    if not isinstance(guidance_policy, SafeSAGEGuidancePolicy):
        raise TypeError(
            "guidance_policy must be a SafeSAGEGuidancePolicy."
        )
    if not isinstance(config, SafeSAGEFieldPipelineConfig):
        raise TypeError(
            "config must be a SafeSAGEFieldPipelineConfig."
        )

    semantic_sequence = _validated_semantic_obstacles(
        semantic_obstacles
    )
    _validate_source_semantic_metadata(
        source_grid,
        guidance_policy=guidance_policy,
        semantic_obstacles=semantic_sequence,
    )

    configuration_space = build_safe_sage_configuration_space(
        source_grid,
        config.configuration_space_config,
    )
    guidance_fields = tuple(
        _solve_guidance_slice(
            occupancy_grid,
            yaw_index=yaw_index,
            yaw_radians=float(
                configuration_space.yaw_angles_radians[yaw_index]
            ),
            guidance_policy=guidance_policy,
            semantic_obstacles=semantic_sequence,
            config=config.guidance_field_config,
        )
        for yaw_index, occupancy_grid in enumerate(
            configuration_space.occupancy_grids
        )
    )

    try:
        h_full = solve_safe_sage_h_full(
            configuration_space.occupancy_grids,
            guidance_fields,
            config.h_full_config,
        )
    except (ValueError, RuntimeError) as error:
        raise SafeSAGEFieldPipelineError(
            "Safe-SAGE h_full construction failed after all guidance "
            f"fields were solved: {error}"
        ) from error

    if not np.array_equal(
        h_full.yaw_angles_radians,
        configuration_space.yaw_angles_radians,
    ):
        raise RuntimeError(
            "Internal yaw ordering differs between configuration space "
            "and h_full."
        )

    return SafeSAGEFieldPipelineResult(
        configuration_space=configuration_space,
        guidance_fields=guidance_fields,
        h_full=h_full,
        semantic_obstacle_count=len(semantic_sequence),
    )


def _solve_guidance_slice(
    occupancy_grid: LocalOccupancyGrid,
    *,
    yaw_index: int,
    yaw_radians: float,
    guidance_policy: SafeSAGEGuidancePolicy,
    semantic_obstacles: tuple[SemanticObstacle, ...],
    config: TangentialLaplaceGuidanceFieldConfig,
) -> TangentialLaplaceGuidanceFieldResult:
    """Solve one yaw slice and attach useful context to failures."""

    try:
        return solve_tangential_laplace_guidance_field(
            occupancy_grid,
            guidance_policy=guidance_policy,
            semantic_obstacles=semantic_obstacles,
            config=config,
        )
    except (ValueError, RuntimeError) as error:
        raise SafeSAGEFieldPipelineError(
            "Semantic guidance solve failed at yaw slice "
            f"{yaw_index} ({math.degrees(yaw_radians):.2f} degrees): "
            f"{error}"
        ) from error


def _validated_semantic_obstacles(
    semantic_obstacles: Iterable[SemanticObstacle],
) -> tuple[SemanticObstacle, ...]:
    """Materialize semantic obstacles and reject invalid track mappings."""

    try:
        obstacles = tuple(semantic_obstacles)
    except TypeError as error:
        raise TypeError(
            "semantic_obstacles must be an iterable."
        ) from error

    seen_track_ids: set[int] = set()
    for obstacle in obstacles:
        if not isinstance(obstacle, SemanticObstacle):
            raise TypeError(
                "Every semantic obstacle must be a SemanticObstacle."
            )
        if obstacle.track_id in seen_track_ids:
            raise ValueError(
                "semantic_obstacles contains duplicate track ID "
                f"{obstacle.track_id}."
            )
        seen_track_ids.add(obstacle.track_id)
    return obstacles


def _validate_source_semantic_metadata(
    source_grid: LocalOccupancyGrid,
    *,
    guidance_policy: SafeSAGEGuidancePolicy,
    semantic_obstacles: tuple[SemanticObstacle, ...],
) -> None:
    """Require grid metadata and guidance policy to describe one policy."""

    semantic_by_track_id = {
        obstacle.track_id: obstacle
        for obstacle in semantic_obstacles
    }
    occupied_track_ids = np.unique(
        source_grid.cell_track_ids[source_grid.occupied_mask]
    )

    for raw_track_id in occupied_track_ids:
        track_id = int(raw_track_id)
        semantic_obstacle = semantic_by_track_id.get(track_id)
        parameters = (
            guidance_policy.parameters_for_obstacle(
                semantic_obstacle
            )
            if semantic_obstacle is not None
            else guidance_policy.parameters_for(
                guidance_policy.unknown_class
            )
        )
        track_mask = (
            source_grid.occupied_mask
            & (source_grid.cell_track_ids == track_id)
        )
        risks = source_grid.cell_risks[track_mask]
        boundary_fluxes = source_grid.cell_boundary_fluxes[track_mask]

        if not np.allclose(
            risks,
            parameters.risk,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Source-grid risk metadata disagrees with the Safe-SAGE "
                f"guidance policy for track ID {track_id}."
            )
        if not np.allclose(
            boundary_fluxes,
            parameters.boundary_flux,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Source-grid boundary-flux metadata disagrees with the "
                f"Safe-SAGE guidance policy for track ID {track_id}."
            )
