"""Validate per-track CRA integration in the live Safe-SAGE runtime.

This headless numerical demo creates two separated LiDAR clusters with the
same semantic label, ``object``.  A controlled oracle assigns an ordinary
chair profile to one track and an exposed-knife profile to the other.  The
real tracker, occupancy grid, configuration-space expansion, harmonic field,
and ``h_full`` pipeline then run without Isaac Sim, YOLO, MPC, or wheel motion.

The acceptance condition is deliberately controlled: semantic class and
geometry treatment are shared, while CRA produces different per-track risk
and normal boundary flux.  Default tangential strength remains fixed at the
repository paper-baseline value ``lambda = 1.25``.

All physical-factor values are illustrative software-validation hypotheses,
not calibrated research measurements.

Run from the project root:

    python -u scripts/demos/cra_safe_sage_runtime_demo.py
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot.risk.cra_oracle_scenarios import (
    exposed_knife_profile,
    ordinary_chair_profile,
)
from robot.safety.class_safety_policy import (
    ClassSafetyParameters,
    ClassSafetyPolicy,
)
from robot.safety.cra_safe_sage_policy import CRASafeSAGEPolicy
from robot.safety.cra_track_policy import TrackIdCRARiskProfileProvider
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.safety.poisson_safety_function import PoissonSafetyFunctionConfig
from robot.safety.safe_sage_configuration_space import (
    CircularRobotFootprint,
    SafeSAGEConfigurationSpaceConfig,
)
from robot.safety.safe_sage_field_pipeline import SafeSAGEFieldPipelineConfig
from robot.safety.safe_sage_guidance_policy import (
    PassingSide,
    SafeSAGEGuidanceParameters,
    SafeSAGEGuidancePolicy,
)
from robot.safety.safe_sage_h_full import SafeSAGEHFullConfig
from robot.safety.safe_sage_runtime import (
    SafeSAGEPerceptionObservation,
    SafeSAGEPerceptionRuntime,
)
from robot.safety.semantic_label_provider import (
    SemanticClassLabel,
    TrackIdSemanticLabelProvider,
)
from robot.safety.tangential_laplace_guidance_field import (
    TangentialLaplaceGuidanceFieldConfig,
)


def make_runtime() -> SafeSAGEPerceptionRuntime:
    """Return the controlled numerical Safe-SAGE runtime."""

    class_policy = ClassSafetyPolicy(
        {
            "unknown": ClassSafetyParameters(
                risk=1.0,
                boundary_flux_magnitude=1.7,
            ),
            "object": ClassSafetyParameters(
                risk=0.0,
                boundary_flux_magnitude=0.5,
            ),
        }
    )
    guidance_policy = SafeSAGEGuidancePolicy(
        {
            "unknown": SafeSAGEGuidanceParameters(
                safety_parameters=class_policy.parameters_for("unknown"),
                tangential_strength=1.25,
                passing_side=PassingSide.RIGHT,
            ),
            "object": SafeSAGEGuidanceParameters(
                safety_parameters=class_policy.parameters_for("object"),
                tangential_strength=1.25,
                passing_side=PassingSide.RIGHT,
            ),
        }
    )
    return SafeSAGEPerceptionRuntime(
        obstacle_tracker=ObstacleTracker(
            ObstacleTrackerConfig(
                cluster_radius=0.5,
                minimum_cluster_points=2,
                maximum_association_distance=0.8,
                velocity_smoothing_factor=1.0,
            )
        ),
        class_policy=class_policy,
        guidance_policy=guidance_policy,
        grid_config=LocalOccupancyGridConfig(
            resolution=1.0,
            minimum_x=-5.0,
            maximum_x=8.0,
            minimum_y=-7.0,
            maximum_y=7.0,
        ),
        field_config=SafeSAGEFieldPipelineConfig(
            configuration_space_config=SafeSAGEConfigurationSpaceConfig(
                footprint=CircularRobotFootprint(radius=0.5),
                yaw_slice_count=3,
            ),
            guidance_field_config=TangentialLaplaceGuidanceFieldConfig(
                convergence_tolerance=1e-6,
                maximum_iterations=10_000,
                residual_check_interval=5,
                tangential_buffer_radius=2.0,
                allow_dense_scene_arbitration=True,
            ),
            h_full_config=SafeSAGEHFullConfig(
                yaw_slice_count=3,
                poisson_config=PoissonSafetyFunctionConfig(
                    convergence_tolerance=1e-6,
                    maximum_iterations=10_000,
                    residual_check_interval=5,
                ),
            ),
        ),
        enable_temporal_updates=False,
        cra_policy=CRASafeSAGEPolicy(),
    )


def make_observation() -> SafeSAGEPerceptionObservation:
    """Return two equal-sized separated geometric clusters."""

    obstacle_points = np.array(
        [
            [1.9, -2.1],
            [1.9, -1.9],
            [2.1, -2.1],
            [2.1, -1.9],
            [1.9, 1.9],
            [1.9, 2.1],
            [2.1, 1.9],
            [2.1, 2.1],
        ],
        dtype=np.float64,
    )
    return SafeSAGEPerceptionObservation(
        sensor_frame_id=1,
        sensor_timestamp_seconds=1.0,
        simulation_time_seconds=1.0,
        obstacle_points_robot_xy=obstacle_points,
        robot_position_world_xy=np.zeros(2),
        robot_yaw=0.0,
    )


def main() -> None:
    """Run the controlled per-track CRA field construction."""

    label = SemanticClassLabel(
        "object",
        confidence=1.0,
        source="controlled_oracle",
    )
    labels = TrackIdSemanticLabelProvider({0: label, 1: label})
    profiles = TrackIdCRARiskProfileProvider(
        {
            0: ordinary_chair_profile(),
            1: exposed_knife_profile(),
        }
    )
    runtime = make_runtime()
    status = runtime.initialize_with_cra(
        make_observation(),
        labels,
        profiles,
    )
    field = status.snapshot.payload
    cra_result = field.cra_policy_result
    if cra_result is None:
        raise RuntimeError("The accepted field contains no CRA diagnostics.")

    print("\n" + "=" * 88)
    print("CRA SAFE-SAGE RUNTIME - PER-TRACK CONTROLLED REPORT")
    print("=" * 88)
    print(f"  Sensor frame:              {field.sensor_frame_id}")
    print(
        f"  Tracked obstacles:         {len(field.tracking_result.tracked_obstacles)}"
    )
    print(f"  Selected semantic tracks:  {field.selected_track_ids}")
    print(f"  CRA policy tracks:         {field.cra_track_ids}")
    print(f"  CRA fallback tracks:       {field.cra_fallback_track_ids}")
    print(f"  Field yaw slices:          {field.pipeline.yaw_slice_count}")
    print(
        f"  Tangential cells/slice:   {field.pipeline.tangential_interface_cell_counts}"
    )
    print("\n  track  semantic  profile            phi        b      lambda  dominant")
    print("  " + "-" * 82)
    profile_names = {0: "ordinary chair", 1: "exposed knife"}
    for assignment in cra_result.assignments:
        policy = assignment.policy_result
        dominant = (
            "none"
            if policy.assessment.dominant_factor is None
            else policy.assessment.dominant_factor.value
        )
        print(
            f"  {assignment.track_id:>5}  "
            f"{assignment.obstacle.semantic_class:<8}  "
            f"{profile_names[assignment.track_id]:<18} "
            f"{policy.risk:>7.4f}  "
            f"{policy.boundary_flux:>7.4f}  "
            f"{policy.tangential_strength:>8.4f}  "
            f"{dominant}"
        )

    low = cra_result.assignment_for(0)
    high = cra_result.assignment_for(1)
    grid = field.pipeline.configuration_space.occupancy_grids[0]
    grid_metadata_valid = True
    for assignment in cra_result.assignments:
        track_mask = grid.occupied_mask & (grid.cell_track_ids == assignment.track_id)
        grid_metadata_valid &= bool(np.any(track_mask))
        grid_metadata_valid &= bool(
            np.allclose(
                grid.cell_risks[track_mask],
                assignment.policy_result.risk,
                atol=1e-12,
                rtol=0.0,
            )
        )
        grid_metadata_valid &= bool(
            np.allclose(
                grid.cell_boundary_fluxes[track_mask],
                assignment.policy_result.boundary_flux,
                atol=1e-12,
                rtol=0.0,
            )
        )
    same_semantic_class = low.obstacle.semantic_class == high.obstacle.semantic_class
    stronger_contact_response = (
        high.policy_result.risk > low.policy_result.risk
        and abs(high.policy_result.boundary_flux) > abs(low.policy_result.boundary_flux)
    )
    lambda_held_fixed = np.isclose(
        low.policy_result.tangential_strength,
        high.policy_result.tangential_strength,
        atol=1e-12,
        rtol=0.0,
    )
    no_fallback = field.cra_fallback_track_ids == ()

    print("\n" + "=" * 88)
    print("CONTROLLED RUNTIME CHECKS")
    print("=" * 88)
    print(f"  Both tracks share semantic class:     {same_semantic_class}")
    print(f"  Higher CRA risk strengthens |b|:      {stronger_contact_response}")
    print(f"  Baseline lambda remains fixed:        {lambda_held_fixed}")
    print(f"  Grid preserves per-track parameters:  {grid_metadata_valid}")
    print(f"  No fallback profile was required:     {no_fallback}")
    passed = (
        same_semantic_class
        and stronger_contact_response
        and lambda_held_fixed
        and grid_metadata_valid
        and no_fallback
    )
    if passed:
        print(
            "\n  [PASS] The live field used distinct CRA parameters for "
            "same-class geometric tracks."
        )
    else:
        print("\n  [WARNING] A controlled CRA runtime check failed.")
    print("  No MPC, CBF, or wheel command was executed.")
    print("=" * 88)


if __name__ == "__main__":
    main()
