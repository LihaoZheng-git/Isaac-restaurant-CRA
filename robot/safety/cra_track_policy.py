"""Associate contextual-risk profiles with existing semantic LiDAR tracks.

Geometry and identity remain owned by LiDAR clustering and tracking.  This
module only attaches a ``PhysicalRiskProfile`` to each already labeled track,
evaluates the CRA-to-Safe-SAGE policy, and returns immutable per-track
parameters.  A provider that has no profile for a selected track resolves to
``PhysicalRiskProfile.unknown()`` and therefore receives the conservative CRA
fallback.

The provider protocol is intentionally independent of cameras.  Controlled
tests use a track-ID mapping; a later vision module can implement the same
``profile_for`` method without changing field construction or control.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from robot.risk.cra import PhysicalRiskProfile
from robot.safety.class_safety_policy import ClassSafetyParameters
from robot.safety.cra_safe_sage_policy import (
    CRASafeSAGEPolicy,
    CRASafeSAGEPolicyResult,
)
from robot.safety.safe_sage_guidance_policy import SafeSAGEGuidanceParameters
from robot.safety.semantic_obstacle import SemanticObstacle


@runtime_checkable
class CRARiskProfileProvider(Protocol):
    """Return contextual physical-risk data for an existing semantic track."""

    def profile_for(
        self,
        obstacle: SemanticObstacle,
    ) -> PhysicalRiskProfile | None:
        """Return a profile, or ``None`` when contextual risk is unavailable."""


class TrackIdCRARiskProfileProvider:
    """Immutable controlled provider keyed by geometric track ID."""

    def __init__(
        self,
        profiles_by_track_id: Mapping[int, PhysicalRiskProfile],
    ) -> None:
        if not isinstance(profiles_by_track_id, Mapping):
            raise TypeError("profiles_by_track_id must be a mapping.")
        profiles: dict[int, PhysicalRiskProfile] = {}
        for track_id, profile in profiles_by_track_id.items():
            if isinstance(track_id, bool) or not isinstance(track_id, int):
                raise TypeError("CRA profile track IDs must be integers.")
            if track_id < 0:
                raise ValueError("CRA profile track IDs cannot be negative.")
            if not isinstance(profile, PhysicalRiskProfile):
                raise TypeError(
                    "Every configured CRA profile must be a PhysicalRiskProfile."
                )
            profiles[track_id] = profile
        self._profiles = MappingProxyType(profiles)

    @property
    def profiles_by_track_id(self) -> Mapping[int, PhysicalRiskProfile]:
        """Return the immutable controlled track-to-profile mapping."""

        return self._profiles

    def profile_for(
        self,
        obstacle: SemanticObstacle,
    ) -> PhysicalRiskProfile | None:
        """Return the profile configured for the obstacle's geometric ID."""

        if not isinstance(obstacle, SemanticObstacle):
            raise TypeError("obstacle must be a SemanticObstacle.")
        return self._profiles.get(obstacle.track_id)


@dataclass(frozen=True)
class CRATrackPolicyAssignment:
    """One semantic track and its evaluated CRA-to-Safe-SAGE parameters."""

    obstacle: SemanticObstacle
    profile: PhysicalRiskProfile
    policy_result: CRASafeSAGEPolicyResult
    used_fallback_profile: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.obstacle, SemanticObstacle):
            raise TypeError("obstacle must be a SemanticObstacle.")
        if not isinstance(self.profile, PhysicalRiskProfile):
            raise TypeError("profile must be a PhysicalRiskProfile.")
        if not isinstance(self.policy_result, CRASafeSAGEPolicyResult):
            raise TypeError("policy_result must be a CRASafeSAGEPolicyResult.")
        if not isinstance(self.used_fallback_profile, bool):
            raise TypeError("used_fallback_profile must be a boolean.")

    @property
    def track_id(self) -> int:
        """Return the underlying geometric track identifier."""

        return self.obstacle.track_id

    @property
    def safety_parameters(self) -> ClassSafetyParameters:
        """Return source-grid risk and normal-flux parameters."""

        return self.policy_result.guidance_parameters.safety_parameters

    @property
    def guidance_parameters(self) -> SafeSAGEGuidanceParameters:
        """Return complete Safe-SAGE normal and tangential parameters."""

        return self.policy_result.guidance_parameters

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible track, semantics, and policy diagnostics."""

        return {
            "track_id": self.track_id,
            "semantic_class": self.obstacle.semantic_class,
            "semantic_confidence": self.obstacle.semantic_confidence,
            "used_fallback_profile": self.used_fallback_profile,
            "policy": self.policy_result.as_dict(),
        }


@dataclass(frozen=True)
class CRATrackPolicyResult:
    """Validated per-track CRA policy assignments for one sensor frame."""

    assignments: tuple[CRATrackPolicyAssignment, ...]

    def __post_init__(self) -> None:
        try:
            assignments = tuple(self.assignments)
        except TypeError as error:
            raise TypeError("assignments must be an iterable.") from error
        seen_track_ids: set[int] = set()
        for assignment in assignments:
            if not isinstance(assignment, CRATrackPolicyAssignment):
                raise TypeError("Every assignment must be a CRATrackPolicyAssignment.")
            if assignment.track_id in seen_track_ids:
                raise ValueError(
                    f"assignments contains duplicate track ID {assignment.track_id}."
                )
            seen_track_ids.add(assignment.track_id)
        object.__setattr__(self, "assignments", assignments)

    @property
    def track_ids(self) -> tuple[int, ...]:
        """Return assigned track identifiers in deterministic order."""

        return tuple(assignment.track_id for assignment in self.assignments)

    @property
    def fallback_track_ids(self) -> tuple[int, ...]:
        """Return tracks whose unavailable profile resolved fail-closed."""

        return tuple(
            assignment.track_id
            for assignment in self.assignments
            if assignment.used_fallback_profile
        )

    @property
    def safety_parameters_by_track_id(
        self,
    ) -> Mapping[int, ClassSafetyParameters]:
        """Return immutable source-grid parameters indexed by track ID."""

        return MappingProxyType(
            {
                assignment.track_id: assignment.safety_parameters
                for assignment in self.assignments
            }
        )

    @property
    def guidance_parameters_by_track_id(
        self,
    ) -> Mapping[int, SafeSAGEGuidanceParameters]:
        """Return immutable guidance parameters indexed by track ID."""

        return MappingProxyType(
            {
                assignment.track_id: assignment.guidance_parameters
                for assignment in self.assignments
            }
        )

    def assignment_for(self, track_id: int) -> CRATrackPolicyAssignment:
        """Return one assignment or raise for an unassigned track ID."""

        if isinstance(track_id, bool) or not isinstance(track_id, int):
            raise TypeError("track_id must be an integer.")
        for assignment in self.assignments:
            if assignment.track_id == track_id:
                return assignment
        raise KeyError(f"No CRA policy assignment exists for track ID {track_id}.")

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible per-frame CRA policy diagnostics."""

        return {
            "track_ids": list(self.track_ids),
            "fallback_track_ids": list(self.fallback_track_ids),
            "assignments": [assignment.as_dict() for assignment in self.assignments],
        }


def resolve_cra_track_policy(
    semantic_obstacles: Iterable[SemanticObstacle],
    *,
    profile_provider: CRARiskProfileProvider,
    policy: CRASafeSAGEPolicy,
) -> CRATrackPolicyResult:
    """Evaluate one complete, deterministic set of semantic track profiles."""

    if not isinstance(profile_provider, CRARiskProfileProvider):
        raise TypeError("profile_provider must implement CRARiskProfileProvider.")
    if not isinstance(policy, CRASafeSAGEPolicy):
        raise TypeError("policy must be a CRASafeSAGEPolicy.")
    try:
        obstacles = tuple(semantic_obstacles)
    except TypeError as error:
        raise TypeError("semantic_obstacles must be an iterable.") from error
    by_track_id: dict[int, SemanticObstacle] = {}
    for obstacle in obstacles:
        if not isinstance(obstacle, SemanticObstacle):
            raise TypeError("Every semantic obstacle must be a SemanticObstacle.")
        if obstacle.track_id in by_track_id:
            raise ValueError(
                f"semantic_obstacles contains duplicate track ID {obstacle.track_id}."
            )
        by_track_id[obstacle.track_id] = obstacle

    assignments: list[CRATrackPolicyAssignment] = []
    for track_id in sorted(by_track_id):
        obstacle = by_track_id[track_id]
        profile = profile_provider.profile_for(obstacle)
        used_fallback = profile is None
        if profile is None:
            profile = PhysicalRiskProfile.unknown(source="cra_profile_unavailable")
        if not isinstance(profile, PhysicalRiskProfile):
            raise TypeError(
                "profile_provider.profile_for must return a "
                "PhysicalRiskProfile or None."
            )
        assignments.append(
            CRATrackPolicyAssignment(
                obstacle=obstacle,
                profile=profile,
                policy_result=policy.evaluate(profile),
                used_fallback_profile=used_fallback,
            )
        )
    return CRATrackPolicyResult(tuple(assignments))
