"""Conservative continuity stop for the separated dynamic-actor experiment.

Remembering a positive observation is not observing free space. This watchdog
does not supply geometry, motion, factor evidence or a replacement controller.
Its distances are explicit engineering scope limits, not a safety theorem.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class NearbyTrackLossConfig:
    maximum_missing_seconds: float = 0.15
    stop_distance_m: float = 1.2
    enrollment_distance_m: float = 2.0
    reassociation_distance_m: float = 0.25
    allow_observed_fragment_merges: bool = False
    positive_merge_distance_m: float = 0.10
    minimum_positive_merge_returns: int = 3
    use_observed_merge_velocity: bool = False
    allow_positive_extent_merges: bool = False
    allow_positive_surface_merges: bool = False
    allow_unique_full_surface_merges: bool = False
    allow_many_to_one_full_surface_merges: bool = False
    allow_stationary_furniture_memory: bool = False
    allow_current_visual_part_associations: bool = False
    maximum_visual_mask_boundary_pixels: int = 0
    maximum_merge_speed_mps: float = 2.0
    maximum_merge_translation_m: float = 0.25
    maximum_merge_velocity_change_mps: float = 0.3
    maximum_merge_age_seconds: float = 0.35

    def __post_init__(self):
        if not isinstance(self.allow_stationary_furniture_memory, bool) or not isinstance(self.allow_observed_fragment_merges, bool) or not isinstance(
            self.use_observed_merge_velocity, bool
        ) or not isinstance(self.allow_positive_extent_merges, bool) or not isinstance(
            self.allow_positive_surface_merges, bool
        ) or not isinstance(
            self.allow_current_visual_part_associations, bool
        ) or not isinstance(
            self.allow_unique_full_surface_merges, bool
        ) or not isinstance(
            self.allow_many_to_one_full_surface_merges, bool
        ):
            raise TypeError("fragment merge opt-in must be boolean")
        if self.allow_positive_extent_merges and not self.allow_observed_fragment_merges:
            raise ValueError("positive extent association requires fragment merges")
        if self.allow_positive_surface_merges and not self.allow_positive_extent_merges:
            raise ValueError("positive surface association requires positive extent merges")
        if self.allow_unique_full_surface_merges and not self.allow_positive_surface_merges:
            raise ValueError("unique full surface association requires positive surface merges")
        if self.allow_many_to_one_full_surface_merges and not self.allow_unique_full_surface_merges:
            raise ValueError("many-to-one coverage requires unique full surface association")
        if (isinstance(self.maximum_visual_mask_boundary_pixels, bool)
            or not isinstance(self.maximum_visual_mask_boundary_pixels, int)
            or self.maximum_visual_mask_boundary_pixels not in (0, 1)
            or self.maximum_visual_mask_boundary_pixels and not self.allow_current_visual_part_associations):
            raise ValueError("one-pixel mask boundary prior requires explicit current visual parts")
        for name, value in vars(self).items():
            if name in {
                "allow_observed_fragment_merges",
                "use_observed_merge_velocity",
                "allow_positive_extent_merges",
                "allow_positive_surface_merges",
                "allow_unique_full_surface_merges",
                "allow_many_to_one_full_surface_merges",
                "allow_stationary_furniture_memory",
                "allow_current_visual_part_associations",
                "maximum_visual_mask_boundary_pixels",
            }:
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("track-loss guard limits must be finite and positive")
        if self.enrollment_distance_m < self.stop_distance_m:
            raise ValueError("enroll observations before they enter the stop region")
        if (
            self.positive_merge_distance_m > self.reassociation_distance_m
            or self.maximum_merge_translation_m > self.reassociation_distance_m
            or isinstance(self.minimum_positive_merge_returns, bool)
            or not isinstance(self.minimum_positive_merge_returns, int)
        ):
            raise ValueError("invalid positive merge geometry gates")


class NearbyTrackLossGuard:
    def __init__(self, config=None, *, fragment_coverage_config=None):
        self.config = config if config is not None else NearbyTrackLossConfig()
        if not isinstance(self.config, NearbyTrackLossConfig):
            raise TypeError("expected NearbyTrackLossConfig")
        self.observed = {}
        self.last_source_time = None
        self.velocities = {}
        self.coobserved = {}
        self.positive_surfaces = {}
        from robot.perception.human_fragment_coverage import HumanFragmentCoverage
        self.fragment_coverage = HumanFragmentCoverage(fragment_coverage_config) if fragment_coverage_config is not None else None

    def observe(
        self,
        tracks,
        *,
        source_time,
        robot_world_xy,
        current_points_world_xy=None,
        current_point_track_ids=None,
        current_points_source_time=None,
        current_track_risks=None,
        current_body_envelopes=None,
        current_occupancy_grid=None,
        current_grid_anchor_xy=None,
        current_grid_anchor_yaw=None,
        current_camera_timestamp_seconds=None,
        current_visual_part_associations=(),
        current_furniture_snapshot=None,
    ):
        if (
            isinstance(source_time, bool)
            or not math.isfinite(source_time)
            or source_time < 0
            or self.last_source_time is not None
            and source_time <= self.last_source_time
        ):
            raise ValueError("track-loss guard source epochs must increase")
        robot = np.asarray(robot_world_xy, dtype=float)
        if robot.shape != (2,) or not np.isfinite(robot).all():
            raise ValueError("robot world position must be a finite planar vector")
        if any(
            isinstance(track.track_id, bool)
            or not isinstance(track.track_id, (int, np.integer))
            or track.track_id < 0
            or track.point_count <= 0
            for track in tracks
        ):
            raise ValueError("the guard requires positive observed geometric tracks")
        current = {
            track.track_id: np.asarray(track.centroid_world_xy, dtype=float)
            for track in tracks
        }
        if len(current) != len(tracks) or any(
            p.shape != (2,) or not np.isfinite(p).all() for p in current.values()
        ):
            raise ValueError(
                "current geometric tracks must have unique IDs and finite centers"
            )
        velocities = {
            t.track_id: np.asarray(t.velocity_world_xy, dtype=float) for t in tracks
        }
        if any(
            v.shape != (2,) or not np.isfinite(v).all() for v in velocities.values()
        ):
            raise ValueError("finite planar track velocities required")
        observations = {t.track_id: t.observation_count for t in tracks}
        geometry = None
        if any(
            v is not None
            for v in (
                current_points_world_xy,
                current_point_track_ids,
                current_points_source_time,
            )
        ):
            xyz = np.asarray(current_points_world_xy, dtype=float)
            ids = np.asarray(current_point_track_ids)
            if (
                xyz.ndim != 2
                or xyz.shape[1] != 2
                or not np.isfinite(xyz).all()
                or ids.shape != (len(xyz),)
                or not np.issubdtype(ids.dtype, np.integer)
                or current_points_source_time is None
                or not math.isfinite(current_points_source_time)
                or abs(current_points_source_time - source_time) > 1e-9
                or np.any(ids < -1)
                or any(int(k) not in current for k in np.unique(ids) if k >= 0)
                or any(
                    np.count_nonzero(ids == t.track_id) != t.point_count for t in tracks
                )
            ):
                raise ValueError(
                    "fragment merge requires aligned current positive sensor geometry"
                )
            geometry = (xyz, ids)
        if self.config.allow_positive_surface_merges and geometry is None:
            raise ValueError("positive surface association requires aligned current geometry")
        part_children = set()
        if current_visual_part_associations:
            if not self.config.allow_current_visual_part_associations or geometry is None:
                raise ValueError("current visual parts require explicit opt-in and positive geometry")
            for proof in current_visual_part_associations:
                child, parent = proof["child_track_id"], proof["parent_track_id"]
                strict_mask = (proof["basis"] == "fresh_person_mask_whole_positive_small_fragment"
                    and proof["status"] == "inferred")
                boundary_prior = (self.config.maximum_visual_mask_boundary_pixels == 1
                    and proof["basis"] == "fresh_person_mask_boundary_prior_positive_small_fragment"
                    and proof["status"] == "prior" and proof.get("mask_boundary_prior_pixels") == 1
                    and .95 <= proof.get("child_original_mask_coverage", -1) < 1
                    and 0 < proof.get("child_boundary_return_count", 0) <= .05*proof["child_positive_count"] + 1e-9
                    and proof.get("all_child_returns_inside_bounded_mask") is True)
                if (child == parent or child not in current or parent not in current
                    or child in part_children
                    or not (strict_mask or boundary_prior)
                    or proof["source_timestamp_seconds"] != source_time
                    or not 0 <= proof["camera_timestamp_seconds"] - source_time <= .15 + 1e-9
                    or proof["child_positive_count"] != np.count_nonzero(geometry[1] == child)
                    or proof["parent_positive_count"] != np.count_nonzero(geometry[1] == parent)
                    or not 3 <= proof["child_positive_count"] <= .1 * proof["parent_positive_count"]
                    or not 0 <= proof["maximum_child_to_parent_3d_distance_m"] <= .3
                    or not .9 <= proof["parent_mask_coverage"] <= 1
                    or not .5 <= proof["person_confidence"] <= 1):
                    raise ValueError("invalid current positive visual-part proof")
                part_children.add(child)
            if any(p["parent_track_id"] in part_children for p in current_visual_part_associations):
                raise ValueError("visual part associations cannot be nested")
        prior_coverage = {"covered": [], "rejected": []}
        if self.fragment_coverage is not None:
            import copy
            from robot.perception.human_fragment_coverage import validate_coverage_grid
            if (geometry is None or current_track_risks is None or current_body_envelopes is None
                or current_occupancy_grid is None or current_grid_anchor_xy is None
                or current_grid_anchor_yaw is None):
                raise ValueError("prior coverage requires the current field, risks, geometry and envelopes")
            fragment_memory = copy.deepcopy(self.fragment_coverage)
            prior_coverage = fragment_memory.update(tracks, *geometry, current_track_risks,
                current_body_envelopes, source_time=source_time,
                camera_timestamp_seconds=current_camera_timestamp_seconds)
            checked = []
            for proof in prior_coverage["covered"]:
                check = validate_coverage_grid(proof, current_occupancy_grid,
                    current_grid_anchor_xy, current_grid_anchor_yaw)
                if check["covered"]:
                    checked.append({**proof, "field_coverage": check})
                else:
                    prior_coverage["rejected"].append({**proof, "reason": check["reason"]})
            prior_coverage["covered"] = checked
        pending = dict(self.observed)
        part_parents = {p["parent_track_id"] for p in current_visual_part_associations
                        if p["child_track_id"] in pending or np.linalg.norm(
                            current[p["child_track_id"]] - robot) <= self.config.enrollment_distance_m}
        # Resolve only STILL OBSERVED children with a fresh whole-positive proof.
        # This cannot clear an already missing fragment or transfer its factors.
        for child in part_children:
            pending.pop(child, None)
        coobserved = {
            pair: t
            for pair, t in self.coobserved.items()
            if source_time - t <= self.config.maximum_merge_age_seconds + 1e-9
        }
        missing = {key: value for key, value in pending.items() if key not in current}
        candidates = {
            key: [
                new
                for new, point in current.items()
                if new not in pending
                and np.linalg.norm(point - old[0])
                <= self.config.reassociation_distance_m
            ]
            for key, old in missing.items()
        }
        # Only unambiguous one-to-one positive geometry can resolve an ID change.
        # This never transfers or refreshes old mass/contact/plume/support cues.
        reassociated = set()
        for key, matches in candidates.items():
            if (
                len(matches) == 1
                and sum(matches[0] in others for others in candidates.values()) == 1
            ):
                if self.config.allow_observed_fragment_merges and geometry is not None:
                    xyz, ids = geometry
                    near = ids[
                        np.linalg.norm(xyz - missing[key][0], axis=1)
                        <= self.config.positive_merge_distance_m
                    ]
                    supported = [
                        int(n)
                        for n in np.unique(near)
                        if np.count_nonzero(near == n)
                        >= self.config.minimum_positive_merge_returns
                    ]
                    if supported != matches:
                        continue
                del pending[key]
                reassociated.add(matches[0])
        # An articulated body can split then rejoin a still-observed cluster.
        # Resolve this only with a unique current cluster and multiple actually
        # observed returns near its old or velocity-advected location. The
        # latter is an explicit bounded kinematic association hypothesis.
        # No free-space, old body box, factor or semantic identity transfers.
        merged = []
        surface_proofs, surface_rejections = {}, []
        if self.config.allow_observed_fragment_merges and geometry is not None:
            xyz, ids = geometry
            surface_tree = None

            def stable_current_velocity(old_time, new):
                dt = source_time - old_time
                previous_v = self.velocities.get(new)
                v = velocities[new]
                return (
                    self.config.use_observed_merge_velocity
                    and previous_v is not None
                    and observations[new] >= 3
                    and 0
                    < source_time - previous_v[1]
                    <= self.config.maximum_missing_seconds + 1e-9
                    and 0 < dt <= self.config.maximum_merge_age_seconds + 1e-9
                    and np.linalg.norm(v) <= self.config.maximum_merge_speed_mps
                    and np.linalg.norm(v - previous_v[0])
                    <= self.config.maximum_merge_velocity_change_mps
                    and np.linalg.norm(dt * v)
                    <= self.config.maximum_merge_translation_m
                )

            def match_query(old_key, old_point, old_time, new, *, extent_pair=False):
                nonlocal surface_tree
                queries = [(old_point, "old_centroid_positive_returns", np.zeros(2))]
                dt = source_time - old_time
                v = velocities[new]
                translation = dt * v
                if stable_current_velocity(old_time, new):
                    queries.append(
                        (
                            old_point + translation,
                            "inferred_current_cluster_velocity_positive_returns",
                            translation,
                        )
                    )
                for q, basis, shift in queries:
                    if (
                        np.linalg.norm(current[new] - q)
                        > self.config.reassociation_distance_m
                    ):
                        # A connected body can cover a fragment's old location
                        # while its centroid is farther away. Only a recent
                        # actual co-observation and fresh direct positive returns
                        # can take this route; never an advected query or an
                        # assumed body box. All occupancy remains in the field.
                        if not (extent_pair and basis == "old_centroid_positive_returns"):
                            continue
                        basis = "coobserved_cluster_extent_positive_returns"
                    near = ids[
                        np.linalg.norm(xyz - q, axis=1)
                        <= self.config.positive_merge_distance_m
                    ]
                    supported = [
                        int(n)
                        for n in np.unique(near)
                        if np.count_nonzero(near == n)
                        >= self.config.minimum_positive_merge_returns
                    ]
                    if supported == [new]:
                        return q, basis, shift
                if self.config.allow_positive_surface_merges and extent_pair:
                    original = self.positive_surfaces.get(old_key)
                    reason = None
                    maximum_distance = None
                    if original is None or original[1] != old_time:
                        reason = "no_original_surface_at_fragment_epoch"
                    elif (not self.config.allow_unique_full_surface_merges
                        and np.linalg.norm(current[new] - old_point) > self.config.reassociation_distance_m):
                        reason = "original_centroid_distance_gate"
                    else:
                        from scipy.spatial import cKDTree

                        old_surface = original[0]
                        minimum = self.config.minimum_positive_merge_returns
                        if len(np.unique(old_surface, axis=0)) < minimum:
                            reason = "too_few_distinct_original_returns"
                        else:
                            parent_points = xyz[ids == new]
                            distances, _ = cKDTree(parent_points).query(old_surface)
                            maximum_distance = float(np.max(distances))
                            if np.any(distances > self.config.positive_merge_distance_m):
                                reason = "original_surface_not_fully_observed_now"
                            else:
                                if surface_tree is None:
                                    surface_tree = cKDTree(xyz)
                                neighborhoods = surface_tree.query_ball_point(
                                    old_surface, self.config.positive_merge_distance_m
                                )
                                indices = np.unique(np.concatenate(neighborhoods)).astype(int)
                                near_ids = ids[indices]
                                supported = [int(k) for k in np.unique(near_ids)
                                    if np.count_nonzero(near_ids == k) >= minimum]
                                neighborhood_supported = supported.copy()
                                if self.config.allow_unique_full_surface_merges and -1 not in supported:
                                    # A nearby patch is not a competing WHOLE
                                    # surface correspondence. Keep ambiguity if
                                    # two clusters cover the entire old surface;
                                    # unclustered support remains unresolved.
                                    supported = [k for k in supported if np.all(
                                        cKDTree(xyz[ids == k]).query(old_surface)[0]
                                        <= self.config.positive_merge_distance_m)]
                                covering = xyz[indices[near_ids == new]]
                                if supported != [new]:
                                    reason = "ambiguous_surface_ownership"
                                elif len(np.unique(covering, axis=0)) < minimum:
                                    reason = "too_few_distinct_current_returns"
                                else:
                                    surface_proofs[old_key, new] = {
                                        "original_surface_source_time": old_time,
                                        "current_surface_source_time": source_time,
                                        "original_positive_return_count": len(old_surface),
                                        "current_covering_return_count": len(covering),
                                        "maximum_original_to_current_distance_m": maximum_distance,
                                        "all_original_positive_returns_covered": True,
                                        **({"correspondence_basis": "unique_whole_surface_positive_coverage",
                                            "neighborhood_supported_track_ids": neighborhood_supported,
                                            "whole_surface_supported_track_ids": supported,
                                            "parent_centroid_distance_m": float(np.linalg.norm(current[new]-old_point))}
                                           if self.config.allow_unique_full_surface_merges else {}),
                                        "original_surface_points_world_xy": old_surface.tolist(),
                                        "current_covering_points_world_xy": covering.tolist(),
                                        "scope": "whole measured planar positive-surface association; no hidden-body coverage, motion/factor transfer or new occupancy",
                                    }
                                    return old_point, "coobserved_whole_positive_surface_returns", np.zeros(2)
                    surface_rejections.append({"lost_track_id": int(old_key),
                        "current_track_id": int(new), "original_source_time": old_time,
                        "current_source_time": source_time, "reason": reason,
                        "original_surface_points_world_xy": None
                        if original is None else original[0].tolist(),
                        "maximum_original_to_current_distance_m": maximum_distance})
                return None

            proposals = {}
            matches = {}
            coexistence = {}
            for key, (old_point, old_time) in missing.items():
                if key not in pending:
                    continue
                choices = []
                for new in current:
                    pair_time = coobserved.get(tuple(sorted((key, new))))
                    recent_parent = (
                        new in pending
                        and abs(pending[new][1] - old_time)
                        <= self.config.maximum_missing_seconds + 1e-9
                    )
                    retained_pair = (
                        pair_time is not None
                        and abs(pair_time - old_time)
                        <= self.config.maximum_missing_seconds + 1e-9
                        and stable_current_velocity(old_time, new)
                    )
                    extent_pair = (
                        self.config.allow_positive_extent_merges
                        and pair_time is not None
                        and 0 <= old_time - pair_time <= self.config.maximum_missing_seconds + 1e-9
                        and 0 < source_time - old_time <= self.config.maximum_merge_age_seconds + 1e-9
                    )
                    if (
                        new in pending
                        and new != key
                        and (recent_parent or retained_pair or extent_pair)
                    ):
                        found = match_query(key, old_point, old_time, new, extent_pair=extent_pair)
                        if found is not None:
                            choices.append(new)
                            matches[key, new] = found
                            coexistence[key, new] = (
                                pair_time,
                                "recent_original_coobservation_positive_extent"
                                if found[1] == "coobserved_cluster_extent_positive_returns"
                                else "recent_original_coobservation_positive_surface"
                                if found[1] == "coobserved_whole_positive_surface_returns"
                                else "recent_parent_epoch"
                                if recent_parent
                                else "retained_original_coobservation",
                            )
                proposals[key] = choices
            many_proofs = {}
            if self.config.allow_many_to_one_full_surface_merges:
                from scipy.spatial import cKDTree
                for new in current:
                    children = [key for key, choices in proposals.items() if new in choices]
                    if len(children) < 2 or any(proposals[key] != [new] for key in children):
                        continue
                    proofs = {}
                    for key in children:
                        original = self.positive_surfaces.get(key)
                        old_time = missing[key][1]
                        pair_time = coobserved.get(tuple(sorted((key, new))))
                        if (original is None or original[1] != old_time or pair_time is None
                            or not 0 <= old_time-pair_time <= self.config.maximum_missing_seconds+1e-9
                            or not 0 < source_time-old_time <= self.config.maximum_merge_age_seconds+1e-9):
                            break
                        old_surface = original[0]
                        minimum = self.config.minimum_positive_merge_returns
                        if len(np.unique(old_surface, axis=0)) < minimum:
                            break
                        distance, _ = cKDTree(xyz[ids == new]).query(old_surface)
                        if np.any(distance > self.config.positive_merge_distance_m):
                            break
                        tree = cKDTree(xyz)
                        neighbors = tree.query_ball_point(old_surface, self.config.positive_merge_distance_m)
                        indices = np.unique(np.concatenate(neighbors)).astype(int)
                        near_ids = ids[indices]
                        supported = [int(k) for k in np.unique(near_ids)
                            if np.count_nonzero(near_ids == k) >= minimum]
                        if -1 in supported:
                            break
                        whole = [k for k in supported if np.all(cKDTree(xyz[ids == k]).query(
                            old_surface)[0] <= self.config.positive_merge_distance_m)]
                        covering = xyz[indices[near_ids == new]]
                        if whole != [new] or len(np.unique(covering, axis=0)) < minimum:
                            break
                        proofs[key, new] = dict(original_surface_source_time=old_time,
                            current_surface_source_time=source_time,
                            original_positive_return_count=len(old_surface),
                            current_covering_return_count=len(covering),
                            maximum_original_to_current_distance_m=float(distance.max()),
                            all_original_positive_returns_covered=True,
                            original_surface_points_world_xy=old_surface.tolist(),
                            current_covering_points_world_xy=covering.tolist(),
                            whole_surface_supported_track_ids=whole,
                            jointly_covered_original_track_ids=children,
                            scope='fresh complete positive-surface coverage for each coobserved fragment; geometry continuity only, no identity/factor transfer or retained occupancy')
                    if len(proofs) == len(children):
                        many_proofs.update(proofs)
            for key, choices in proposals.items():
                if (
                    len(choices) == 1
                    and (sum(choices[0] in v for v in proposals.values()) == 1
                         or (key, choices[0]) in many_proofs)
                ):
                    new = choices[0]
                    old_point, old_time = pending.pop(key)
                    query, basis, shift = matches[key, new]
                    if (key, new) in many_proofs:
                        basis = 'coobserved_many_to_one_whole_positive_surfaces'
                        query, shift = old_point, np.zeros(2)
                        surface_proofs[key, new] = many_proofs[key, new]
                    merged.append(
                        {
                            "lost_track_id": int(key),
                            "current_track_id": int(new),
                            "original_last_seen_time": old_time,
                            "current_source_time": source_time,
                            "basis": basis,
                            "old_centroid_world_xy": old_point.tolist(),
                            "query_world_xy": query.tolist(),
                            "translation_world_xy": shift.tolist(),
                            "current_velocity_source_time": source_time,
                            "previous_velocity_source_time": self.velocities[new][1]
                            if new in self.velocities
                            else None,
                            "current_cluster_velocity_world_xy": velocities[
                                new
                            ].tolist(),
                            "current_centroid_world_xy": current[new].tolist(),
                            "current_centroid_to_query_distance_m": float(np.linalg.norm(current[new] - query)),
                            "previous_cluster_velocity_world_xy": self.velocities[new][0].tolist()
                            if new in self.velocities else None,
                            "coobservation_source_time": coexistence[key, new][0],
                            "coexistence_basis": coexistence[key, new][1],
                            "original_fragment_age_seconds": source_time - old_time,
                            "positive_returns_near_query": int(
                                np.count_nonzero(
                                    (ids == new)
                                    & (
                                        np.linalg.norm(xyz - query, axis=1)
                                        <= self.config.positive_merge_distance_m
                                    )
                                )
                            ),
                            **({"positive_surface_proof": surface_proofs[key, new]}
                               if (key, new) in surface_proofs else {}),
                        }
                    )
        for key, point in current.items():
            if key in part_children:
                continue
            if (
                key in pending
                or key in part_parents
                or key in reassociated
                or np.linalg.norm(point - robot) <= self.config.enrollment_distance_m
            ):
                pending[key] = (point.copy(), float(source_time))
        prior_coverage["covered"] = [p for p in prior_coverage["covered"]
            if p["track_id"] in pending and p["track_id"] not in current]
        furniture_coverage = []
        if self.config.allow_stationary_furniture_memory:
            from robot.perception.stationary_furniture_memory import FurnitureSnapshot
            if (not isinstance(current_furniture_snapshot, FurnitureSnapshot)
                or current_furniture_snapshot.source_time != source_time
                or current_occupancy_grid is None or current_grid_anchor_xy is None
                or current_grid_anchor_yaw is None):
                raise ValueError("furniture coverage requires the current typed snapshot and built field")
            for key, (point, last_seen) in pending.items():
                if key not in current:
                    proof = current_furniture_snapshot.coverage(key, last_seen, point,
                        current_occupancy_grid, current_grid_anchor_xy, current_grid_anchor_yaw)
                    if proof is not None:
                        furniture_coverage.append(proof)
        elif current_furniture_snapshot is not None:
            raise ValueError("furniture coverage requires explicit opt-in")
        blocking = []
        for key, (point, last_seen) in pending.items():
            age = float(source_time - last_seen)
            distance = float(np.linalg.norm(point - robot))
            if (
                age > self.config.maximum_missing_seconds + 1e-9
                and distance <= self.config.stop_distance_m
                and not any(p["track_id"] == key for p in prior_coverage["covered"])
                and not any(p["track_id"] == key for p in furniture_coverage)
            ):
                blocking.append(
                    {
                        "track_id": key,
                        "last_observed_world_xy": point.tolist(),
                        "original_last_seen_time": last_seen,
                        "missing_seconds": age,
                        "current_robot_distance_m": distance,
                    }
                )
        self.observed, self.last_source_time = pending, float(source_time)
        if self.fragment_coverage is not None:
            self.fragment_coverage = fragment_memory
        self.velocities = {
            key: (velocities[key].copy(), float(source_time))
            if key in current
            else self.velocities[key]
            for key in pending
            if key in current or key in self.velocities
        }
        # Only simultaneous positive observations renew a pair, never borrowing
        # or continued observation of its parent alone. This is identity history,
        # not a body box, factor value, inferred free space or a new source epoch.
        if self.config.use_observed_merge_velocity or self.config.allow_positive_extent_merges:
            visible = sorted(set(current) & set(pending))
            for i, a in enumerate(visible):
                for b in visible[i + 1 :]:
                    coobserved[a, b] = float(source_time)
        self.coobserved = {
            pair: t for pair, t in coobserved.items() if all(k in pending for k in pair)
        }
        if self.config.allow_positive_surface_merges:
            # Association history only: never feed these old points to the field
            # or renew their source epoch when only the parent is still visible.
            self.positive_surfaces = {
                key: (xyz[ids == key].copy(), float(source_time))
                if key in current else self.positive_surfaces[key]
                for key in pending
                if key in current or key in self.positive_surfaces
                and source_time - self.positive_surfaces[key][1]
                <= self.config.maximum_merge_age_seconds + 1e-9
            }
        return {
            "blocked": bool(blocking),
            "blocking_tracks": blocking,
            "watched_tracks": len(pending),
            "source_timestamp": source_time,
            "observed_fragment_merges": merged,
            "current_visual_part_associations": list(current_visual_part_associations),
            "positive_extent_route_enabled": self.config.allow_positive_extent_merges,
            "body_prior_coverage_enabled": self.fragment_coverage is not None,
            "prior_covered_missing_tracks": prior_coverage["covered"],
            "body_prior_coverage_rejections": prior_coverage["rejected"],
            "furniture_prior_covered_missing_tracks": furniture_coverage,
            **({"positive_surface_route_enabled": True,
                "positive_surface_merge_rejections": surface_rejections,
                "many_to_one_full_surface_route_enabled": self.config.allow_many_to_one_full_surface_merges,
                "unique_full_surface_route_enabled": self.config.allow_unique_full_surface_merges}
               if self.config.allow_positive_surface_merges else {}),
            "scope": "nearby positive LiDAR-track continuity watchdog; lost occupancy is unresolved, not free; bounded separated actors; not a dynamic belief map or general occlusion guarantee",
        }
