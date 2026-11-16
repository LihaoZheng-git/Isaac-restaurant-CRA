"""Exact positive-surface replacement witnesses; no identity or factor inference."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


SUPPORT_DISTANCE_M = .10
MAXIMUM_DISTINCT_ORIGINAL_POINTS = 256
MAXIMUM_CURRENT_POINTS = 65_536
MAXIMUM_CANDIDATE_EDGES = 65_536
INJECTIVE_BASIS = "full_injective_positive_surface_support"


def _injective_witness(original, current, current_indices):
    old, old_indices = np.unique(original, axis=0, return_index=True)
    new, new_indices = np.unique(current[current_indices], axis=0, return_index=True)
    if not 3 <= len(old) <= MAXIMUM_DISTINCT_ORIGINAL_POINTS or len(new) < len(old):
        return None
    tree = cKDTree(new)
    counts = tree.query_ball_point(old, SUPPORT_DISTANCE_M, return_length=True)
    edges = int(np.sum(counts))
    if np.any(counts == 0) or edges > MAXIMUM_CANDIDATE_EDGES:
        return None
    neighbors = tree.query_ball_point(old, SUPPORT_DISTANCE_M, return_sorted=True)
    columns = np.concatenate(neighbors).astype(np.int64)
    indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
    graph = csr_matrix((np.ones(edges, dtype=np.uint8), columns, indptr),
                       shape=(len(old), len(new)))
    matches = maximum_bipartite_matching(graph, perm_type="column")
    if np.any(matches < 0):
        return None
    witnesses = current_indices[new_indices[matches]]
    distances = np.linalg.norm(original[old_indices] - current[witnesses], axis=1)
    # Check explicit witnesses too: a tree tolerance may never widen the bound.
    if np.any(distances > SUPPORT_DISTANCE_M):
        return None
    return dict(basis=INJECTIVE_BASIS, maximum_distance_m=SUPPORT_DISTANCE_M,
        original_point_count=len(original), original_distinct_point_count=len(old),
        current_owner_point_count=len(current_indices), current_owner_distinct_point_count=len(new),
        original_unique_indices=old_indices.tolist(), current_witness_indices=witnesses.tolist(),
        original_unique_xy=old.tolist(), current_witness_xy=current[witnesses].tolist(),
        maximum_witness_distance_m=float(distances.max()), candidate_edge_count=edges,
        resource_limits=dict(maximum_distinct_original_points=MAXIMUM_DISTINCT_ORIGINAL_POINTS,
            maximum_current_points=MAXIMUM_CURRENT_POINTS, maximum_candidate_edges=MAXIMUM_CANDIDATE_EDGES))


def whole_surface_replacement(original_points, current_points, current_ids, *,
                              raw_tree=None, allow_injective=True):
    """Return (owner, nearest support distance, optional injective proof).

    Existing nearest-witness acceptance is unchanged. Only when that diversity
    is below three does a bounded sparse matching try to witness EVERY distinct
    old point with a different current XY from the same uniquely eligible owner.
    Budget exhaustion rejects this additional proof; it never relaxes a bound.
    The caller retains original age, policy and complete current-field checks.
    """
    old, current, ids = np.asarray(original_points), np.asarray(current_points), np.asarray(current_ids)
    if (old.ndim != 2 or old.shape[1] != 2 or not len(old) or not np.isfinite(old).all()
            or current.ndim != 2 or current.shape[1] != 2 or not np.isfinite(current).all()
            or ids.shape != (len(current),) or not np.issubdtype(ids.dtype, np.integer)
            or not isinstance(allow_injective, bool)):
        raise ValueError("aligned finite positive geometry and integer owners required")
    if not len(current):
        return None
    tree = cKDTree(current) if raw_tree is None else raw_tree
    distances, nearest = tree.query(old)
    if not np.all(distances <= SUPPORT_DISTANCE_M):
        return None
    distinct = len(np.unique(current[nearest], axis=0))
    query = old
    if distinct < 3:
        if not allow_injective or len(current) > MAXIMUM_CURRENT_POINTS:
            return None
        query = np.unique(old, axis=0)
        if not 3 <= len(query) <= MAXIMUM_DISTINCT_ORIGINAL_POINTS:
            return None
        counts = tree.query_ball_point(query, SUPPORT_DISTANCE_M, return_length=True)
        if np.sum(counts) > MAXIMUM_CANDIDATE_EDGES:
            return None
    neighbors = tree.query_ball_point(query, SUPPORT_DISTANCE_M)
    indices = np.unique(np.concatenate(neighbors)).astype(int)
    owners = [int(key) for key in np.unique(ids[indices]) if key >= 0
        and np.all(cKDTree(current[ids == key]).query(old)[0] <= SUPPORT_DISTANCE_M)]
    unowned = np.count_nonzero(ids[indices] < 0) >= 3
    if len(owners) != 1 or unowned:
        return None
    replacement, proof = owners[0], None
    if distinct < 3:
        proof = _injective_witness(old, current, np.flatnonzero(ids == replacement))
        if proof is None:
            return None
        proof["nearest_distinct_current_point_count"] = distinct
        proof["maximum_nearest_distance_m"] = float(distances.max())
        proof["current_track_id"] = replacement
    return replacement, float(distances.max()), proof
