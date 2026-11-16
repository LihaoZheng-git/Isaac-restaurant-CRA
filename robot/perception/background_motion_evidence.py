"""Bounded translation evidence for revoking a qualified static-map prior.

This is inferred motion under a rigid-translation hypothesis, not identity,
free space, a calibrated hidden-mover detector, or fresh CRA factor evidence.
Ambiguous/partial/linear geometry does not disprove the opted-in static prior.
Current raw obstacles and unconditional human exclusions remain independent.
"""

import numpy as np
from scipy.spatial import cKDTree


def _translation(first, second, resolution):
    """Fit on bounded samples, verify the full two observed surfaces."""
    a, b = np.unique(first, axis=0), np.unique(second, axis=0)
    if min(len(a), len(b)) < 6:
        return None
    # A straight wall/cropped edge cannot identify tangential displacement.
    for cloud in (a, b):
        if np.linalg.eigvalsh(np.cov(cloud.T))[0] < resolution**2:
            return None
    aa = a[np.linspace(0, len(a)-1, min(256, len(a)), dtype=int)]
    bb = b[np.linspace(0, len(b)-1, min(256, len(b)), dtype=int)]
    atree, btree = cKDTree(a), cKDTree(b)
    shift = np.median(b, axis=0)-np.median(a, axis=0)
    for _ in range(8):
        _, ai = btree.query(aa+shift)
        _, bi = atree.query(bb-shift)
        correction = np.vstack((b[ai]-aa, bb-a[bi]))
        updated = np.median(correction, axis=0)
        if np.linalg.norm(updated-shift) < 1e-6:
            shift = updated
            break
        shift = updated
    aligned = np.r_[btree.query(a+shift)[0], atree.query(b-shift)[0]]
    stationary = np.r_[btree.query(a)[0], atree.query(b)[0]]
    # Full bidirectional support, and a substantial improvement over no motion.
    if (aligned.max() > resolution/2
            or np.sqrt(np.mean(stationary**2))-np.sqrt(np.mean(aligned**2)) < resolution/2):
        return None
    return dict(translation_xy=shift.tolist(), maximum_residual_m=float(aligned.max()),
        stationary_rms_m=float(np.sqrt(np.mean(stationary**2))),
        aligned_rms_m=float(np.sqrt(np.mean(aligned**2))))


def confirm_background_motion(history, track_id, *, maximum_gap=.35, resolution=.06):
    """Require three continuous real sources and mutually consistent geometry.

    History consists of (source_time, {track_id: non-human observed XY points}).
    A missing ID breaks the sequence; no identity transfer or long-gap merge.
    The oldest/middle/current observations are compared, with exact full-surface
    residual checks. Resolution is the existing fixed-world correspondence bound.
    """
    sequence = []
    for time, tracks in reversed(history):
        if track_id not in tracks or sequence and sequence[-1][0]-time > maximum_gap+1e-9:
            break
        sequence.append((time, tracks[track_id]))
    sequence.reverse()
    proof = dict(confirmed=False, reason='insufficient_continuous_geometry',
        source_times=[t for t, _ in sequence], sampled_source_times=[], fits=[])
    if len(sequence) < 3 or sequence[-1][0]-sequence[0][0] < .2-1e-9:
        return proof
    middle = min(range(1, len(sequence)-1),
        key=lambda i: abs(sequence[i][0]-(sequence[0][0]+sequence[-1][0])/2))
    selected = (sequence[0], sequence[middle], sequence[-1])
    proof['sampled_source_times'] = [t for t, _ in selected]
    fits = [_translation(selected[a][1], selected[b][1], resolution)
        for a, b in ((0, 1), (1, 2), (0, 2))]
    if any(f is None for f in fits):
        proof['reason'] = 'ambiguous_or_inconsistent_shape'
        return proof
    proof['fits'] = fits
    shifts = [np.asarray(f['translation_xy']) for f in fits]
    dt = np.diff(proof['sampled_source_times'])
    velocities = [shifts[i]/dt[i] for i in range(2)]
    speed = [np.linalg.norm(v) for v in velocities]
    # Resolve displacement beyond two correspondence radii; both intervals
    # must agree in direction/speed. No relaxed residual or clock on failure.
    consistent = (np.linalg.norm(shifts[2]) > 2*resolution
        and min(speed) > .2 and max(speed)/min(speed) <= 1.5
        and np.dot(velocities[0], velocities[1])/(speed[0]*speed[1]) >= .9
        and np.linalg.norm(shifts[0]+shifts[1]-shifts[2]) <= resolution/2)
    proof.update(confirmed=bool(consistent),
        reason='consistent_observed_translation' if consistent else 'unresolved_or_inconsistent_displacement')
    return proof
