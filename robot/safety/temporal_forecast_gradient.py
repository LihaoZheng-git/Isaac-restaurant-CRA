"""Analytic chain rule for the state-sampled Safe-SAGE temporal forecast.

The safety values, clipping policy and execution filter remain independent.
"""

import numpy as np

from robot.safety.safe_sage_cbf_filter import _TrajectoryFieldSampler
from robot.safety.safe_sage_temporal_derivative import SafeSAGETemporalDerivativeResult


def scaled_temporal_forecast_gradient(
    config,
    states,
    *,
    configuration_space,
    h_full,
    guidance_fields,
    temporal_safety_derivative,
):
    """Differentiate the existing scaled temporal interpolant analytically.

    Values/fields are unchanged. Call after the original field validation.
    Derivatives follow the active interpolation and clamp branch; grid/yaw/
    clamp knots need not be differentiable. No finite differences or new
    motion/uncertainty assumptions enter this calculation.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 3 or not np.isfinite(states).all():
        raise ValueError("finite (N,3) safety states required")
    sampler = _TrajectoryFieldSampler(
        states,
        grid=configuration_space.occupancy_grids[0],
        yaw_count=h_full.yaw_slice_count,
    )
    h, g, hyaw = sampler.scalar_with_gradient(h_full.values)
    hgrad = np.column_stack((g, hyaw))
    v00, v01, v10, v11 = sampler._corners(h_full.values)
    dx = (
        (1 - sampler.rf) * (v01 - v00) + sampler.rf * (v11 - v10)
    ) / sampler.resolution
    dy = (
        (1 - sampler.cf) * (v10 - v00) + sampler.cf * (v11 - v01)
    ) / sampler.resolution
    cross = sampler._yaw_sample((v11 - v10 - v01 + v00) / sampler.resolution**2)
    hessian = np.zeros((len(states), 2, 3))
    hessian[:, 0, 1] = cross
    hessian[:, 1, 0] = cross
    hessian[:, 0, 2] = (dx[:, 1] - dx[:, 0]) / sampler.yaw_spacing
    hessian[:, 1, 2] = (dy[:, 1] - dy[:, 0]) / sampler.yaw_spacing
    value = np.empty(sampler.yaws.shape + (2,))
    gx = np.empty_like(value)
    gy = np.empty_like(value)
    for yaw in np.unique(sampler.yaws):
        rows, slots = np.nonzero(sampler.yaws == yaw)
        a = guidance_fields[int(yaw)].vectors_robot_xy
        a00 = a[sampler.r0[rows], sampler.c0[rows]]
        a01 = a[sampler.r0[rows], sampler.c1[rows]]
        a10 = a[sampler.r1[rows], sampler.c0[rows]]
        a11 = a[sampler.r1[rows], sampler.c1[rows]]
        cf, rf = sampler.cf[rows], sampler.rf[rows]
        value[rows, slots] = (1 - rf) * ((1 - cf) * a00 + cf * a01) + rf * (
            (1 - cf) * a10 + cf * a11
        )
        gx[rows, slots] = (
            (1 - rf) * (a01 - a00) + rf * (a11 - a10)
        ) / sampler.resolution
        gy[rows, slots] = (
            (1 - cf) * (a10 - a00) + cf * (a11 - a01)
        ) / sampler.resolution
    def blend(v):
        return ((1 - sampler.yaw_weight[:, None]) * v[:, 0]
            + sampler.yaw_weight[:, None] * v[:, 1])
    guide = blend(value)
    jguide = np.stack(
        (blend(gx), blend(gy), (value[:, 1] - value[:, 0]) / sampler.yaw_spacing),
        axis=2,
    )
    anorm = np.linalg.norm(guide, axis=1)
    bnorm = np.linalg.norm(g, axis=1)
    anormgrad = np.divide(
        np.einsum("ni,nij->nj", guide, jguide),
        anorm[:, None],
        out=np.zeros((len(states), 3)),
        where=anorm[:, None] > 0,
    )
    bnormgrad = np.divide(
        np.einsum("ni,nij->nj", g, hessian),
        bnorm[:, None],
        out=np.zeros((len(states), 3)),
        where=bnorm[:, None] > 0,
    )
    cfg = config
    reg = cfg.temporal_scaling_epsilon * (
        1 - np.exp(-cfg.temporal_scaling_kappa * np.maximum(h, 0))
    )
    denominator = bnorm + reg
    denomgrad = (
        bnormgrad
        + (
            cfg.temporal_scaling_epsilon
            * cfg.temporal_scaling_kappa
            * np.exp(-cfg.temporal_scaling_kappa * np.maximum(h, 0))
            * (h > 0)
        )[:, None]
        * hgrad
    )
    ratio = np.divide(
        anorm,
        denominator,
        out=(anorm > 0).astype(float),
        where=denominator > np.finfo(float).eps,
    )
    scale = np.minimum(ratio, 1.0)
    scalegrad = np.divide(
        anormgrad * denominator[:, None] - anorm[:, None] * denomgrad,
        denominator[:, None] ** 2,
        out=np.zeros((len(states), 3)),
        where=((denominator > np.finfo(float).eps) & (ratio < 1.0))[:, None],
    )
    if isinstance(temporal_safety_derivative, SafeSAGETemporalDerivativeResult):
        tau, txy, tyaw = sampler.scalar_with_gradient(temporal_safety_derivative.values)
        taugrad = np.column_stack((txy, tyaw))
    else:
        tau = np.full(len(states), temporal_safety_derivative)
        taugrad = np.zeros((len(states), 3))
    result = scale[:, None] * taugrad + tau[:, None] * scalegrad
    if not np.isfinite(result).all():
        raise ValueError("nonfinite scaled temporal gradient")
    return np.ascontiguousarray(result)
