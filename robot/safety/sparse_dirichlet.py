"""Direct solve of the existing five-point Dirichlet equations, opt-in.

Every call rebuilds the operator for its exact mask. Public solvers retain
their independent equation-residual gates and legacy default backends.
"""

import numpy as np


def solve_five_point_dirichlet(values, fixed_mask, *, forcing, resolution):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu

    result = np.array(values, dtype=float, copy=True)
    scalar = result.ndim == 2
    if scalar:
        result = result[:, :, None]
    free = ~fixed_mask
    rows, columns = np.nonzero(free)
    count = len(rows)
    if not count:
        return result[:, :, 0] if scalar else result
    indices = np.full(free.shape, -1, dtype=np.int64)
    indices[free] = np.arange(count)
    rhs = (
        -np.broadcast_to(np.asarray(forcing)[:, :, None], result.shape)[free]
        * resolution**2
    )
    matrix_rows = [np.arange(count)]
    matrix_columns = [np.arange(count)]
    entries = [np.full(count, 4.0)]
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = rows + dy, columns + dx
        neighboring_free = free[nr, nc]
        matrix_rows.append(np.nonzero(neighboring_free)[0])
        matrix_columns.append(indices[nr[neighboring_free], nc[neighboring_free]])
        entries.append(np.full(np.count_nonzero(neighboring_free), -1.0))
        fixed_neighbors = ~neighboring_free
        rhs[fixed_neighbors] += result[nr[fixed_neighbors], nc[fixed_neighbors]]
    matrix = coo_matrix(
        (
            np.concatenate(entries),
            (np.concatenate(matrix_rows), np.concatenate(matrix_columns)),
        ),
        shape=(count, count),
    ).tocsc()
    result[free] = splu(matrix).solve(rhs)
    return result[:, :, 0] if scalar else result
