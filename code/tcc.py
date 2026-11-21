import numpy as np


def tcc(phi, coefficient, observable, groups, lags):
    """Return TCC scores and descending group indices for equal spatial weights.

    phi: Spatial modes, shape (M, r).
    coefficient: Full-set temporal coefficients, shape (r, N_raw).
    observable: Real-valued observable aligned with snapshots, shape (N_raw,).
    groups: Column indices of conjugate mode pairs.
    lags: Nonnegative integer lag indices in ascending order.
    """
    indices = np.arange(observable.size - lags[-1])
    lagged_y = observable[indices[None, :] + lags[:, None]]
    lagged_y = lagged_y - lagged_y.mean(axis=1, keepdims=True)
    cross_response = coefficient[:, indices] @ lagged_y.T
    cross_response /= indices.size * np.sqrt(lags.size)

    scores = np.zeros(len(groups))
    for g, members in enumerate(groups):
        response = phi[:, members] @ cross_response[members]
        scores[g] = np.sum(np.abs(response) ** 2)

    order = np.argsort(-scores)
    return scores, order
