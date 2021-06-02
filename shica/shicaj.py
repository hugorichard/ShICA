import numpy as np
from scipy import linalg
from qndiag import qndiag


def update_Sigmai(covY, Sigmas, k):
    m = len(covY)
    V = 1 / (1 + np.sum(1 / Sigmas, axis=0))
    return (
        covY[k, k]
        - 2
        * V
        * np.sum([1 / Sigmas[i] * covY[i, k] for i in range(m)], axis=0)
        + V ** 2
        * np.sum(
            [
                1 / Sigmas[i] * np.sum(covY[i, :] * 1 / Sigmas, axis=0)
                for i in range(m)
            ],
            axis=0,
        )
        + V
    )


def minimize_i(D, CY, i):
    m, p = D.shape
    D_old = np.copy(D[i])
    CD2C = np.sum(
        [CY[i, j] * D[j] ** 2 * CY[j, i] for j in range(m) if j != i], axis=0,
    )
    DC = np.sum([CY[i, j] * D[j] for j in range(m) if j != i], axis=0)
    Di = DC / CD2C
    return Di, np.max(np.abs(Di - D_old))


def loss(D, CY):
    m, k = D.shape
    l = 0
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            l += np.sum((D[i] * CY[i, j] * D[j] - np.ones(k)) ** 2)
    return l


def grad(D, CY):
    m, k = D.shape
    G = []
    for i in range(m):
        Gij = np.zeros_like(D[i])
        for j in range(m):
            if i == j:
                continue
            Gij += (D[i] * CY[i, j] * D[j] - np.ones(k)) * CY[i, j] * D[j]
        G.append(Gij)
    return np.array(G)


def global_min(D_init, CY, n_iters=100, verbose=False, tol=1e-4):
    D_list = np.copy(D_init)
    m, _ = D_list.shape
    for i in range(n_iters):
        norm = 0
        for j in range(m):
            D_list[j], g_norm = minimize_i(D_list, CY, j)
            norm = np.max((g_norm, norm))
        if verbose:
            print(
                "it % i, loss %.2e, gradient %.2e"
                % (i + 1, loss(D_list, CY), np.max(np.abs(grad(D_list, CY))))
            )
        if norm < tol:
            break
    if norm >= tol:
        print(
            "WARNING scaling optimization did not converge."
            "Gradient norm is %.5f" % np.max(np.abs(grad(D_list, CY)))
        )
    return D_list


def shica_j(
    X_list,
    max_iter=1000,
    tol=1e-5,
    use_jointdiag=True,
    use_scaling=True,
    verbose=False,
):
    """

    Parameters
    ----------
    X_list : ndarray of shape (m, k, n)
        input data

    max_iter: int
        Maximum number of iterations to perform

    tol: float
        Tolerance. The algorithm stops when the loss
        decrease is below this value.

    verbose : bool
        If True, prints information about convergence

    Returns
    -------
    W_list : ndarray of shape (m, k, k)
        Unmixing matrices

    Sigmas: ndarray of shape (k,)
        Noise covariances

    Y_avg: ndarray of shape (k, n)
        Source estimates
    """
    m, k, n = X_list.shape
    # Compute cross covariance
    Cred = (
        np.transpose(X_list.dot(np.transpose(X_list, (0, 2, 1))), (0, 2, 1, 3))
        / n
    )
    C = np.concatenate(np.concatenate(Cred, axis=1), axis=1)
    D = linalg.block_diag(*[Cred[i, i] for i in range(m)])
    _, A = linalg.eigh(C, D, subset_by_index=[m * k - k, m * k - 1])
    A = np.array(np.split(A, m, axis=0))
    W_list = np.array([a.T for a in A])

    if use_jointdiag:
        Ds = []
        for i in range(m):
            Yi = W_list[i].dot(X_list[i])
            Di = Yi.dot(Yi.T)
            Ds.append(Di)

        B, _ = qndiag(np.array(Ds), max_iter=max_iter, verbose=verbose,)
        W_list = np.array([B.dot(w) for w in W_list])

    if use_scaling:
        slices = [slice(i * k, (i + 1) * k) for i in range(m)]
        CY = np.zeros((m, m, k))
        for i in range(m):
            for j in range(m):
                CY[i, j, :] = np.diag(
                    W_list[i].dot(C[slices[i], slices[j]]).dot(W_list[j].T)
                )
        D = global_min(np.ones((m, k)), CY, n_iters=max_iter, verbose=verbose)

        for i in range(m):
            for j in range(m):
                CY[i, j, :] = D[i] * CY[i, j] * D[j]
        Sigmas = np.ones((m, k))
        for it in range(max_iter):
            sprec = np.copy(Sigmas)
            for i in range(m):
                Sigmai = update_Sigmai(CY, Sigmas, i)
                Sigmas[i] = Sigmai
            if np.max(np.abs(sprec - Sigmas)) < tol:
                break
        W_list = np.array([d[:, None] * w for d, w in zip(D, W_list)])
        Ys = np.array([w.dot(x) for w, x in zip(W_list, X_list)])
    else:
        Ys = np.array([w.dot(x) for w, x in zip(W_list, X_list)])
        Sigmas = np.ones((m, k))

    if use_scaling:
        Y_avg = (
            1
            / (np.sum(1 / Sigmas, axis=0) + 1).reshape(-1, 1)
            * np.sum(
                [1 / Sigmas[i].reshape(-1, 1) * Ys[i] for i in range(m)],
                axis=0,
            )
        )
    else:
        Y_avg = np.sum(Ys, axis=0)
    return W_list, Sigmas, Y_avg
