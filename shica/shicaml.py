import numpy as np
import warnings
from amvica.loss import loss_total, loggauss
import scipy.stats
from shica import shica_j


def plugin_noise(Xs):
    m = len(Xs)
    return (
        1
        / (m - 1)
        * np.mean(
            np.sum(Xs ** 2, axis=0) - 1 / m * np.sum(Xs, axis=0) ** 2, axis=1
        )
    )


def f_gauss(Y, sigmas, n_subjects):
    return np.sum(
        [fi_gauss(Yk, sigma, n_subjects) for Yk, sigma in zip(Y, sigmas)],
        axis=0,
    )


def fi_gauss(Y, sigma, n_subjects):
    res = -loggauss(Y, np.sqrt(sigma ** 2 / n_subjects + 1))
    return res


def loss_total_gauss(basis_list, Y_list, l, sigmas, check_l=True):
    m, p, p3 = basis_list.shape
    m2, p2, n = Y_list.shape
    m3, p4 = l.shape
    (p5,) = sigmas.shape
    assert m == m2
    assert m == m3
    assert p == p2
    assert p == p3
    assert p == p4
    assert p == p5
    if check_l:
        np.testing.assert_array_almost_equal(
            np.sum(l ** 2, axis=0), np.ones(p)
        )

    Y_avg = np.sum(
        [(li.reshape(-1, 1) ** 2) * y for li, y in zip(l, Y_list)], axis=0,
    )
    loss = 0
    loss += np.mean(f_gauss(Y_avg, sigmas, m))
    for j in range(p):
        loss += (m - 1) / 2 * np.log(sigmas[j] ** 2)
    for i, (W, Y) in enumerate(zip(basis_list, Y_list)):
        loss -= np.linalg.slogdet(W)[1]
        for j in range(p):
            loss -= 1 / 2 * np.log(l[i][j] ** 2)
            loss += (
                1
                / (2 * sigmas[j] ** 2)
                * np.mean((Y[j] - Y_avg[j]) ** 2)
                * m
                * (l[i][j] ** 2)
            )
    return loss


def mmse(source, noise, a, b, m):
    n_components = source.shape[0]
    S = []
    for j in range(n_components):
        num = 0
        denum = 0
        for n in [a, b]:
            norm = scipy.stats.norm(0, np.sqrt(n + noise[j] ** 2 / m))
            denum += norm.pdf(source[j])
            coef = n * source[j] / (n + noise[j] ** 2 / m)
            num += norm.pdf(source[j]) * coef
        S.append(num / denum)
    return np.array(S)


def mmse_gauss(source, noise, m):
    n_components = source.shape[0]
    S = []
    for j in range(n_components):
        S.append(m * source[j] / (m + noise[j] ** 2))
    return np.array(S)


def var_s(source, noise, a, b, m):
    n_components = source.shape[0]
    S = []
    for j in range(n_components):
        num = 0
        denum = 0
        for n in [a, b]:
            norm = scipy.stats.norm(0, np.sqrt(n + noise[j] ** 2 / m))
            denum += norm.pdf(source[j])
            coef = n * noise[j] ** 2 / m / (n + noise[j] ** 2 / m)
            num += norm.pdf(source[j]) * coef
        S.append(num / denum)
    return np.array(S)


def var_s_gauss(source, noise, m):
    n_components, n_timeframes = source.shape
    S = []
    for j in range(n_components):
        S.append(noise[j] ** 2 / (m + noise[j] ** 2) * np.ones(n_timeframes))
    return np.array(S)


def Sigma_to_sigma_lambda(Sigma, eps2=0):
    """
    Parameters
    Sigma: shape (m, k)
    Returns
    sigma: shape (k)
    l: shape (m, k)
    l-parameter ready for gradient computation
    """
    m, k = Sigma.shape
    sigma = m / np.sum(1 / Sigma, axis=0)
    l = sigma / (Sigma * m)
    sigma = np.sqrt(sigma)
    l = l - eps2
    if not (l > 0).all():
        print(l)
    assert (l > 0).all()
    l = np.sqrt(l)
    return sigma, l


def sigma_lambda_to_Sigma(sigma, l, eps2=0):
    """
    Parameters
    Sigma: shape (m, k)
    Returns
    sigma: shape (m, k)
    l: shape (m, k)
    l-parameter ready for gradient computation
    """
    m = len(l)
    return sigma ** 2 / (m * (l ** 2 + eps2))


def update_Sigmai(Yi, Es, Vars):
    """
    Return new Sigma_i: shape k
    """
    return np.mean((Yi - Es) ** 2, axis=1) + np.mean(Vars, axis=1)


def loss_Wi(Wi, Xi, Yi, Sigmai, Es, Vars):
    k, n = Yi.shape
    loss = 0
    loss -= np.linalg.slogdet(Wi)[1]
    loss += 0.5 * (np.sum(((Yi - Es) ** 2 + Vars) / Sigmai.reshape(-1, 1))) / n
    loss += 0.5 * np.sum(np.log(Sigmai))
    return loss


def grad_Wi(Xi, Yi, Sigmai, Es):
    k, n = Xi.shape
    grad = -np.eye(k)
    grad += 1 / Sigmai.reshape(-1, 1) * ((Yi - Es).dot(Yi.T)) / n
    return grad


def hess_Wi(Xi, Yi, Sigmai, Es):
    p, n = Yi.shape
    sigmas2 = Sigmai
    return (
        np.dot(np.ones_like(Yi), (Yi ** 2).T,) * 1 / sigmas2.reshape(-1, 1) / n
    )


def update_Wi(
    Wi,
    Xi,
    Yi,
    Sigmai,
    Es,
    Vars,
    lambda_min=1e-4,
    n_ls_tries=20,
    it=0,
    info=None,
):
    loss0 = loss_Wi(Wi, Xi, Yi, Sigmai, Es, Vars)
    G = grad_Wi(Xi, Yi, Sigmai, Es)
    h = hess_Wi(Xi, Yi, Sigmai, Es)

    discr = np.sqrt((h - h.T) ** 2 + 4.0)
    eigenvalues = 0.5 * (h + h.T - discr)
    problematic_locs = eigenvalues < lambda_min
    np.fill_diagonal(problematic_locs, False)
    i_pb, j_pb = np.where(problematic_locs)
    h[i_pb, j_pb] += lambda_min - eigenvalues[i_pb, j_pb]
    # Compute Newton's direction
    det = h * h.T - 1
    direction = (h.T * G - G.T) / det
    # Line search
    step = 1
    for j in range(n_ls_tries):
        new_Wi = Wi - step * direction.dot(Wi)
        new_Yi = new_Wi.dot(Xi)
        new_loss = loss_Wi(new_Wi, Xi, new_Yi, Sigmai, Es, Vars)
        if new_loss < loss0:
            return new_Wi
        else:
            step /= 2.0

    step = 1
    for j in range(n_ls_tries):
        new_Wi = Wi - step * G.dot(Wi)
        new_Yi = new_Wi.dot(Xi)
        new_loss = loss_Wi(new_Wi, Xi, new_Yi, Sigmai, Es, Vars)
        if new_loss < loss0:
            return new_Wi
        else:
            step /= 2.0
    return np.zeros_like(Wi)


def shica_ml(
    X_list,
    max_iter=1000,
    init="shica_j",
    W_init=None,
    Sigmas_init=None,
    tol=1e-5,
    verbose=False,
):
    """
    Parameters
    ----------
    X_list : np array of shape (m, k, n)
        input data

    max_iter: int
        Maximum number of iterations to perform

    init: None or "shica_j"
        If "shica_j" uses shica_j to initialize
        unmixing matrices and noise covariance matrices.
        Parameters `W_init` and `Sigmas_init` are ignored.
        If `None` parameters `W_init` and `Sigmas_init` are used
        to initialize unmixing matrices and noise covariance matrices

    W_init : np array of shape (m, k, k)
        Initial unmixing matrices

    Sigmas_init : np array of shape (m, k)
        Initial noise covariances

    tol: float
        Tolerance. The algorithm stops when the loss
        decrease is below this value.

    verbose : bool
        If True, prints information about convergence

    Returns
    -------
    W_list : np array of shape (m, k, k)
        Unmixing matrices

    Sigmas: np array of shape (k,)
        Noise covariances

    Y_avg: np array of shape (k, n)
        Source estimates
    """
    m, k, n = X_list.shape
    if Sigmas_init is None and init is None:
        Sigmas = np.array([np.eye(k) for _ in range(m)])
    else:
        Sigmas = Sigmas_init

    if W_init is not None and init is None:
        W_list = W_init.copy()
    else:
        W_list = np.random.randn(m, k, k)

    if init == "shica_j":
        W_list, Sigmas, _ = shica_j(X_list)

    sigmas, l_list = Sigma_to_sigma_lambda(Sigmas)
    l_list = np.array(l_list)

    Y_list = np.array([W.dot(X) for W, X in zip(W_list, X_list)])
    Y_avg = np.sum(
        [
            (l.reshape(-1, 1) ** 2) * w.dot(x)
            for w, l, x in zip(W_list, l_list, X_list)
        ],
        axis=0,
    )
    loss_prec = 0
    dl = 0
    for it in range(1, max_iter + 1):
        # E_step
        Es = mmse(Y_avg, sigmas, 0.5, 1.5, m)
        Vars = var_s(Y_avg, sigmas, 0.5, 1.5, m)
        Sigma = sigma_lambda_to_Sigma(sigmas, l_list)
        loss0 = loss_total(W_list, Y_list, l_list, sigmas, 0.5, 1.5, 0)

        # M_step
        convergence = False
        for i in range(m):
            Sigma_i = update_Sigmai(Y_list[i], Es, Vars)
            Wi = update_Wi(
                W_list[i],
                X_list[i],
                Y_list[i],
                Sigma_i,
                Es,
                Vars,
                it=it,
                info=dl,
            )
            if (Wi == 0).all():
                continue
            else:
                convergence = True
            Y_list[i] = Wi.dot(X_list[i])
            W_list[i] = Wi
            Sigma[i] = Sigma_i
            assert (Sigma_i > 0).all()
        if not convergence:
            warnings.warn(
                "AVICA EM did not converge. Loss decrease is %.20e" % dl
            )
            break

        sigmas, l_list = Sigma_to_sigma_lambda(Sigma)
        loss0 = loss_total(W_list, Y_list, l_list, sigmas, 0.5, 1.5, 0)
        dl = np.abs(loss_prec - loss0)
        if np.abs(loss_prec - loss0) < tol:
            break
        if verbose:
            print(
                "it %i loss %.4f diff %.10f" % (it, loss0, loss_prec - loss0)
            )
        loss_prec = loss0
        Y_avg = np.sum(
            [
                (l.reshape(-1, 1) ** 2) * w.dot(x)
                for w, l, x in zip(W_list, l_list, X_list)
            ],
            axis=0,
        )
    Y_avg = np.sum(
        [
            (l.reshape(-1, 1) ** 2) * w.dot(x)
            for w, l, x in zip(W_list, l_list, X_list)
        ],
        axis=0,
    )
    Sigmas = sigma_lambda_to_Sigma(sigmas, l_list)
    return W_list, Sigmas, Y_avg
