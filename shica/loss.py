import numpy as np


def loss_total(basis_list, Y_list, l, sigmas, a, b, eps2, check_l=True):
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
            np.sum(l ** 2 + eps2, axis=0), np.ones(p)
        )

    Y_avg = np.sum(
        [(li.reshape(-1, 1) ** 2 + eps2) * y for li, y in zip(l, Y_list)],
        axis=0,
    )
    loss = 0
    loss += np.mean(f(Y_avg, sigmas, m, a, b))
    for j in range(p):
        loss += (m - 1) / 2 * np.log(sigmas[j] ** 2)
    for i, (W, Y) in enumerate(zip(basis_list, Y_list)):
        loss -= np.linalg.slogdet(W)[1]
        for j in range(p):
            loss -= 1 / 2 * np.log(l[i][j] ** 2 + eps2)
            loss += (
                1
                / (2 * sigmas[j] ** 2)
                * np.mean((Y[j] - Y_avg[j]) ** 2)
                * m
                * (l[i][j] ** 2 + eps2)
            )
    return loss


def loggauss(X, sigma):
    var = sigma ** 2
    return -(X ** 2) / (2 * var) - 0.5 * np.log(2 * np.pi * var)


def f(Y, sigmas, n_subjects, a, b):
    return np.sum(
        [
            numerical_fi(Yk, sigma, n_subjects, a, b)
            for Yk, sigma in zip(Y, sigmas)
        ],
        axis=0,
    )


def numerical_fi(Y, sigma, n_subjects, a, b):
    noise = sigma ** 2
    m = n_subjects
    res = -np.logaddexp(
        loggauss(Y, np.sqrt(a + noise / m)),
        loggauss(Y, np.sqrt(b + noise / m)),
    ) + np.log(2)
    return res
