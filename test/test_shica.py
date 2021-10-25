import numpy as np
from shica import shica_ml, shica_j
from shica.exp_utils import amari_d
import matplotlib.pyplot as plt

m, p, n = 4, 10, 1000
S = np.random.randn(p, n)
A = np.random.randn(m, p, p)
N = np.random.randn(m, p, n)
powers = np.random.rand(m, p)
X = np.array([a.dot(S + p[:, None] * n) for p, a, n in zip(powers, A, N)])


def test_shicaj():
    W_pred, Sigmas, S = shica_j(X)
    for w_pred, a in zip(W_pred, A):
        assert amari_d(w_pred, a) < 0.1
