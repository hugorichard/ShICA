import numpy as np
from mvlearn.decomposition import GroupICA
from joblib import Parallel, delayed, dump
from shica.exp_utils import amari_d
from shica import shica_j, shica_ml
import os

os.makedirs("../results", exist_ok=True)


def do_exp(m, k, n, seed):
    n = int(n)
    rng = np.random.RandomState(seed)
    S = np.zeros((k, n))
    S[: int(k / 2)] = rng.laplace(size=(int(k / 2), n))
    S[int(k / 2) :] = rng.randn(int(k / 2), n)
    sigmas = 1 * rng.randn(m, k, n)
    powers = rng.rand(m, k)
    powers[:, : int(k / 2)] = np.ones((m, int(k / 2)))
    sigmas = np.array([p.reshape(-1, 1) * e for p, e in zip(powers, sigmas)])
    A = np.array([rng.randn(k, k) for _ in range(m)])
    X = np.array([a.dot(S + eps) for a, eps in zip(A, sigmas)])
    # Shica J
    W, _, _ = shica_j(X, use_scaling=False)
    # Multiset CCA
    W2, _, _ = shica_j(X, use_jointdiag=False, use_scaling=False)
    # CanICA
    W3 = np.array(
        GroupICA(prewhiten=True).fit([x.T for x in X]).individual_components_
    )
    # ShICA ML
    W4, _, _ = shica_ml(X, init="shica_j",)

    res = np.mean([amari_d(W[i], A[i]) for i in range(m)])
    res2 = np.mean([amari_d(W2[i], A[i]) for i in range(m)])
    res3 = np.mean([amari_d(W3[i], A[i]) for i in range(m)])
    res4 = np.mean([amari_d(W4[i], A[i]) for i in range(m)])
    return res, res2, res3, res4


num_points = 4
seeds = np.arange(10)
ns = np.logspace(2, 4, num_points)
res = np.zeros((len(seeds), 4, len(ns)))
seed = 0

k = 4
m = 5


res_all = Parallel(n_jobs=-1)(
    delayed(do_exp)(m, k, n, seed)
    for i, n in enumerate(ns)
    for j, seed in enumerate(seeds)
)

res_all = np.array(res_all).reshape((len(ns), len(seeds), -1))
for i, n in enumerate(ns):
    for j, seed in enumerate(seeds):
        res[j, :, i] = res_all[i, j]
dump(res, "../results/semigaussian_res")
