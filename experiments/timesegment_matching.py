"""
==============================
Time-segment matching
==============================


"""


# Authors: Hugo Richard, Pierre Ablin
# License: BSD 3 clause

#%%
import numpy as np
from sklearn.model_selection import KFold

from fmri_utils import load_and_concat
from timesegment_matching_utils import time_segment_matching
import os
import matplotlib.pyplot as plt
from plot_utils import confidence_interval
from shica import shica_j, shica_ml
from shica.shicaml import Sigma_to_sigma_lambda
from fastsrm.identifiable_srm import IdentifiableFastSRM


n_subjects = 17
n_runs = 5
n_components = 5

n_subjects = 17
n_runs = 5
paths = np.array(
    [
        [
            os.path.join("data", "masked_movie_files", "sub%i_run%i.npy" % (i, j))
            for j in range(n_runs)
        ]
        for i in range(n_subjects)
    ]
)

algos = [
    ("ShICA-J", lambda x: shica_j(x)),
    ("ShICA-ML", lambda x: shica_ml(x, max_iter=10000)),
]

cv = KFold(n_splits=5, shuffle=False)
res = []
for i, (train_runs, test_runs) in enumerate(cv.split(np.arange(n_runs))):
    train_paths = paths[:, train_runs]
    test_paths = paths[:, test_runs]
    data_test = load_and_concat(test_paths)
    data_train = load_and_concat(train_paths)
    res_ = []
    for name, algo in algos:
        print(name)
        srm = IdentifiableFastSRM(
            n_components=n_components,
            tol=1e-4,
            verbose=True,
            aggregate=None,
            n_iter=1000,
        )
        X = np.array(srm.fit_transform([x for x in data_train]))
        K = np.array([w.T for w in srm.basis_list])
        W, Sigmas, S = algo(X)
        sigmas, l_list = Sigma_to_sigma_lambda(Sigmas)
        forward = [W[i].dot(K[i]) for i in range(n_subjects)]
        backward = [np.linalg.pinv(forward[i]) for i in range(n_subjects)]
        shared = np.array(
            l_list[i].reshape(-1, 1) ** 2
            / sigmas.reshape(-1, 1)
            * [forward[i].dot(data_test[i]) for i in range(n_subjects)]
        )
        cv_scores = time_segment_matching(shared, win_size=9)
        res_.append(cv_scores)
    res.append(res_)

# %%

# Plotting
cm = plt.cm.tab20

algos = [
    ("ShICA-J", cm(0)),
    ("ShICA-ML", cm(7)),
]

res = np.array(res)

fig, ax = plt.subplots()
for i, (algo, color) in enumerate(algos):
    res_algo = res[:, i, :].flatten()
    av = np.mean(res_algo)
    low, high = confidence_interval(res_algo)
    low = av - low
    high = high - av
    ax.bar(
        i,
        height=[av],
        width=0.8,
        label=algo,
        color=color,
        yerr=np.array([[low], [high]]),
    )
plt.ylabel(r"Accuracy")
plt.xticks([0, 1], ["ShICA-J", "ShICA-ML"])
fig.legend(
    ncol=3,
    loc="upper center",
)
plt.savefig(
    "../figures/timesegment_matching.png",
    bbox_inches="tight",
)
