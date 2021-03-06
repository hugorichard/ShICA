import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

from parameters import COLORS, NAMES

os.makedirs("../figures", exist_ok=True)

rc = {
    "pdf.fonttype": 42,
    "text.usetex": True,
    "font.size": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "text.latex.preview": True,
    "font.family": "serif",
}
plt.rcParams.update(rc)


num_points = 4
seeds = np.arange(10)
ns = np.logspace(2, 4, num_points)
res = np.zeros((len(seeds), 4, len(ns)))
seed = 0

res = load("../results/full_nongaussian_res")
plt.figure(figsize=(6, 3))
for i, algo in enumerate(["shica_j", "multisetcca", "canica", "shica_ml"]):
    plt.plot(
        ns,
        np.median(res[:, i], axis=0),
        color=COLORS[algo],
        label=NAMES[algo],
    )
    plt.fill_between(
        ns,
        np.quantile(res[:, i], 0.01, axis=0),
        np.quantile(res[:, i], 0.99, axis=0),
        color=COLORS[algo],
        alpha=0.3,
    )
plt.xscale("log")
plt.ylabel("Amari distance")
plt.xlabel("Number of samples")
plt.yscale("log")
plt.legend(ncol=2, bbox_to_anchor=(0.45, 1.4), loc="upper center")
plt.savefig("../figures/full_nongaussian.png", bbox_inches="tight")
