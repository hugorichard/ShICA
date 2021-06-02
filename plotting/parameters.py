from matplotlib.cm import get_cmap

vir = get_cmap("Set1", 9)

COLORS = {
    # "permica": cmb(0),
    "shica_j": vir(1),
    "shica_ml": vir(2),
    "canica": vir(4),
    "multisetcca": vir(6),
    "Nothing": "black",
}


hue_matching = {
    "CanICA": "canica",
    "ShICA-ML": "shica_ml",
    "ShICA-J": "shica_j",
    "Multiset CCA": "multisetcca",
}

plot_ordering = [
    "shica_ml",
    "shica_j",
    "canica",
    "multisetcca",
]
NAMES = {v: k for k, v in hue_matching.items()}
