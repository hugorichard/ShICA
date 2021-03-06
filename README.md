# ShICA

[![CircleCI](https://circleci.com/gh/hugorichard/ShICA.svg?style=svg)](https://circleci.com/gh/hugorichard/ShICA)

Code accompanying the paper Shared Independent Component Analysis for Multi-subject Neuroimaging

## Install 
Move into the ShICA directory
``cd ShICA``

Install ShICA
``pip install -e .``

## Reproduce synthetic experiments in Figure 2
Move into the experiments directory
``cd experiments``

Run the bash script to produce results (should take approximately 3 minutes on a modern laptop)
``bash run_all.bash``

Move into the plotting directory
``cd plotting``

Run the bash script to produce figures from the results
``bash plot_all.bash``

Figures are available in the ``figures`` directory.

Performances on Gaussian sources:

![Full non Gaussian](./figures/rotation.png)

Performances on non Gaussian sources:

![Full Gaussian](./figures/full_nongaussian.png)

Performances when some sources are Gaussian and some non-Gaussian:

![Semi Gaussian](./figures/semigaussian.png)


__Note__
The current implementation uses only 10 seeds and 4 different number of samples in the curves so that computation time is low even on a laptop. In order to obtain exactly the same curves as in the paper you should modify the files `rotation.py`, `full_nongaussian.py` and `semigaussian.py` in the `experiments` directory so that 
```
num_points = 20
seeds = np.arange(40)
ns = np.logspace(2, 5, num_points)
```

## Real data experiments

We give the code to run experiments on timesegment matching.

#### Download and mask Sherlock data

Move into the data directory

``cd experiments/data``

Launch the download script (Runtime ``34m6.751s``)

`` bash download_data.sh ``

Mask the data (Runtime ``15m27.104s``)

``python mask_data.py``


#### Timesegment matching

Move into the `experiments` directory

``cd experiments``

Run the experiment on masked data (Runtime ``17m39.520s``)

``python timesegment_matching.py``

![Timesegment matching](./figures/timesegment_matching.png)

This runs the experiment with ``n_components = 5`` and benchmark `ShiCA-J` and `ShICA-ML` with `SRM` as the dimension reduction method.

Documentation
--------------

https://hugorichard.github.io/ShICA/index.html

Cite
--------------
If you use this code in your project, please cite:
```
@inproceedings{NEURIPS2021_fb508ef0,
 author = {Richard, Hugo and Ablin, Pierre and Thirion, Bertrand and Gramfort, Alexandre and Hyvarinen, Aapo},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {29962--29971},
 publisher = {Curran Associates, Inc.},
 title = {Shared Independent Component Analysis for Multi-Subject Neuroimaging},
 url = {https://proceedings.neurips.cc/paper/2021/file/fb508ef074ee78a0e58c68be06d8a2eb-Paper.pdf},
 volume = {34},
 year = {2021}
}
```
