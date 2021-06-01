# ShICA
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
``bash run_all.bash ``

Move into the plotting directory
`` cd plotting ``

Run the bash script to produce figures from the results
``bash plot_all.bash ``

Figures are available in the ``figures`` directory.

Performances on Gaussian sources:

![Full non Gaussian](./figures/rotation.png)

Performances on non Gaussian sources:

![Full Gaussian](./figures/full_nongaussian.png)

Performances when some sources are Gaussian and some non-Gaussian:

![Semi Gaussian](./figures/semigaussian.png)
