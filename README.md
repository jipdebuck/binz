# binz
A redshift distribution reconstruction package using Gaussian processes and normalising flows to construct priors on $n(z)$ hyperparameters for cosmological inference.

## Overview
`binz` is a package designed for on-the-fly $n(z)$ reconstruction in Stage-IV photometric cosmological inference pipelines, specifically Euclid. It consists of two main ingredients:
1. **Gaussian processes** (GPs) to reconstruct per-bin redshift distributions in a flexible manner. Parameters associated with the GP are the training nodes and the correlation length, which can be customized by the user.
2. **Normalising flows** to construct a data-driven prior on the GP hyperparameters from an ensemble of redshift distributions. `binz` also includes functions to generate such an ensemble based on simple shifts and stretches in the original $n(z)$.

## Installation
To install `binz`, first clone the repository to your local machine:
```
git clone https://github.com/jipdebuck/binz.git
```
Then, from the root directory of `binz`:
```
pip install .
```

## Usage
Running `binz` starts with the tomographic redshift distributions $n(z)$, after which a typical workflow is the following: 
1. **Generate the ensemble** from the original $n(z)$ (if this is not an ensemble already):
```python
  from binz.ensemble import generate_bin1_ensemble
  z_ens_list, nz_ensemble_list = zip(*[generate_bin1_ensemble(N_ens=200, nz=dndz_she_norm[i], N_z=3001) for i in range(1, 7)])
  # This generate an ensemble with 200 realisations based on the original $n(z)$ in dndz_she_norm with 6 tomographic bins
```
2. **Fit GP hyperparameters** to the ensemble:
```python
   from binz.ensemble import fit_params_ens
   GP_params, fit_results = fit_params_ens(z_grid=z_grid, nz_ens=nz_ens, M=3)
   # This fits GP hyperparameters with M training nodes to a single realisation in the ensemble nz_ens
```
3. **Train a normalising flow** to the resulting GP hyperparameter distributions:
```python
   from binz.norm_flow import norm_flow_per_bin
   flow_b, mean_b, std_b = norm_flow_per_bin(GP_params_single_bin, n_layers=5, hidden_features=32)
   # This trains the flow on the GP params in a single bin with 5 layers and 32 hidden features
```
4. **Use the trained flow during cosmological inference** to evaluate the log probability of the current set of proposed GP hyperparameters:
```python
   from binz.norm_flow import get_flow_log_prob
   log_prob = get_flow_log_prob(flow, mean, std, x_nodes, y_nodes, ell)
   # This returns the log probability according to the flow based on the current set of x nodes, y nodes and the correlation length
```
**Note**: The normalising flow provides a data-driven prior on the GP hyperparameters, but the GP-based $n(z)$ reconstruction can also be used independently. For more detailed examples, see the example notebooks in the `notebooks` folder. 
   


