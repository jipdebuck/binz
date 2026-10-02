# binz
A redshift distribution reconstruction package using Gaussian processes and normalising flows to construct priors on $n(z)$ hyperparameters for cosmological inference.

## Overview
`binz` is a package designed for on-the-fly $n(z)$ reconstruction in Stage-IV photometric cosmological inference pipelines, specifically Euclid. It consists of two main ingredients:
1. **Gaussian processes** (GPs) to reconstruct per-bin redshift distributions in a flexible manner. Parameters associated with the GP are the training nodes and the correlation length, which can be customized by the user.
2. **Normalising flows** to construct a data-driven prior on the GP hyperparameters from an ensemble of redshift distributions. `binz` also includes functions to generate such an ensemble based on simple shifts and stretches in the original $n(z)$.

## Installation

