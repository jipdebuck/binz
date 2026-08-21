# General imports

import os
os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from copy import deepcopy
import time
from mpi4py.futures import MPIPoolExecutor

import dynesty
from dynesty.pool import Pool as DynestyPool
from scipy.stats import norm

# Import euclidlib for reading the Euclid data
import euclidlib as el

# Import cloelib for cosmology and theoretical predictions
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations

# Import cloelike for likelihoods
from cloelike.EuclidLikelihood_photo_Cls import EuclidLikelihood_3x2pt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ConstantKernel as C
)

import torch 
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform
torch.set_num_threads(1)

# We read with euclidlib v2025.2 (v2025.1 is also compatible)

name_to_chain = '3x2pt_LCDM_DR1_GP_3_flow'

#cells_data = el.photo.harmonic_space.angular_power_spectra('/disks/shear16/canasherrera/SheariouslyCalibrated/data/DR1/synthetic_angular_spectra_DR1.fits')
full_cov=np.load('/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/data/covariance_Gauss_DR1_TR1like_1589_2D.npz')['Gauss']

# Get n(z)
z_nz, nz_heracles = el.phz.redshift_distributions('/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/data/nz_example.fits')

# Normalize and resample n(z) for both position and shear
myz = np.linspace(1e-4, 3.0, 100)

def normalize_dndz(nz_example, z_nz):
    normalized = {}
    for key in nz_example:
        normalized[key] = nz_example[key] / integrate.trapezoid(nz_example[key], z_nz)
    return normalized

def resample_dndz(nz_example, z_nz, myz):
    my_dndz = np.vstack(list(nz_example.values())) #stacking arrays vertically
    my_dndz_norm = np.zeros([len(nz_example), len(myz)]) #creating an empty array
    for i in range(len(nz_example)):
        my_dndz_norm[i, :] = np.interp(myz, z_nz, my_dndz[i, :]) #interpolating to new myz values
    return my_dndz_norm

dndz_pos_norm = normalize_dndz(nz_heracles, z_nz)
dndz_she_norm = normalize_dndz(nz_heracles, z_nz)
my_dndz_pos_norm = resample_dndz(nz_heracles, z_nz, myz)
my_dndz_she_norm = resample_dndz(nz_heracles, z_nz, myz)

### Preparation for Gaussian processes ###

# Define fixed GP hyperparameters 
z_bounds = (0,3)                              # bounds for the redshift
z_pred = np.linspace(0, 3, 400)       # prediction points for GP
alpha = 1e-8                                  # noise level during fitting 
kernel_class = RBF
nu = 1.5                                      # necessary for Matern only, defines smoothness 
const_value = 0.04                            # amplitude of constant kernel
const_bounds = (1e-4, 1.0)                    # bounds for constant kernel amplitude
z_pred_res = z_pred.reshape(-1, 1)

# GP functions

def compute_cdf_nodes(z_grid, nz_org, M=3, qmin=0.05, qmax=0.95):
    X_nodes = []
    Y_nodes = []   
    for key, nz in nz_org.items():
        cdf = integrate.cumulative_trapezoid(nz, z_grid, initial=0)
        cdf /= cdf[-1]
        z_peak = z_grid[np.argmax(nz)] # one node always at the peak
        if M > 1: # others at quantiles
            quantiles = np.linspace(qmin, qmax, M)
            x_nodes = np.interp(quantiles, cdf, z_grid)
            x_nodes[M//2] = z_peak
        else:
            x_nodes = np.array([z_peak])
        x_nodes = np.sort(x_nodes)
        y_nodes = np.interp(x_nodes, z_grid, nz)
        X_nodes.append(x_nodes)
        Y_nodes.append(y_nodes)
    return X_nodes, Y_nodes

def x_log1p(z, zmax):
    """Map z to x in [0,1] using log(1+z)."""
    return np.log1p(z) / np.log1p(zmax)

def kernel_fixed(kernel_class, amp, ell, nu=1.5, amp_bounds=(1e-6, 10.0)):
    """
    Kernel with fixed length-scale ell, fixed (or bounded) amplitude amp.
    We keep optimizer=None, so amp and ell are not learned during GP fitting.
    """
    if kernel_class is Matern:
        return C(amp, amp_bounds) * Matern(length_scale=ell, nu=nu, length_scale_bounds="fixed")
    else:
        return C(amp, amp_bounds) * RBF(length_scale=ell, length_scale_bounds="fixed")
    
def reconstruct_nz_from_nodes_gp_square(
    z_pred,
    z_nodes,
    theta_nodes,
    z_bounds=(0.0, 3.0),
    corr_len=0.03,
    amp=0.04,
    alpha=0,                 # <-- set 0 for interpolation
    kernel_class=RBF,
    nu=1.5,
    return_std=False

):
    """
    This is the "inferred by the nodes" GP:
      Train on (zmin,0), (z_nodes, theta_nodes), (zmax,0).
      Predict on z_pred.

    IMPORTANT:
    - Do NOT clip/renormalize here if you want the GP mean to pass through nodes.
    - If you need positivity/normalization, do it in the node parameterization
      (e.g., sample log-theta) or use a log-GP model.
    """
    zmin, zmax = z_bounds
    z_pred = np.asarray(z_pred, float)    
    z_nodes = np.atleast_1d(np.asarray(z_nodes, dtype=float))
    theta_nodes = np.atleast_1d(np.asarray(theta_nodes, dtype=float))
    z_train = np.concatenate(([zmin], z_nodes, [zmax]))
    y_train_squareroot = np.concatenate(([0.0], np.sqrt(theta_nodes), [0.0]))


    X_train = x_log1p(z_train, zmax).reshape(-1, 1)
    X_pred  = x_log1p(z_pred, zmax).reshape(-1, 1)

    kernel = kernel_fixed(kernel_class, amp=amp, ell=corr_len, nu=nu, amp_bounds=(1e-6, 10.0))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        normalize_y=False,
        alpha=float(alpha),
        copy_X_train=False,
        random_state=0,
    )
    gp.fit(X_train, y_train_squareroot)

    if return_std:
        y_pred_squareroot, y_std = gp.predict(X_pred, return_std=True)
        y_pred = y_pred_squareroot**2
        return y_pred, y_std, gp
    else:
        y_pred_squareroot = gp.predict(X_pred, return_std=False)
        y_pred = y_pred_squareroot**2
        return y_pred
    
def reconstruct_all_bins_square(
    z_pred,
    z_nodes_all_bins,
    n_nodes_all_bins,            # list of arrays, each shape (M,)
    z_bounds=(0.0, 3.0),
    corr_lens=[0.5, 0.4, 0.4, 0.4, 0.4, 0.5],
    amp=0.04,
    alpha=0.0,                 # <-- use 0.0 if you want interpolation
    kernel_class=RBF,
    nu=1.5,
    return_std=False
):
    """
    Decode all bins at once: returns list of n_b(z_pred).
    If return_std=True, returns list of (mean,std) per bin.
    """
    out = {}
    for b, z_node in enumerate(z_nodes_all_bins):
        if return_std:
            mean, std, _ = reconstruct_nz_from_nodes_gp_square(
                z_pred=z_pred,
                z_nodes=z_node,
                theta_nodes=n_nodes_all_bins[b],
                z_bounds=z_bounds,
                corr_len=corr_lens[b],
                amp=amp,
                alpha=alpha,
                kernel_class=kernel_class,
                nu=nu,
                return_std=True
            )
            out[b+1] = (mean, std)
        else:
            nz = reconstruct_nz_from_nodes_gp_square(
                z_pred=z_pred,
                z_nodes=z_node,
                theta_nodes=n_nodes_all_bins[b],
                z_bounds=z_bounds,
                corr_len=corr_lens[b],
                amp=amp,
                alpha=alpha,
                kernel_class=kernel_class,
                nu=nu,
                return_std=False
            )
            out[b+1] = nz
    return out


def resample_dndz_gp(gp_predictions, z_pred, zs):
    gp_predictions_norm = {}
    for key in gp_predictions.keys():
        gp_predictions_norm[key] = gp_predictions[key]/integrate.trapezoid(gp_predictions[key], z_pred[:,0])
    my_dndz = np.vstack(list(gp_predictions_norm.values())) 
    my_dndz_norm = np.zeros([len(gp_predictions_norm), len(zs)])
    for i in range(len(gp_predictions_norm)):
        # gp_predictions[i][0] is 1D array of n(z) for bin i

        my_dndz_norm[i, :] = np.interp(zs, z_pred[:,0], my_dndz[i, :])
    
    return my_dndz_norm

x_nodes_all, y_nodes_all = compute_cdf_nodes(z_nz, dndz_she_norm, 3)

x_pos_fid  = np.array([x_nodes_all[b][0] for b in range(6)]) # array of the first training node for each bin, length N_bin
x_pos2_fid = np.array([x_nodes_all[b][1] for b in range(6)])
x_pos3_fid = np.array([x_nodes_all[b][2] for b in range(6)])

y_pos_fid  = np.array([y_nodes_all[b][0] for b in range(6)])
y_pos2_fid = np.array([y_nodes_all[b][1] for b in range(6)])
y_pos3_fid = np.array([y_nodes_all[b][2] for b in range(6)])

x_she_fid  = np.array([x_nodes_all[b][0] for b in range(6)])
x_she2_fid = np.array([x_nodes_all[b][1] for b in range(6)])
x_she3_fid = np.array([x_nodes_all[b][2] for b in range(6)])

y_she_fid  = np.array([y_nodes_all[b][0] for b in range(6)])
y_she2_fid = np.array([y_nodes_all[b][1] for b in range(6)])
y_she3_fid = np.array([y_nodes_all[b][2] for b in range(6)])

### End of GP preparation ###

fiducial_cosmology = {
    'H0': 67,                # Hubble constant
    'Omega_cdm0': 0.27,        # Cold dark matter density
    'Omega_b0': 0.049,         # Baryon density
    'Omega_k0': 0.0,           # Curvature
    'mnu': 0.077,                # Neutrino mass
    'w0': -1.0,                # Dark energy equation of state (present)
    'wa': 0.0,                 # Dark energy equation of state (evolution)
    'ns': 0.96,                # Scalar spectral index
    'As': 2.1e-9, # Scalar amplitude
    'gamma_MG': 0.545,         # Modified gravity parameter
    'log10TAGN': 7.75,          # AGN feedback parameter
    'N_mnu': 1,                  # Number of massive neutrino species
    'N_eff': 3.046             # Effective number of relativistic species
}

fiducial_nuisance = {
    # Intrinsic alignment
    'AIA': 0.16, 'EtaIA': 1.66, 'CIA': 0.0134,
    # Galaxy bias (poly coefficients)
    'b1_photo_poly0': 1.33291, 'b1_photo_poly1': -0.72414, 
    'b1_photo_poly2': 1.0183, 'b1_photo_poly3': -0.14913,
    # Magnification bias (per bin)
    'magnification_bias_1': 0.0, 'magnification_bias_2': 0.0,
    'magnification_bias_3': 0.0, 'magnification_bias_4': 0.0,
    'magnification_bias_5': 0.0, 'magnification_bias_6': 0.0,
    # Redshift errors (positions)
    'dz_pos_1': 0.0, 'dz_pos_2': 0.0, 'dz_pos_3': 0.0, 
    'dz_pos_4': 0.0, 'dz_pos_5': 0.0, 'dz_pos_6': 0.0,
    'width_pos_1': 1.0, 'width_pos_2': 1.0,
    'width_pos_3': 1.0, 'width_pos_4': 1.0,
    'width_pos_5': 1.0, 'width_pos_6': 1.0,
    # Multiplicative shear bias
    'multiplicative_bias_1': 0.0, 'multiplicative_bias_2': 0.0,
    'multiplicative_bias_3': 0.0, 'multiplicative_bias_4': 0.0,
    'multiplicative_bias_5': 0.0, 'multiplicative_bias_6': 0.0,
    # Redshift errors (shear)
    'dz_shear_1': 0.0, 'dz_shear_2': 0.0, 'dz_shear_3': 0.0,
    'dz_shear_4': 0.0, 'dz_shear_5': 0.0, 'dz_shear_6': 0.0,
    'width_shear_1': 1.0, 'width_shear_2': 1.0,
    'width_shear_3': 1.0, 'width_shear_4': 1.0,
    'width_shear_5': 1.0, 'width_shear_6': 1.0
    
}

ell_theory = np.array([
    10.97557970,
    13.11709027,
    15.67644367,
    18.73516773,
    22.39069761,
    26.75947964,
    31.98068069,
    38.22062129,
    45.67807378,
    54.59059412,
    65.24208925,
    77.97186088,
    93.18541388,
    111.36737359,
    133.09692347,
    159.06625492,
    190.10261691,
    227.19466787,
    271.52396925,
    324.50262398,
    387.81825878,
    463.48778323,
    553.92163814,
    662.00057974,
    791.16744571,
    945.53682628,
    1130.02613377,
    1350.51224608,
    1614.01871364,
    1928.93949355,
    2305.30633773,
    2755.10835283
])

# --- Cosmology and Tracers: Easy to Change! ---

# Choose which perturbation model to use: 'linear' or 'nonlinear'
perturbation_model = 'nonlinear'  # change to 'linear' if needed

# Cosmology parameters (easy to modify)
cosmo_params = fiducial_cosmology.copy()
nuisance_params = fiducial_nuisance.copy()

# Create background
background = CAMBBackground(
    H0=cosmo_params['H0'],
    Omega_cdm0=cosmo_params['Omega_cdm0'],
    Omega_b0=cosmo_params['Omega_b0'],
    w0=cosmo_params['w0'],
    wa=cosmo_params['wa'],
    Omega_k0=cosmo_params['Omega_k0'],
    ns=cosmo_params['ns'],
    As=cosmo_params['As'],
    mnu=cosmo_params['mnu'],
    gamma_MG=cosmo_params['gamma_MG'],
    N_mnu=cosmo_params['N_mnu']
)

# Choose perturbations
if perturbation_model == 'linear':
    perturbations = HMemuLinearPerturbations(background, myz)
else:
    perturbations = HMemuNonLinearPerturbations(
        background, 
        HMemuLinearPerturbations(background, myz), 
        myz, 
        log10TAGN=cosmo_params['log10TAGN']
    )


# Tracer definitions (easy to swap dndz, bias model, etc.)
tracer_pos = PositionsTracer(
    perturbations=perturbations,
    dndz=my_dndz_pos_norm,
    z=myz,
    galaxy_bias_model='poly',
    nuisance_params=nuisance_params
)

tracer_she = ShearTracer(
    perturbations=perturbations,
    dndz=my_dndz_she_norm,
    z=myz,
    nuisance_params=nuisance_params
)


# 🎯 Compute all the 2-point angular power spectra in one go!
cls_sheshe = AngularTwoPoint(tracer_she, tracer_she).get_Cl(ell_theory, 0, perturbations.k)
cls_posshe = AngularTwoPoint(tracer_pos, tracer_she).get_Cl(ell_theory, 0, perturbations.k)
cls_pospos = AngularTwoPoint(tracer_pos, tracer_pos).get_Cl(ell_theory, 0, perturbations.k)

# 🚀 Combine all spectra into a single dictionary for easy access!
uncoupled_cls = {**cls_posshe, **cls_sheshe, **cls_pospos}

def build_data(cls, mapa_key, cov, zs, nzs, include_pos=False, include_she=False):
    data = {
        'cells': cls,
        'ells': cls[mapa_key].ell,
        'z_arr': zs,
        'cov': cov,
        'mixmat': {},
    }
    if include_pos:
        data['dndz_pos'] = nzs[0]
    if include_she:
        data['dndz_she'] = nzs[1]
    return data

def build_settings(cls):
    scale_cuts = {}

    for key in cls:
        if key[:2] == ('POS', 'POS'):
            scale_cuts[key] = [10, 750]

        elif key[:2] == ('POS', 'SHE'):
            scale_cuts[key] = [10, 750]

        elif key[:2] == ('SHE', 'SHE'):
            scale_cuts[key] = [10, 1500]

    return {
        'n_ell_bins': 32,
        'scale_cuts': scale_cuts
    }

data_3x2pt  = build_data(uncoupled_cls, ('SHE', 'SHE', 1, 1), 
                             full_cov, 
                             myz, [my_dndz_pos_norm, my_dndz_she_norm],
                             include_pos=True, include_she=True)
settings_3x2pt  = build_settings(uncoupled_cls)

like_instance = EuclidLikelihood_3x2pt(
        data=data_3x2pt,
        settings=settings_3x2pt,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="uncoupled"

    )

loglike = like_instance.loglike(fiducial_cosmology | fiducial_nuisance)
print("✨ Log-likelihood at fiducial: ", loglike)

# Priors
zmean_dr1 = np.array([0.4371, 0.726, 0.9424, 1.164, 1.437, 1.963])

# --- Dynesty prior: parameter names and prior_transform ---

param_names = [
    'Omega_b0', 'Omega_cdm0', 'logAs', 'ns', 'H0',
    'AIA', 'EtaIA', 
    'b1_photo_poly0', 'b1_photo_poly1', 'b1_photo_poly2', 'b1_photo_poly3',
]
param_names += [f'multiplicative_bias_{i}' for i in range(1, 7)]
param_names += [f'magnification_bias_{i}' for i in range(1, 7)]
param_names += [f'x{i}_pos' for i in range(1, 7)]
param_names += [f'x{i}_pos_2' for i in range(1, 7)]
param_names += [f'x{i}_pos_3' for i in range(1, 7)]
param_names += [f'x{i}_she' for i in range(1, 7)]
param_names += [f'x{i}_she_2' for i in range(1, 7)]
param_names += [f'x{i}_she_3' for i in range(1, 7)]
param_names += [f'y{i}_pos' for i in range(1, 7)]
param_names += [f'y{i}_pos_2' for i in range(1, 7)]
param_names += [f'y{i}_pos_3' for i in range(1, 7)]
param_names += [f'y{i}_she' for i in range(1, 7)]
param_names += [f'y{i}_she_2' for i in range(1, 7)]
param_names += [f'y{i}_she_3' for i in range(1, 7)]

param_names += [f'log_l_pos_{i}' for i in range(1,7)]
param_names += [f'log_l_she_{i}' for i in range(1,7)]
#param_names += ['w0', 'wa']

ndim = len(param_names)  

# Pre-compute logAs bounds for efficiency
_logAs_lo = np.log(0.6e-9 * 1e10)
_logAs_hi = np.log(5e-9 * 1e10)

# Prior boundaries of uniform distribution for x- and y-nodes 
x_min = np.array([[0.05, 0.4, 0.55, 0.7, 1.0, 1.3], [0.4, 0.5, 0.6, 0.8, 1.0, 1.6], [0.4, 0.6, 0.8, 1.1, 1.3, 2.1]]) # [node 1, node 2, node 3]
x_max = np.array([[0.5, 0.9, 1.1, 1.4, 1.7, 2.1], [0.8, 1.0, 1.2, 1.4, 1.7, 2.4], [0.9, 1.1, 1.4, 1.7, 1.9, 2.9]])
y_min = np.array([[1.0, 2.0, 0.01, 0.01, 0.5, 1.4], [1.0, 2.5, 3.0, 2.5, 1.0, 0.6], [1.0, 0.01, 0.01, 0.01, 1.0, 0.1]])
y_max = np.array([[1.7, 4.0, 3.0, 2.5, 3.5, 2.0], [1.7, 5.5, 5.5, 4.5, 4.0, 1.1], [1.6, 4.5, 4.5, 1.5, 4.0, 0.5]])
corr_len_min = np.array([0.04, 0.01, 0.03, 0.03, 0.01, 0.02])
corr_len_max = np.array([0.09, 0.06, 0.09, 0.09, 0.09, 0.07])


def prior_transform(u):
    """Maps unit-cube samples u ~ Unif[0,1)^ndim to the parameter space."""
    v = np.empty(ndim)

    # Cosmology + IA + bias (uniform priors)
    v[0]  = 0.01 + (0.1   - 0.01 ) * u[0]   # Omega_b0     [0.01, 0.1)
    v[1]  = 0.1  + (0.8   - 0.1  ) * u[1]   # Omega_cdm0   [0.1, 0.8)
    v[2]  = _logAs_lo + (_logAs_hi - _logAs_lo) * u[2]  # logAs
    v[3]  = 0.6  + (1.2   - 0.6  ) * u[3]   # ns           [0.6, 1.2)
    v[4]  = 50.  + (90.   - 50.  ) * u[4]   # H0           [50, 90)
    v[5]  = -1.  + 2.              * u[5]   # AIA          [-1, 1)
    v[6]  = -5.  + 10.             * u[6]   # EtaIA        [-5, 5)

    v[7]  = -3.  + 6.              * u[7]   # b1_photo_poly0
    v[8]  = -3.  + 6.              * u[8]   # b1_photo_poly1
    v[9]  = -3.  + 6.              * u[9]   # b1_photo_poly2
    v[10] = -3.  + 6.              * u[10]  # b1_photo_poly3

    # multiplicative_bias_1..6 ~ N(0, 0.01)
    for i in range(6):
        v[11 + i] = norm.ppf(u[11 + i], loc=0.0, scale=0.01)

    # magnification_bias_1..6 ~ Uniform[-2, 2)
    for i in range(6):
        v[17 + i] = -2.0 + 4.0 * u[17 + i]
    
    offset = 23
    # x_pos nodes (3 sets) ~ U(x_min, x_max)
    for node in range(3):
        for b in range(6):
            v[offset + b] = x_min[node, b] + (x_max[node, b] - x_min[node, b]) * u[offset + b]
        offset += 6

    # x_she nodes (3 sets) ~ U(x_min, x_max)
    for node in range(3):
        for b in range(6):
            v[offset + b] = x_min[node, b] + (x_max[node, b] - x_min[node, b]) * u[offset + b]
        offset += 6

    # y_pos nodes (3 sets) ~ U(y_min, y_max)
    for node in range(3):
        for b in range(6):
            v[offset + b] = y_min[node, b] + (y_max[node, b] - y_min[node, b]) * u[offset + b]
        offset += 6

    # y_she nodes (3 sets) ~ U(y_min, y_max)
    for node in range(3):
        for b in range(6):
            v[offset + b] = y_min[node, b] + (y_max[node, b] - y_min[node, b]) * u[offset + b]
        offset += 6

    # log_l_pos, log_l_she ~ log(U[])
    for b in range(6):
        v[offset + b] = np.log(corr_len_min[b]) + (np.log(corr_len_max[b]) - np.log(corr_len_min[b])) * u[offset + b]
    offset += 6
    for b in range(6):
        v[offset + b] = np.log(corr_len_min[b]) + (np.log(corr_len_max[b]) - np.log(corr_len_min[b])) * u[offset + b]

    return v


print("✨ Prior set up completed.")
print("Parameters sampled: ", param_names)
print("Number of sampled parameters:", ndim)

default_pars = fiducial_cosmology | fiducial_nuisance

def get_flow_log_prob(flow, mean, std, x_nodes, y_nodes, ell):
    GP_params = np.concatenate([x_nodes, y_nodes, [ell]])
    GP_params_norm = (GP_params - mean) / std
    GP_params_tensor = torch.tensor(GP_params_norm, dtype=torch.float32).unsqueeze(0)
    log_prob = flow.log_prob(GP_params_tensor).item()
    return log_prob

def build_flow(M, n_layers=5, hidden_features=32):
    n_params = 2*M + 1
    base = StandardNormal([n_params])
    transforms = [
        MaskedAffineAutoregressiveTransform(
            features=n_params,
            hidden_features=hidden_features
        )
        for _ in range(n_layers)
    ]
    transform = CompositeTransform(transforms)
    return Flow(transform, base)

print("Retrieving saved flows")
flows_she = []
flows_pos = []
means_she = []
stds_she = []
means_pos = []
stds_pos = []

for b in range(6):
    flow_she = build_flow(M=3)
    flow_she.load_state_dict(torch.load(f"flow_bin_{b+1}.pt", map_location="cpu"))
    flow_she.eval()
    flows_she.append(flow_she)

    flow_pos = build_flow(M=3)
    flow_pos.load_state_dict(torch.load(f"flow_bin_{b+1}.pt", map_location="cpu"))
    flow_pos.eval()
    flows_pos.append(flow_pos)

    means_she.append(np.load(f"flow_mean_bin_{b+1}.npy"))
    stds_she.append(np.load(f"flow_std_bin_{b+1}.npy"))
    means_pos.append(np.load(f"flow_mean_bin_{b+1}.npy"))
    stds_pos.append(np.load(f"flow_std_bin_{b+1}.npy"))

print("Done!")

def loglike_dynesty(v):
    """Full 3x2pt log-likelihood with sigma8 as a blob."""
    z_nodes_she = []
    n_nodes_she = []
    z_nodes_pos = []
    n_nodes_pos = []
    l_pos = []
    l_she = []
    param_dict = dict(zip(param_names, v))

    # Enforce physical prior on w0 + wa
#    if param_dict['w0'] + param_dict['wa'] > -1. / 3:
#        return (-np.inf, np.array([0.0]))
    
    # Extract parameters from param_dict
    for i in range(6):
        z_nodes_she.append([param_dict[f"x{i+1}_she"], param_dict[f"x{i+1}_she_2"], param_dict[f"x{i+1}_she_3"]]) 
        n_nodes_she.append([param_dict[f"y{i+1}_she"], param_dict[f"y{i+1}_she_2"], param_dict[f"y{i+1}_she_3"]])
        z_nodes_pos.append([param_dict[f"x{i+1}_pos"], param_dict[f"x{i+1}_pos_2"], param_dict[f"x{i+1}_pos_3"]])
        n_nodes_pos.append([param_dict[f"y{i+1}_pos"], param_dict[f"y{i+1}_pos_2"], param_dict[f"y{i+1}_pos_3"]])
        l_pos.append(np.exp(param_dict[f"log_l_pos_{i+1}"]))
        l_she.append(np.exp(param_dict[f"log_l_she_{i+1}"]))
        
    # enforce monotonic nodes for shear bins
    for nodes in z_nodes_she:
        if not np.all(np.diff(nodes) > 0):
            return (-np.inf, np.array([0.0]))

    # enforce monotonic nodes for position bins
    for nodes in z_nodes_pos:
        if not np.all(np.diff(nodes) > 0):
            return (-np.inf, np.array([0.0]))
    
    # Reconstruct n(z) using GP
    predictions_she = reconstruct_all_bins_square(z_pred, z_nodes_she, n_nodes_she, z_bounds, l_she, 0.04, alpha, kernel_class, nu)
    my_dndz_norm_gp_she = resample_dndz_gp(predictions_she, z_pred_res, myz)
    predictions_pos = reconstruct_all_bins_square(z_pred, z_nodes_pos, n_nodes_pos, z_bounds, l_pos, 0.04, alpha, kernel_class, nu)
    my_dndz_norm_gp_pos = resample_dndz_gp(predictions_pos, z_pred_res, myz)
    
    # Pass reconstructed n(z) as data
    like_instance.data.update({'dndz_she': my_dndz_norm_gp_she, 'dndz_pos': my_dndz_norm_gp_pos})
    
    pars_in = default_pars.copy()
    pars_in.update(param_dict)
    pars_in['As'] = np.exp(param_dict['logAs']) * 1e-10
    logl = like_instance.loglike(pars_in)

    logl_flow = 0.0
    # Compute the probability of the current set of parameters according to the flow. Add log(l) as Jacobian, since flow was trained on l. 
    for i in range(6):
        logl_flow += get_flow_log_prob(flows_pos[i], means_pos[i], stds_pos[i], z_nodes_pos[i], n_nodes_pos[i], l_pos[i]) + np.log(l_pos[i])
        logl_flow += get_flow_log_prob(flows_she[i], means_she[i], stds_she[i], z_nodes_she[i], n_nodes_she[i], l_she[i]) + np.log(l_she[i])
    logl += logl_flow

    sigma8 = like_instance.derived['sigma8_0']
    
    return (logl, np.array([sigma8]))

if __name__ == '__main__':

    checkpoint_file = name_to_chain + '_dynesty.save'
    pool = MPIPoolExecutor()
#    pool = MPIPool()
#    if not pool.is_master():
#      pool.wait()
#      sys.exit(0)

    # 100 workers to match the original Nautilus pool size
#    with DynestyPool(32, loglike_dynesty, prior_transform) as pool:

    if os.path.exists(checkpoint_file):
        print('Resuming from checkpoint...')
        sampler = dynesty.DynamicNestedSampler.restore(
                checkpoint_file, pool=pool
            )
    else:
        sampler = dynesty.DynamicNestedSampler(
                loglike_dynesty,
                prior_transform,
                ndim,
                nlive=ndim*50,      # increased from 500: better bounding in 65D
                pool=pool,
                blob=True,       # store sigma8 alongside each sample
                sample='rslice', # reflective slice sampling: far better than rwalk in high-D (avoids high nc)
                bound='multi',   # multi-ellipsoid bounding (explicit)
                bootstrap=0,     # skip bootstrap resampling → faster bounding updates
                enlarge=1.25,    # ellipsoid enlargement factor
                slices=10,       # number of slice directions per update
                queue_size=128
            )

    t_start = time.time()
    print('Starting sampling...')

    sampler.run_nested(
            dlogz_init=0.01,        # convergence on evidence
            n_effective=10000,      # also stop when ESS >= 10000 (whichever comes first)
            print_progress=True,
            checkpoint_file=checkpoint_file,
            checkpoint_every=300,   # checkpoint every 5 min
            resume=os.path.exists(checkpoint_file),
        )

    t_end = time.time()
    total_seconds = t_end - t_start
    days    = int(total_seconds // 86400)
    hours   = int((total_seconds % 86400) // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f'Total time: {days}d {hours}h {minutes}m {seconds}s')
    print('Sampling finished!')
    print('Saving chains...')

    results = sampler.results

        # --- Sampled parameters ---
    chain_dict = {
            name: results.samples[:, i]
            for i, name in enumerate(param_names)
        }

        # --- Derived parameters (sigma8 from blob) ---
    sigma8_arr = results.blob[:, 0]
    derived_dict = {'sigma8_0': sigma8_arr}

        # --- Save everything in a labelled way ---
    np.savez_compressed(
            name_to_chain + '.npz',
            chain=chain_dict,
            derived=derived_dict,
            weights=results.logwt,
            logl=results.logl,
            logz=results.logz,
            time=total_seconds,
            param_names=param_names,
            derived_names=['sigma8_0'],
        )

    print('Chains saved with name:', name_to_chain + '.npz')

