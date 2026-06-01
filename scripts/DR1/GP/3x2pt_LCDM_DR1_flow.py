import os
os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
from scipy import integrate
from multiprocessing import get_context
import time

from nautilus import Prior
from nautilus import Sampler
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
# Import euclidlib for reading the Euclid data
import euclidlib as el

# Import cloelib for cosmology and theoretical predictions
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations

# Import cloelike for likelihoods
from cloelike.EuclidLikelihood_photo_Cls import EuclidLikelihood_3x2pt

# Normalizing flow imports
from tqdm import tqdm
import torch 
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform
torch.set_num_threads(1)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ConstantKernel as C
)

name_to_chain_gp = '3x2pt_LCDM_DR1_flow_jac'

myz = np.linspace(1e-4, 3.0, 100)
def normalize_dndz(nz_example, z_nz):
    normalized = {}
    for key in nz_example:
        normalized[key] = nz_example[key] / integrate.trapezoid(nz_example[key], z_nz)
    return normalized
full_cov=np.load('/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/data/covariance_Gauss_DR1_TR1like_1589_2D.npz')['Gauss']

# Get n(z)
z_nz, nz_heracles = el.phz.redshift_distributions('/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/data/nz_example.fits')


# Normalize both dndz_pos and dndz_she
dndz_pos_norm = normalize_dndz(nz_heracles, z_nz)
dndz_she_norm = normalize_dndz(nz_heracles, z_nz)

# Find boundaries for x-nodes priors 
def compute_cdf_nodes_per_bin(z_grid, nz_ens, M=5, qmin=0.05, qmax=0.95):
    nz_mean = nz_ens.mean(axis=0)
    nz_mean /= np.trapezoid(nz_mean, z_grid) # normalized mean of ensemble
    cdf = integrate.cumulative_trapezoid(nz_mean, z_grid, initial=0) # cdf of ensemble mean
    cdf /= cdf[-1]
    quantiles = np.linspace(qmin, qmax, M)
    x_nodes = np.interp(quantiles, cdf, z_grid) # x nodes at given quantiles
    Y_nodes = []
    for k in range(nz_ens.shape[0]):
        nz_k = nz_ens[k] / np.trapezoid(nz_ens[k], z_grid)
        y_k = np.interp(x_nodes, z_grid, nz_k)
        Y_nodes.append(y_k)
    Y_nodes = np.stack(Y_nodes, axis=0)  # shape: (N_ens, M) 
    return x_nodes, Y_nodes

def generate_bin1_ensemble(
    N_ens=5,
    z_lo=0.001,
    z_hi=6,
    N_z=3001,
    mean_shift_sigma=0.01,
    width_shift_sigma=0.02,
    shape_noise_sigma=0.07,
    smooth_scale=3,
    nz = dndz_she_norm[1]
):
    """
    Generate an ensemble of n(z) realisations for the FIRST bin only:
        z ∈ [0.001, 0.42]
    using small shifts + smooth perturbations.
    """

    # ---------------------------
    # 1. Grid + base bin n(z)
    # ---------------------------
    z = np.linspace(z_lo, z_hi, N_z)
    nz /= np.trapezoid(nz, z)        # ensure norm within [0, 3] part
    nz_bin = nz.copy()               # for first bin, that's it

    # ---------------------------
    # Build ensemble
    # ---------------------------
    ensemble = []

    z_mean = np.trapezoid(z * nz_bin, z)

    for _ in range(N_ens):

        # Mean shift
        dz = np.random.normal(0, mean_shift_sigma) * z_mean
        z_shift = np.clip(z + dz, z_lo, z_hi)

        # Width/stretch shift
        width = 1 + np.random.normal(0, width_shift_sigma)
        z_stretch = z_mean + width * (z_shift - z_mean)
        z_stretch = np.clip(z_stretch, z_lo, z_hi)

        # Smooth random shape fluctuation
        noise = np.random.normal(0, shape_noise_sigma, N_z)
        noise_smooth = gaussian_filter1d(noise, smooth_scale)
        nz_pert = nz_bin * (1 + noise_smooth)
        nz_pert = np.clip(nz_pert, 1e-12, None)

        # Interpolate to shifted grid
        nz_new = np.interp(z, z_stretch, nz_pert, left=0, right=0)

        # Normalise inside the bin
        nz_new /= np.trapezoid(nz_new, z)

        ensemble.append(nz_new)

    return z, np.array(ensemble)

z_ens_list, nz_ensemble_list = zip(*[
    generate_bin1_ensemble(
        N_ens=50, 
        nz=dndz_she_norm[i], 
        N_z=3001
    )
    for i in range(1, 7)
])

nz_ens = nz_ensemble_list[0]     
z_grid = z_ens_list[0]           
x_nodes, Y_nodes = compute_cdf_nodes_per_bin(z_grid, nz_ens, M=5)
print("x_nodes:", x_nodes)

# Function to resample the normalized dndz
def resample_dndz(nz_example, z_nz, myz):
    my_dndz = np.vstack(list(nz_example.values())) #stacking arrays vertically
    my_dndz_norm = np.zeros([len(nz_example), len(myz)]) #creating an empty array
    for i in range(len(nz_example)):
        my_dndz_norm[i, :] = np.interp(myz, z_nz, my_dndz[i, :]) #interpolating to new myz values
    return my_dndz_norm

my_dndz_pos_norm = resample_dndz(dndz_pos_norm, z_nz, myz)
my_dndz_she_norm = resample_dndz(dndz_she_norm, z_nz, myz)

def x_log1p(z, zmax):
    """Map z to x in [0,1] using log(1+z)."""
    return np.log1p(z) / np.log1p(zmax)

def normalize_nz(z, nz): 
    """
    Normalize the n(z)
    """
    area = np.trapezoid(nz, z)
    area = area if area > 0 else np.finfo(float).tiny
    return nz / area

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
    alpha=0.0,                 # <-- set 0 for interpolation
    kernel_class=RBF,
    nu=1.5,
    return_std=False,          # <-- optional like your example
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

    # training data (anchors)
    z_train = np.concatenate([[zmin], np.asarray(z_nodes, float), [zmax]])
    y_train_squareroot = np.concatenate([[0], np.asarray(np.sqrt(theta_nodes), float), [0]])

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


# Define fixed GP hyperparameters 
z_bounds = (0,3)                              # bounds for the redshift
z_pred = np.linspace(0, 3, 400)       # prediction points for GP
alpha = 1e-8                                  # noise level during fitting 
kernel_class = RBF
nu = 1.5                                      # necessary for Matern only, defines smoothness 
const_value = 0.04                            # amplitude of constant kernel
const_bounds = (1e-4, 1.0)                    # bounds for constant kernel amplitude
z_pred_res = z_pred.reshape(-1, 1)


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
    'magnification_bias_1': 0.0, 
    # Redshift errors (positions)
    'dz_pos_1': 0.0,
    'width_pos_1': 1.0,
    # Multiplicative shear bias
    'multiplicative_bias_1': 0.0,
    # Redshift errors (shear)
    'dz_shear_1': 0.0,
    'width_shear_1': 1
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

tracer_pos = PositionsTracer(
    perturbations=perturbations,
    dndz=my_dndz_pos_norm[0][None, :],
    z=myz,
    galaxy_bias_model='poly',
    nuisance_params=nuisance_params
)

tracer_she = ShearTracer(
    perturbations=perturbations,
    dndz=my_dndz_she_norm[0][None, :],
    z=myz,
    nuisance_params=nuisance_params
)


cls_sheshe = AngularTwoPoint(tracer_she, tracer_she).get_Cl(ell_theory, 0, perturbations.k)
cls_posshe = AngularTwoPoint(tracer_pos, tracer_she).get_Cl(ell_theory, 0, perturbations.k)
cls_pospos = AngularTwoPoint(tracer_pos, tracer_pos).get_Cl(ell_theory, 0, perturbations.k)

# Combine all spectra into a single dictionary for easy access
uncoupled_cls = {**cls_posshe, **cls_sheshe, **cls_pospos}

n_ell = 32
idx_wl  = np.arange(0,        0 + n_ell)      # 0:32
idx_ggl = np.arange(672,      672 + n_ell)    # 672:704
idx_gc  = np.arange(1824,     1824 + n_ell)   # 1824:1856
idx_bin1 = np.concatenate([idx_wl, idx_ggl, idx_gc])
cov_bin1 = full_cov[np.ix_(idx_bin1, idx_bin1)]

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

data_3x2pt = build_data(uncoupled_cls, ('SHE', 'SHE', 1, 1), 
                             cov_bin1, 
                             myz, [my_dndz_pos_norm[0][None,:], my_dndz_she_norm[0][None,:]],
                             include_pos=True, include_she=True)
settings_3x2pt  = build_settings(uncoupled_cls)

print('New settings:', settings_3x2pt)

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

x_nodes, Y_nodes = compute_cdf_nodes_per_bin(z_grid, nz_ens, M=5)
def build_flow(M, n_layers=5, hidden_features=64):
    base = StandardNormal([M])
    transforms = [
        MaskedAffineAutoregressiveTransform(
            features=M,
            hidden_features=hidden_features
        )
        for _ in range(n_layers)
    ]
    transform = CompositeTransform(transforms)
    return Flow(transform, base)
print("Retrieving saved flows")
flow_she_bin1 = build_flow(M=5)
flow_she_bin1.load_state_dict(torch.load("/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/chains/DR1/GP/norm_flow_2/trained_flows/flow_bin1.pt", map_location="cpu"))
flow_she_bin1.eval()

flow_mean_she = np.load("/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/chains/DR1/GP/norm_flow_2/trained_flows/flow_mean_bin1.npy")
flow_std_she  = np.load("/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/chains/DR1/GP/norm_flow_2/trained_flows/flow_std_bin1.npy")

flow_pos_bin1 = build_flow(M=5)
flow_pos_bin1.load_state_dict(torch.load("/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/chains/DR1/GP/norm_flow_2/trained_flows/flow_bin1.pt", map_location="cpu"))
flow_pos_bin1.eval()

flow_mean_pos = np.load("/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/chains/DR1/GP/norm_flow_2/trained_flows/flow_mean_bin1.npy")
flow_std_pos  = np.load("/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/chains/DR1/GP/norm_flow_2/trained_flows/flow_std_bin1.npy")
print("Done!")
# Priors
zmean_dr1 = np.array([0.4371, 0.726, 0.9424, 1.164, 1.437, 1.963])

prior_gp = Prior()
prior_gp.add_parameter('Omega_b0', dist=(0.01, 0.1))
prior_gp.add_parameter('Omega_cdm0', dist=(0.1, 0.8))
prior_gp.add_parameter('logAs', dist=(np.log(0.6e-9*1e10), np.log(5e-9*1e10)))
prior_gp.add_parameter('ns', dist=(0.6, 1.2))
prior_gp.add_parameter('H0', dist=(50, 90))
prior_gp.add_parameter('AIA', dist=(-1, 1))
prior_gp.add_parameter('EtaIA', dist=(-5, 5))
#prior_gp.add_parameter('w0', dist=(-3.0, -0.5))
#prior_gp.add_parameter('wa', dist=(-3.0, 3.0))
prior_gp.add_parameter('b1_photo_poly0', dist=(-3.0, 3.0))
prior_gp.add_parameter('b1_photo_poly1', dist=(-3.0, 3.0))
prior_gp.add_parameter('b1_photo_poly2', dist=(-3.0, 3.0))
prior_gp.add_parameter('b1_photo_poly3', dist=(-3.0, 3.0))
prior_gp.add_parameter('multiplicative_bias_1', norm(loc=0.0, scale=0.01))
prior_gp.add_parameter('magnification_bias_1', dist=(-2.0, 2.0))
prior_gp.add_parameter('y1_she', dist= (0.3, 2))
prior_gp.add_parameter('y2_she', dist= (2, 5))
prior_gp.add_parameter('y3_she', dist= (3, 6))
prior_gp.add_parameter('y4_she', dist= (3.5, 6.5))
prior_gp.add_parameter('y5_she', dist= (1, 4))
prior_gp.add_parameter('log_l_she', dist=(np.log(0.04), np.log(0.07)))
prior_gp.add_parameter('y1_pos', dist= (0.3, 2))
prior_gp.add_parameter('y2_pos', dist= (2, 5))
prior_gp.add_parameter('y3_pos', dist= (3, 6))
prior_gp.add_parameter('y4_pos', dist= (3.5, 6.5))
prior_gp.add_parameter('y5_pos', dist= (1, 4))
prior_gp.add_parameter('log_l_pos', dist=(np.log(0.04), np.log(0.07)))

print("✨ Prior set up completed.")
print("Parameters sampled: ", prior_gp.keys)
print("Number of sampled parameters:", len(prior_gp.keys))
default_pars = fiducial_cosmology | fiducial_nuisance
n_meds_she = [None]*5
n_meds_pos = [None]*5

step_counter = 0
save_every = 100
def like_Naut_test_gp(param_dict):
    # Enforce physical prior on w0 + wa
#    if param_dict['w0'] + param_dict['wa'] > -1. / 3:
#        return (-np.inf, 0)
    global step_counter
    l_she  = np.exp(param_dict["log_l_she"])
    n_meds_she[0] = param_dict["y1_she"]
    n_meds_she[1] = param_dict["y2_she"]
    n_meds_she[2] = param_dict["y3_she"]
    n_meds_she[3] = param_dict["y4_she"]
    n_meds_she[4] = param_dict["y5_she"]
    l_pos  = np.exp(param_dict["log_l_pos"])
    n_meds_pos[0] = param_dict["y1_pos"]
    n_meds_pos[1] = param_dict["y2_pos"]
    n_meds_pos[2] = param_dict["y3_pos"]
    n_meds_pos[3] = param_dict["y4_pos"]
    n_meds_pos[4] = param_dict["y5_pos"]
    predictions_she = {1: reconstruct_nz_from_nodes_gp_square(z_pred, x_nodes, n_meds_she, corr_len=l_she)}
    my_dndz_norm_gp_she = resample_dndz_gp(predictions_she, z_pred_res, myz)
    predictions_pos = {1: reconstruct_nz_from_nodes_gp_square(z_pred, x_nodes, n_meds_pos, corr_len=l_pos)}
    my_dndz_norm_gp_pos = resample_dndz_gp(predictions_pos, z_pred_res, myz)
    pars_in = default_pars.copy()
    pars_in.update(param_dict)


    #h = pars_in['H0'] / 100.0
    #pars_in['Omega_cdm0'] = param_dict['omch2'] / h**2
    #pars_in['Omega_b0'] = param_dict['ombh2'] / h**2
    pars_in['As'] = np.exp(param_dict['logAs']) * 1e-10
    like_instance.data.update({'dndz_she': my_dndz_norm_gp_she, 'dndz_pos': my_dndz_norm_gp_pos})
    logL = like_instance.loglike(pars_in)
    sigma_8 = like_instance.derived['sigma8_0']
    
    U_she = np.log(n_meds_she)
    U_she_norm = (U_she - flow_mean_she) / flow_std_she
    logp_flow_she = flow_she_bin1.log_prob(
        torch.tensor(U_she_norm[None, :], dtype=torch.float32)
    ).item()

    # --- POS flow prior ---
    U_pos = np.log(n_meds_pos)
    U_pos_norm = (U_pos - flow_mean_pos) / flow_std_pos
    logp_flow_pos = flow_pos_bin1.log_prob(
        torch.tensor(U_pos_norm[None, :], dtype=torch.float32)
    ).item()
    logp_she = logp_flow_she - np.sum(np.log(n_meds_she))
    logp_pos = logp_flow_pos - np.sum(np.log(n_meds_pos))

    # --- Total log-likelihood ---
    logL_total = logL + logp_she + logp_pos
    if step_counter % save_every == 0:
        step = {"params": param_dict.copy(), "dndz_norm": my_dndz_norm_gp_she.copy(), "loglike": logL_total, "logprior": logp_she + logp_pos, "C_ell": like_instance.theory_prediction}
        np.save(f"/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/chains/DR1/GP/norm_flow_jac/saved_steps/gp_sampling_step_{step_counter/100}.npy", step)
    step_counter += 1
    return (logL_total, sigma_8)

blob_vec = [('sigma8_0', float)]

if __name__ == '__main__':
    with get_context("spawn").Pool(32) as p:
        sampler = Sampler(prior_gp, like_Naut_test_gp, 
                              n_live=5000, pool=p,
                              blobs_dtype=blob_vec,
            filepath=name_to_chain_gp + '.h5', 
            resume=True)
        t_start = time.time()
        print('Starting sampling...')
        sampler.run(verbose=True)
        t_end = time.time()
        total_seconds = t_end - t_start
        days = int(total_seconds // 86400)
        hours = int((total_seconds % 86400) // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        print(f'Total time: {days}d {hours}h {minutes}m {seconds}s')
        print('Sampling finished!')
        print('Saving chains...')

        points, log_w, log_l, derived = sampler.posterior(return_blobs=True)
        log_z = sampler.log_z

        # --- Sampled parameters ---
        param_names = list(prior_gp.keys)
        chain_dict = {
            name: points[:, i]
            for i, name in enumerate(param_names)
        }

        # --- Derived parameters ---
        derived_names = [name for name, _ in blob_vec]

        derived_dict = {
            derived_names[0]: np.array([sample[0] for sample in derived])
        }

        # --- Save everything in a labelled way ---
        np.savez_compressed(
            name_to_chain_gp + '.npz',
            chain=chain_dict,
            time=total_seconds,
            derived=derived_dict,
            weights=log_w,
            logl=log_l,
            logz=log_z,
            param_names=param_names,
            derived_names=derived_names
        )

        print('Chains saved with name:', name_to_chain_gp + '.npz')
