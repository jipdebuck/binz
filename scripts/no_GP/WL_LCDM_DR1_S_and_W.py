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
from cloelike.EuclidLikelihood_photo_Cls import EuclidLikelihood_WL

# We read with euclidlib v2025.2 (v2025.1 is also compatible)

name_to_chain = 'WL_LCDM_DR1'

#cells_data = el.photo.harmonic_space.angular_power_spectra('/disks/shear16/canasherrera/SheariouslyCalibrated/data/DR1/synthetic_angular_spectra_DR1.fits')
full_cov=np.load('/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/data/covariance_Gauss_DR1_TR1like_1589_2D.npz')['Gauss']

# Get n(z)
z_nz, nz_heracles = el.phz.redshift_distributions('/leonardo_work/EUHPC_E07_074/jdebuck0/gp_project/data/nz_example.fits')

# Normalize and resample n(z) for both position and shear
myz = np.linspace(1e-4, 3.0, 100)
def normalize_and_resample(nz_dict, z_grid, z_target):
    nz_array = np.vstack([nz / integrate.trapezoid(nz, z_grid) for nz in nz_dict.values()])
    return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])

my_dndz_pos_norm = normalize_and_resample(nz_heracles, z_nz, myz)
my_dndz_she_norm = normalize_and_resample(nz_heracles, z_nz, myz)

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
tracer_she = ShearTracer(
    perturbations=perturbations,
    dndz=my_dndz_she_norm,
    z=myz,
    nuisance_params=nuisance_params
)


# 🎯 Compute all the 2-point angular power spectra in one go!
cls_sheshe = AngularTwoPoint(tracer_she, tracer_she).get_Cl(ell_theory, 0, perturbations.k)

# 🚀 Combine all spectra into a single dictionary for easy access!
uncoupled_cls_WL = {**cls_sheshe}

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

cov_WL = full_cov[:(21)*32, :(21)*32]

data_WL  = build_data(uncoupled_cls_WL, ('SHE', 'SHE', 1, 1), 
                             cov_WL, 
                             myz, [my_dndz_pos_norm, my_dndz_she_norm],
                             include_pos=False, include_she=True)
settings_WL  = build_settings(uncoupled_cls_WL)


like_instance = EuclidLikelihood_WL(
        data=data_WL,
        settings=settings_WL,
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
    'AIA', 'EtaIA'
]
param_names += [f'multiplicative_bias_{i}' for i in range(1, 7)]
param_names += [f'dz_shear_{i}' for i in range(1, 7)]
param_names += [f'width_shear_{i}' for i in range(1, 7)]

ndim = len(param_names)  

# Pre-compute logAs bounds for efficiency
_logAs_lo = np.log(0.6e-9 * 1e10)
_logAs_hi = np.log(5e-9 * 1e10)

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
#    v[7]  = -3.  + 2.5             * u[7]   # w0           [-3.0, -0.5)
#    v[8]  = -3.  + 6.              * u[8]   # wa           [-3.0, 3.0)
    # multiplicative_bias_1..6 ~ N(0, 0.01)
    for i in range(6):
        v[7 + i] = norm.ppf(u[7 + i], loc=0.0, scale=0.01)
    # dz_shear_1..6 ~ N(0, 0.002*(1+zmean))
    for i in range(6):
        v[13 + i] = norm.ppf(u[13 + i], loc=0.0, scale=0.002 * (1 + zmean_dr1[i]))
    # width_shear_1..6 ~ N(1, 0.1)
    for i in range(6):
        v[19 + i] = norm.ppf(u[19 + i], loc=1, scale=0.1)
    return v

print("✨ Prior set up completed.")
print("Parameters sampled: ", param_names)
print("Number of sampled parameters:", ndim)

default_pars = fiducial_cosmology | fiducial_nuisance

def loglike_dynesty(v):
    """Full 3x2pt log-likelihood with sigma8 as a blob."""
    param_dict = dict(zip(param_names, v))

    # Enforce physical prior on w0 + wa
#    if param_dict['w0'] + param_dict['wa'] > -1. / 3:
#        return (-np.inf, np.array([0.0]))

    pars_in = default_pars.copy()
    pars_in.update(param_dict)
    pars_in['As'] = np.exp(param_dict['logAs']) * 1e-10

    logl = like_instance.loglike(pars_in)
    sigma8 = like_instance.derived['sigma8_0']

    return (logl, np.array([sigma8]))

if __name__ == '__main__':

    checkpoint_file = name_to_chain + '_dynesty.save'

    # 100 workers to match the original Nautilus pool size
    with DynestyPool(32, loglike_dynesty, prior_transform) as pool:

        if os.path.exists(checkpoint_file):
            print('Resuming from checkpoint...')
            sampler = dynesty.DynamicNestedSampler.restore(
                checkpoint_file, pool=pool
            )
        else:
            sampler = dynesty.DynamicNestedSampler(
                pool.loglike,
                pool.prior_transform,
                ndim,
                nlive=500,      
                pool=pool,
                blob=True,       # store sigma8 alongside each sample
                sample='rslice', # reflective slice sampling: far better than rwalk in high-D (avoids high nc)
                bound='multi',   # multi-ellipsoid bounding (explicit)
                bootstrap=0,     # skip bootstrap resampling → faster bounding updates
                enlarge=1.25,    # ellipsoid enlargement factor
                slices=10,       # number of slice directions per update
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

