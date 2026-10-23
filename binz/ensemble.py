import numpy as np
from binz.gp import reconstruct_per_bin, compute_cdf_nodes
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF, Matern


def generate_bin1_ensemble(nz, N_ens=5, z_lo=0.001, z_hi=6, N_z=3001, mean_shift_sigma=0.01, width_shift_sigma=0.02, shape_noise_sigma=0.07,
                            smooth_scale=3):
    """
    Generate an ensemble of n(z) realisations for a single bin using small shifts + smooth perturbations.

    Parameters 
    -----------
    nz : array
        Redshift distribution in a single bin.
    N_ens : int 
        Number of realizations to generate.
    z_lo : float
        Lower redshift boundary.
    z_hi : float
        Upper redshift boundary.
    mean_shift_sigma : float
        Sigma of the shift in the mean applied to original n(z).
    width_shift_sigma : float
        Sigma of the shift in the width applied to original n(z).
    shape_noise_sigma : float
        Sigma of the fluctuations in the shape applied to original n(z).
    smooth_scale : float
        Smoothenss scale used in the Gaussian filter.

    Returns
    -----------
    z : array
        Redshift grid.
    ensemble : list
        List of N_ens n(z) realizations generated.
    
    """
    z = np.linspace(z_lo, z_hi, N_z)
    nz /= np.trapezoid(nz, z)        # ensure norm within [0, 3] part
    nz_bin = nz.copy()               # for first bin, that's it
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

def fit_gp_params(z_grid, nz, M=5, z_bounds=(0.0, 3.0), ell_init=0.03, amp=0.04, alpha=1e-8, kernel_class=RBF, nu=1.5, qmin=0.05, qmax=0.95, 
                            maxiter=1000, use_peak = False):
    """
    Fit the GP x-nodes, y-nodes, and correlation length to a single n(z) realisation from the ensemble.

    Parameters 
    -----------
    z_grid : array
        Redshift grid.
    nz : array
        Current n(z) from the ensemble.
    M : int
        Number of nodes.
    z_bounds : tuple
        Redshift boundaries.
    ell_init : float
        Initial guess for correlation length.
    amp : float
        Amplitude of constant kernel. 
    alpha : float
        Noise term used in GP.
    kernel_class : type
        Kernel used in GP reconstruction.
    nu : float
        Smoothness term for Matern kernel.
    qmin : float
        Quantile where first node should be positioned.
    qmax : float
        Quantile where last node should be positioned.
    maxiter : int
        Maximum number of iterations in the fitting process. 
    use_peak : bool
        If True, central node of initial guess is positioned at the peak.

    Returns 
    -----------
    GP_params : list
        Best-fit GP parameters. 
    result : scipy.optimize.OptimizeResult
        Result of the minimization
    """

    zmin, zmax = z_bounds
    x_init_all, y_init_all = compute_cdf_nodes(z_grid, {1: nz}, M=M, qmin=qmin, qmax=qmax, use_peak = use_peak)
    x_init = x_init_all[0]
    y_init = y_init_all[0]
    p0 = np.concatenate([x_init, np.log(y_init), [np.log(ell_init)]]) # initial param vector

    def objective(p): # loss function
        x_nodes = p[:M]
        log_y = p[M:2*M]
        log_ell = p[2*M]
        y_nodes = np.exp(log_y)
        ell = np.exp(log_ell)
        nz_gp = reconstruct_per_bin(z_pred=z_grid, z_nodes=x_nodes, theta_nodes=y_nodes, z_bounds=z_bounds, corr_len=ell,
            amp=amp, alpha=alpha, kernel_class=kernel_class, nu=nu, return_std=False)

        # Integrated squared difference
        loss = np.trapezoid((nz - nz_gp)**2, z_grid)
        return loss

    # Determining the bounds in between which the minimiser will explore
    x_bounds = [(zmin + 1e-5, zmax - 1e-5) for _ in range(M)]
    y_max = max(np.max(nz) * 10.0, 1e-5)
    y_bounds = [(np.log(1e-8), np.log(y_max)) for _ in range(M)]
    ell_bounds = [(np.log(1e-3), np.log(0.5))]
    bounds = x_bounds + y_bounds + ell_bounds

    # Enforcing ordering on the x-nodes
    constraints = []
    min_dx=1e-3
    for i in range(M - 1):
        constraints.append({"type": "ineq", "fun": lambda p, i=i: p[i + 1] - p[i] - min_dx})

    # Mimimising the loss function
    result = minimize(objective, p0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": maxiter, "ftol": 1e-10, "disp": False})

    # Converting back to original parameters
    p = result.x
    x_nodes = p[:M]
    y_nodes = np.exp(p[M:2*M])
    ell = np.exp(p[2*M])

    GP_params = np.concatenate([x_nodes, y_nodes, [ell]])

    return GP_params, result

def fit_params_ens(z_grid, nz_ens, M=5, z_bounds=(0.0, 3.0), ell_init=0.03, amp=0.04, alpha=0.0, kernel_class=RBF, nu=1.5, qmin=0.05,
    qmax=0.95, maxiter=1000, use_peak=False):
    """
    Fit the GP x-nodes, y-nodes, and correlation length to the full ensemble of n(z)'s.
    
    Parameters 
    -----------
    z_grid : array
        Redshift grid.
    nz_ens : list
        Ensemble of n(z)'s.
    M : int
        Number of nodes.
    z_bounds : tuple
        Redshift boundaries.
    ell_init : float
        Initial guess for correlation length.
    amp : float
        Amplitude of constant kernel. 
    alpha : float
        Noise term used in GP.
    kernel_class : type
        Kernel used in GP reconstruction.
    nu : float
        Smoothness term for Matern kernel.
    qmin : float
        Quantile where first node should be positioned.
    qmax : float
        Quantile where last node should be positioned.
    maxiter : int
        Maximum number of iterations in the fitting process. 
    use_peak : bool
        If True, central node of initial guess is positioned at the peak
    
    Returns 
    -----------
    GP_params : list
        Best-fit GP parameters for each realization in the ensemble. 
    result : scipy.optimize.OptimizeResult
        Result of the minimization for each realization in the ensemble 
    """
    GP_params_list = []
    results = []
    for k, nz in enumerate(nz_ens):
        print(f"Fitting realisation {k+1}/{len(nz_ens)}")
        params, result = fit_gp_params(z_grid=z_grid, nz=nz, M=M, z_bounds=z_bounds, ell_init=ell_init, amp=amp, alpha=alpha, 
                                                 kernel_class=kernel_class, nu=nu, qmin=qmin, qmax=qmax, maxiter=maxiter, use_peak=use_peak)
        GP_params_list.append(params)
        results.append(result)
    GP_params = np.stack(GP_params_list, axis=0)
    return GP_params, results

def recon_ens_from_params(z_grid, GP_params, z_bounds=(0.0, 3.0), amp=0.04, alpha=1e-8, kernel_class=RBF, nu=1.5):
    """
    Reconstruct the ensemble from the best-fit GP parameters.
    
    Parameters 
    -----------
    z_grid : array
        Redshift grid.
    GP_params : list
        Best-fit GP params for each realization in the ensemble.
    z_bounds : tuple
        Redshift boundaries.
    amp : float
        Amplitude of constant kernel. 
    alpha : float
        Noise term used in GP.
    kernel_class : type
        Kernel used in GP reconstruction.
    nu : float
        Smoothness term for Matern kernel.
    
    Returns 
    -----------
    nz_gp_ens : ndarray
        Ensemble of reconstructed redshift distributions. Shape (N_ens, N_z).
    """
    N_ens = GP_params.shape[0]
    M = (GP_params.shape[1] - 1) // 2
    nz_gp_ens = np.zeros((N_ens, len(z_grid)))
    for k in range(N_ens):
        x_nodes = GP_params[k, :M]
        y_nodes = GP_params[k, M:2*M]
        ell = GP_params[k, 2*M]
        nz_gp_ens[k] = reconstruct_per_bin(z_pred=z_grid, z_nodes=x_nodes, theta_nodes=y_nodes, z_bounds=z_bounds,
                                                           corr_len=ell, amp=amp, alpha=alpha, kernel_class=kernel_class, nu=nu)
    return nz_gp_ens