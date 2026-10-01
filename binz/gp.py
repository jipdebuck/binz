import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, RBF
from scipy import integrate

def normalize_dndz(nz_example, z_nz):
    """
    Normalize n(z) in all bins. 

    Parameters 
    -----------
    nz_example : dict
        Dictionary containing n(z) distributions for all bins.
    z_nz : array
        Redshift grid corresponding to n(z) distributions.

    Returns 
    -----------
    normalized : dict
        Dictionary containing the normalized n(z) distributions .
    """
    normalized = {}
    for key in nz_example:
        normalized[key] = nz_example[key] / integrate.trapezoid(nz_example[key], z_nz)
    return normalized

def resample_dndz(nz_example, z_nz, myz):
    """
    Resample n(z) from z_nz grid to myz grid.

    Parameters 
    -----------
    nz_example : dict
        Dictionary containing n(z) distributions for all bins. 
    z_nz : array
        Redshift grid corresponding to originally sampled n(z) distributions.
    myz : array
        Redshift grid to resample n(z) distributions on.
    
    Returns 
    -----------
    my_dndz_norm : array
        Array containing the resampled n(z) distributions.

    """
    my_dndz = np.vstack(list(nz_example.values())) #stacking arrays vertically
    my_dndz_norm = np.zeros([len(nz_example), len(myz)]) #creating an empty array
    for i in range(len(nz_example)):
        my_dndz_norm[i, :] = np.interp(myz, z_nz, my_dndz[i, :]) #interpolating to new myz values
    return my_dndz_norm


def x_log1p(z, zmax):
    """
    Map z to x in [0,1] using log(1+z).

    Parameters 
    -----------
    z : array
        Original redshift grid.
    zmax : float 
        Maximum of the redshift range.

    Returns 
    -----------
    x : array
        Remapped redshift coordinate. 
    """
    x = np.log1p(z) / np.log1p(zmax)
    return x

def compute_cdf_nodes(z_grid, nz_org, M=3, qmin=0.05, qmax=0.95):
    """
    Compute training node positions based on CDF. 

    For M = 1, the node is positioned at the peak.
    For M = 3, the central node is positioned at the peak.
    For other values of M, all nodes are positioned at equally-spaced 
    quantiles between qmin and qmax
    
    Parameters 
    -----------
    z_grid : array
        Redshift grid corresponding to nz_org. 
    nz_org : dict
        Dictionary containing normalized redshift distributions.
    M : int
        Number of training nodes.
    qmin : float
        Quantile where first node should be positioned.
    qmax : float
        Quantile where last node should be positioned. 
    
    Returns 
    -----------
    X_nodes : list
        List containing the redshift positions of the nodes in each bin.
    Y_nodes : list
        List containing the n(z) positions of the nodes in each bin. 
    
    """    
    X_nodes = []
    Y_nodes = []   
    for key, nz in nz_org.items():
        cdf = integrate.cumulative_trapezoid(nz, z_grid, initial=0)
        cdf /= cdf[-1]
        if M == 1 or M == 3:
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
        else:
            quantiles = np.linspace(qmin, qmax, M)
            x_nodes = np.interp(quantiles, cdf, z_grid)
            x_nodes = np.sort(x_nodes)
            y_nodes = np.interp(x_nodes, z_grid, nz)
            X_nodes.append(x_nodes)
            Y_nodes.append(y_nodes)
    return X_nodes, Y_nodes


def kernel_fixed(kernel_class, amp, ell, nu=1.5, amp_bounds=(1e-6, 10.0)):
    """
    Kernel with fixed length-scale ell, fixed (or bounded) amplitude amp.
    We keep optimizer=None, so amp and ell are not learned during GP fitting.

    Parameters 
    -----------
    kernel_class : type
        Kernel class to use for GP. Can be either RBF or Matern. 
    amp : float
        Amplitude of constant kernel.
    ell : float
        Correlation length.
    nu : float
        Smoothness parameter for Matern kernel.
    amp_bounds : tuple of floats
        Lower and upper bounds for constant kernel. 
    
    Returns 
    -----------
    kernel : sklearn.gaussian_process.kernels.Kernel
        GP kernel with the specified amplitude and fixed length scale.    
    """
    if kernel_class is Matern:
        return C(amp, amp_bounds) * Matern(length_scale=ell, nu=nu, length_scale_bounds="fixed")
    else:
        return C(amp, amp_bounds) * RBF(length_scale=ell, length_scale_bounds="fixed")


def reconstruct_per_bin(z_pred, z_nodes, theta_nodes, z_bounds=(0.0, 3.0), corr_len=0.03,
    amp=0.04, alpha=0, kernel_class=RBF, nu=1.5, return_std=False):
    """
    Gaussian process prediction based on training nodes and correlation length for a single bin. 
    Transforms (z, n(z)) --> (x, y(x)) inside function

    Parameters 
    -----------
    z_pred : array
        Redshift grid to make GP predictions on. 
    z_nodes : array
        Redshift training node positions (in z coordiantes, not x).
    theta_nodes : array
        n(z) training node positions in original coordinates.
    z_bounds : tuple
        Redshift boundaries.
    corr_len : float
        Correlation length. 
    amp : float
        Amplitude of constant kernel.
    alpha : float
        Noise term used in GP.
    kernel_class : type
        Kernel class used in GP. Can be either Matern or RBF.
    nu : float
        Smoothness parameter for Matern kernel.
    return_std : bool
        If True, std of GP prediction is returned.
    
    Returns 
    -----------
    y_pred : array
        GP prediction in a single bin.
    y_std : array
        Standard deviation of GP prediction in a single bin.
    gp : GaussianProcessRegressor
        Fitted GP regression model
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

    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False, alpha=float(alpha),
                                  copy_X_train=False, random_state=0)
    gp.fit(X_train, y_train_squareroot)

    if return_std:
        y_pred_squareroot, y_std = gp.predict(X_pred, return_std=True)
        y_pred = y_pred_squareroot**2
        return y_pred, y_std, gp
    else:
        y_pred_squareroot = gp.predict(X_pred, return_std=False)
        y_pred = y_pred_squareroot**2
        return y_pred

def reconstruct_all_bins(z_pred, z_nodes_all_bins, n_nodes_all_bins, z_bounds=(0.0, 3.0), corr_len=0.03, corr_lens=None, amp=0.04,
    alpha=0.0, kernel_class=RBF, nu=1.5, return_std=False, single_corr=True):
    """
    Reconstruct all bins at once.
    If return_std=True, returns list of (mean,std) per bin.
    If single_corr=True, use single correlation length globally. 
    Else use correlation length per bin in list corr_lens

    Parameters 
    -----------
    z_pred : array
        Redshift grid to make GP predictions on. 
    z_nodes_all_bins : list of arrays
        Redshift values at node positions (in z coordiantes, not x) in all bins.
    n_nodes_all_bins : list of arrays
        n(z) values at node positions in original coordinates in all bins.
    z_bounds : tuple
        Redshift boundaries.
    corr_len : float
        Single global correlation length. 
    corr_lens : list
        Correlation length for each bin. Used when single_corr is False
    amp : float
        Amplitude of constant kernel.
    alpha : float
        Noise term used in GP.
    kernel_class : type
        Kernel class used in GP. Can be either Matern or RBF.
    nu : float
        Smoothness parameter for Matern kernel.
    return_std : bool
        If True, std of GP prediction is returned.
    single_corr : bool
        If True, use single global correlation length. Else use correlation length per bin
    
    Returns 
    -----------
    out : dict
        Dictionary containing GP prediction per bin.
        If return_std = True, out contains tuples of (nz, std)
    
    """
    
    out = {}

    for b, z_node in enumerate(z_nodes_all_bins):
        if single_corr:
            ell = corr_len
        else:
            ell = corr_lens[b]
        if return_std:
            mean, std, _ = reconstruct_per_bin(z_pred=z_pred, z_nodes=z_node, theta_nodes=n_nodes_all_bins[b],
                z_bounds=z_bounds, corr_len=ell, amp=amp, alpha=alpha, kernel_class=kernel_class,
                nu=nu, return_std=True)
            out[b+1] = (mean, std)
        else:
            nz = reconstruct_per_bin(z_pred=z_pred, z_nodes=z_node, theta_nodes=n_nodes_all_bins[b],
                z_bounds=z_bounds, corr_len=ell, amp=amp, alpha=alpha, kernel_class=kernel_class, nu=nu,
                return_std=False)
            out[b+1] = nz
    return out


def resample_dndz_gp(gp_predictions, z_pred, zs):
    """
    Resample GP prediction from z_pred to zs grid *and* renormalise.
    Currently only works on output of reconstruct_all_bins() with return_std = False.

    Parameters 
    -----------
    gp_predictions : dict
        Dictionary containing GP n(z) predictions in each bin.
    z_pred : array
        Redshift grid corresponding to gp_predictions.
    zs : array
        Redshift grid on which to evaluate gp_predictions.

    Returns
    -----------
    my_dndz_norm : ndarray
        Resampled GP predictions for each bin. 
    """
    z_pred = z_pred.reshape(-1, 1)
    gp_predictions_norm = {}
    for key in gp_predictions.keys():
        gp_predictions_norm[key] = gp_predictions[key]/integrate.trapezoid(gp_predictions[key], z_pred[:,0])
    my_dndz = np.vstack(list(gp_predictions_norm.values())) 
    my_dndz_norm = np.zeros([len(gp_predictions_norm), len(zs)])
    for i in range(len(gp_predictions_norm)):
        my_dndz_norm[i, :] = np.interp(zs, z_pred[:,0], my_dndz[i, :])
    return my_dndz_norm


