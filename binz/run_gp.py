import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy import integrate
import seaborn as sns


def run_gp(dndz, z, z_pred, ell):
    gp_priors = {}
    gp_predictions = {}
    training_nodes = {}
    z_min, z_max = 0.0, 3.0

    for key, nz in dndz.items():
        # --- CDF & median node ---
        cdf = integrate.cumulative_trapezoid(nz, z, initial=0)
        cdf /= cdf[-1]
        z_med = np.interp(0.5, cdf, z)
        n_med = nz[np.argmin(np.abs(z - z_med))]
        kernel = C(0.04, (1e-4, 1.0)) * Matern(length_scale=ell, nu=1.5, length_scale_bounds="fixed")

        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False, alpha=1e-8)

        # --- Universal boundary anchors + median node (no widths) ---
        X_train = np.array([np.log1p(z_min)/np.log1p(z_max), np.log1p(z_med)/np.log1p(z_max), np.log1p(z_max)/np.log1p(z_max)]).reshape(-1, 1)
        y_train = np.array([0.0, n_med, 0.0])

        gp.fit(X_train, y_train)

        x_pred = np.log1p(z_pred)/np.log1p(z_max)

        y_pred, y_std = gp.predict(x_pred, return_std=True)

        gp_priors[key] = gp
        gp_predictions[key] = (y_pred, y_std)
        training_nodes[key] = (np.array([[z_med]]), np.array([n_med]))
    return gp_priors, gp_predictions, training_nodes