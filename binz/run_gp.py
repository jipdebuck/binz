import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C


def build_gp_nz(n_med, z_med, corr_len, z_bounds, z_pred, alpha, kernel_class, nu, const_value, const_bounds):
    gp_priors = {}
    gp_predictions = {}
    for i, nz in enumerate(n_med):
        if kernel_class == Matern:
            kernel = C(const_value, const_bounds) * kernel_class(length_scale=corr_len, nu=nu, length_scale_bounds="fixed")
        else:
            kernel = C(const_value, const_bounds) * kernel_class(length_scale=corr_len, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False, alpha=alpha)
        X_train = np.array([np.log1p(z_bounds[0])/np.log1p(z_bounds[1]), np.log1p(z_med[i])/np.log1p(z_bounds[1]), np.log1p(z_bounds[1])/np.log1p(z_bounds[1])]).reshape(-1, 1)
        y_train = np.array([0.0, nz, 0.0])
        gp.fit(X_train, y_train)
        x_pred = np.log1p(z_pred)/np.log1p(z_bounds[1])
        x_pred = np.array(x_pred).reshape(-1, 1)
        y_pred, y_std = gp.predict(x_pred, return_std=True)
        gp_priors[i] = gp
        gp_predictions[i] = (y_pred, y_std)
    return gp_priors, gp_predictions