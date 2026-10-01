import numpy as np
from tqdm import tqdm
import torch 
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform

def norm_flow_per_bin(GP_params, n_layers=5, hidden_features=32, lr=1e-3, steps=3000, batch_size=128):
    """
    Setup and train a normalising flow for a single bin

    Parameters 
    -----------
    GP_params : array
        Array containing best-fit GP parameters for each realization in the ensemble 
        Shape (N_ens, N_params)
    n_layers : int
        Number of transformation layers in normalising flow
    hidden_features : int
        Number of hidden features in each layer
    lr : float
        Learning rate used by normalising flow
    steps : int
        Number of steps to train the normalising flow
    batch_size : int
        Number of training samples per step

    Returns 
    ------------
    flow : Flow
        Trained normalising flow for a single bin.
    mean : array
        Mean of each GP parameter used for standardization.
    std : array
        Standard deviation of each GP parameter used for standardization.
    """
    U = GP_params.copy() # for 3 training nodes (i.e. M=3), U has shape (N_ens, 7)
    mean = U.mean(axis=0)
    std = U.std(axis=0) + 1e-6
    U_norm = (U - mean) / std

    train_data = torch.tensor(U_norm, dtype=torch.float32)

    n_params = U_norm.shape[1]
    base = StandardNormal(shape=[n_params]) # all flows begin as standard Gaussian vectors

    transforms = [] # here, invertible transformations will be stored
    for _ in range(n_layers): # autoregressive layers
        transforms.append(MaskedAffineAutoregressiveTransform(features=n_params, hidden_features=hidden_features)) # here, each layer learns a transformation

    transform = CompositeTransform(transforms) # combine layers into a single transformation
    flow = Flow(transform, base) # start with Gaussian base and apply transformation to get log_prob(y)

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr) # optimize flow with Adam
    pbar = tqdm(range(steps), desc="Training flow", ncols=100)

    for _ in pbar:
        batch_idx = torch.randint(0, train_data.shape[0], (min(batch_size, train_data.shape[0]),))
        batch = train_data[batch_idx]
        train_loss = -flow.log_prob(batch).mean()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    return flow, mean, std

def get_flow_log_prob(flow, mean, std, x_nodes, y_nodes, ell):
    """
    Get log probability according to flow based on current set of GP params.

    Parameters 
    -----------
    flow : Flow
        Trained normalising flow for a single bin.
    mean : array
        Mean of each GP parameter used for standardization.
    std : array
        Standard deviation of each GP parameter used for standardization.
    x_nodes : list
        List of x-positions of nodes in single bin.
    y_nodes : list
        List of y-positions of nodes in single bin. 
    ell : float
        Correlation length. 

    Returns 
    ------------
    log_prob : float
        Log probability according to flow. 
    """
    GP_params = np.concatenate([x_nodes, y_nodes, [ell]])
    GP_params_norm = (GP_params - mean) / std
    GP_params_tensor = torch.tensor(GP_params_norm, dtype=torch.float32).unsqueeze(0)
    log_prob = flow.log_prob(GP_params_tensor).item()
    return log_prob

def build_flow(M, n_layers=5, hidden_features=32):
    """
    Build normalising flow structure.
    We need this when loading in the trained normalising flow in the sampling script. 
    Should use exactly the same architecture as was used in norm_flow_per_bin

    Parameters
    -----------
    M : int
        Number of nodes.
    n_layers : int
        Number of layers in the normalising flow.
    hidden_features: int
        Number of hidden features in the normalising flow.  
    
    Returns 
    ------------
    Flow : Flow
        Flow object with defined number of layers and hidden features.
    """
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