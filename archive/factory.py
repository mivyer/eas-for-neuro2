# models/factory.py

import numpy as np
from models.rsnn_policy import RSNNPolicy

def policy_builder_from_connectivity(connectivity: np.ndarray,
                                     obs_dim: int = 0,
                                     action_dim: int = 1) -> RSNNPolicy:
    N = connectivity.shape[0]
    W_rec = connectivity * 0.1

    W_in = None
    if obs_dim > 0:
        W_in = 0.1 * np.random.randn(N, obs_dim).astype(np.float32)

    W_out = 0.1 * np.random.randn(action_dim, N).astype(np.float32)

    return RSNNPolicy(W_rec, W_in, W_out)
