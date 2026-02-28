# models/rsnn_policy.py
"""
Simple rate-coded RNN policy for EA/ES/GA evaluation.
Self-contained — no external dependencies beyond numpy.

Interface:
    policy = RSNNPolicy(W_rec, W_in, W_out)
    policy.reset()
    action = policy.act(obs)
"""

import numpy as np


class RSNNPolicy:
    """
    Rate-coded recurrent neural network.
    h[t] = tanh(W_rec @ h[t-1] + W_in @ obs[t])
    action[t] = tanh(W_out @ h[t])
    """

    def __init__(self, W_rec, W_in, W_out):
        self.W_rec = W_rec.astype(np.float32)
        self.W_in = W_in.astype(np.float32)
        self.W_out = W_out.astype(np.float32)
        self.N = W_rec.shape[0]
        self.h = np.zeros(self.N, dtype=np.float32)

    def reset(self):
        self.h = np.zeros(self.N, dtype=np.float32)

    def act(self, obs):
        """obs: (obs_dim,) → action: (action_dim,)"""
        self.h = np.tanh(self.W_rec @ self.h + self.W_in @ obs)
        return np.tanh(self.W_out @ self.h).astype(np.float32)